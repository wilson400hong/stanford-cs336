"""
FSDP implementation
"""

from dataclasses import dataclass
import torch
import torch.distributed as dist
from typing import TypeVar, Optional

from cs336_basics.model import Embedding, Linear


T = TypeVar("T")

def none_throws(val: Optional[T], message: str = "Unexpected None value") -> T:
    if val is None:
        raise ValueError(message)
    return val


SHARDED_MODULE_TYPES = (Embedding, Linear)


def is_shardable(m: torch.nn.Module) -> bool:
    return isinstance(m, SHARDED_MODULE_TYPES)


def use_compute_dtype(m: torch.nn.Module) -> bool:
    return isinstance(m, (Linear, Embedding))


def free_tensor(t: torch.Tensor | None):
    if t is not None and t.untyped_storage().size() > 0:
        t.untyped_storage().resize_(0)


def pad_tensor(t: torch.Tensor, pad_size: int) -> torch.Tensor:
    return (
        torch.cat([t, torch.zeros(pad_size, dtype=t.dtype, device=t.device)])
        if pad_size > 0
        else t
    )
    

@dataclass
class ParamState:
    # Static
    is_shardable: bool
    shape: torch.Size
    dtype: torch.dtype                  # storage dtype
    compute_dtype: torch.dtype          # comm/compute dtype (bfloat16, float16)
    numel: int = 0
    pad_size: int = 0
    shard_size: int = 0

    # Shared storage for unsharded tensor, used for comm/compute and autograd backward.  
    # They can be resized (to 0 memory usage) but never released.
    unsharded_flat: torch.Tensor | None = None      # flat and padded, compute_dtype
    unsharded_view: torch.Tensor | None = None      # unsharded view, compute_dtype
    # sharded tensor
    sharded_data: torch.Tensor | None = None        # sharded, dtype

    # Comm handle and buffer
    all_gather_handle: dist.Work | None = None
    grad_handle: dist.Work | None = None
    grad_input_buf: torch.Tensor | None = None
    reduced_grad_shard: torch.Tensor | None = None


class FSDPModule(torch.nn.Module):
    def __init__(
        self, module: torch.nn.Module, 
        compute_dtype: torch.dtype | None = None
    ):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch distributed not initialized")

        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.rank0_print("Broadcasting module params and buffer...")
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0)

        self.param_states: dict[torch.nn.Parameter, ParamState] = {}

        # Dynamic forward/backward ordering
        self.has_recorded_order = False
        self.fwd_execution_order: list[torch.nn.Module] = []
        self.next_fwd_mod: dict[torch.nn.Module, torch.nn.Module] = {}
        self.next_bwd_mod: dict[torch.nn.Module, torch.nn.Module] = {}
                
        self._shard_params()
        self._attach_hooks()


    def _shard_params(self):
        """
        Build ParamStates for each shardable param
        """
        self.rank0_print("Sharding params...")

        for mod in self.module.modules():
            if is_shardable(mod):
                for p in mod.parameters(recurse=False):
                    flat = p.data.detach().flatten()
                    pad_size = (self.world_size - (flat.numel() % self.world_size)) % self.world_size
                    flat = pad_tensor(flat, pad_size)
                    shard_size = flat.numel() // self.world_size
                    comp_dtype = (
                        self.compute_dtype
                        if (use_compute_dtype(mod) and self.compute_dtype is not None)
                        else p.dtype
                    )

                    # storage
                    sharded = flat[self.rank*shard_size: (self.rank+1)*shard_size].clone() 
                    unsharded_flat = torch.empty(flat.numel(), dtype=comp_dtype, device=p.device)
                    unsharded_view = unsharded_flat[: p.numel()].view(p.shape)  # share data with unsharded_flat
                    free_tensor(unsharded_flat)  # resize to 0 to free memory

                    self.param_states[p] = ParamState(
                        is_shardable=True,
                        shape=p.shape,
                        dtype=p.dtype,
                        compute_dtype=comp_dtype,
                        numel=p.numel(),
                        pad_size=pad_size,
                        shard_size=shard_size,
                        unsharded_flat=unsharded_flat,
                        unsharded_view=unsharded_view,
                        sharded_data=sharded
                    )
                    p.data = sharded
                    free_tensor(flat)
            else:
                for p in mod.parameters(recurse=False):
                    self.param_states[p] = ParamState(
                        is_shardable=False,
                        shape=p.shape,
                        dtype=p.dtype,
                        compute_dtype=p.dtype
                    )



    def _gather_module(self, mod: torch.nn.Module, async_op: bool):
        """All-Gather all params of a module to original shape, and convert to compute_dtype"""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            if ps.all_gather_handle is None and ps.unsharded_flat is not None and ps.unsharded_flat.untyped_storage().size() == 0:
                ps.unsharded_flat.untyped_storage().resize_(
                    self.world_size * ps.shard_size * ps.unsharded_flat.element_size()
                )
                ps.all_gather_handle = dist.all_gather_into_tensor(ps.unsharded_flat, p.data.to(ps.compute_dtype), async_op=async_op)


    def _wait_and_assign_module(self, mod: torch.nn.Module):
        """Wait async and point param.data to unsharded_view"""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            if ps.all_gather_handle is not None:
                ps.all_gather_handle.wait()
                ps.all_gather_handle = None
            p.data = none_throws(ps.unsharded_view)


    def _release_module(self, mod: torch.nn.Module):
        """Swap param.data back to sharded_data and free memory"""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            p.data = none_throws(ps.sharded_data)
            free_tensor(ps.unsharded_flat)




    def _attach_hooks(self):
        for mod in self.module.modules():
            if is_shardable(mod):
                # ----------------- 1. Forward Pre Hook -----------------
                def fwd_pre(m, inp):
                    if list(m.parameters(recurse=False)):
                        self._gather_module(m, async_op=False)
                        self._wait_and_assign_module(m)
                    
                    # Record ordering for first forward
                    if not self.has_recorded_order:
                        self.fwd_execution_order.append(m)
                    else:
                        # Prefetch for second forward
                        nxt = self.next_fwd_mod.get(m)
                        if nxt is not None:
                            self._gather_module(nxt, async_op=True)

                mod.register_forward_pre_hook(fwd_pre)

                # ----------------- 2. Forward Post Hook -----------------
                def fwd_post(m, inp, out):
                    self._release_module(m)

                mod.register_forward_hook(fwd_post)

                # ----------------- 3. Backward Pre Hook -----------------
                def bwd_pre(m, grad_out):
                    self._gather_module(m, async_op=False)
                    self._wait_and_assign_module(m)

                    # Backward prefetch
                    if self.has_recorded_order:
                        nxt: torch.nn.Module | None = self.next_bwd_mod.get(m)
                        if nxt is not None:
                            self._gather_module(nxt, async_op=True)

                mod.register_full_backward_pre_hook(bwd_pre)

                # ----------------- 4. Post Accumulate Grad Hook -----------------
                def make_grad_hook():
                    def hook(p: torch.nn.Parameter):
                        ps = self.param_states[p]
                        raw_grad = none_throws(p.grad)
                        p.grad = None  # defensive: p.grad need to has same shape as p.data when it is not None
                        p.data = none_throws(ps.sharded_data)
                        free_tensor(ps.unsharded_flat)

                        flat_grad = pad_tensor(raw_grad.to(ps.dtype).detach().flatten(), ps.pad_size)
                        flat_grad.div_(self.world_size)
                        del raw_grad

                        ps.grad_input_buf = flat_grad
                        ps.reduced_grad_shard = torch.empty(ps.shard_size, dtype=ps.dtype, device=p.device)

                        ps.grad_handle = dist.reduce_scatter_tensor(
                            output=ps.reduced_grad_shard,
                            input=ps.grad_input_buf,
                            op=dist.ReduceOp.SUM,
                            async_op=True
                        )
                    return hook

                for p in mod.parameters(recurse=False):
                    if p.requires_grad:
                        p.register_post_accumulate_grad_hook(make_grad_hook())

            else:
                # Non-sharded module -- only need grad hook for all_reduce
                def make_grad_hook():
                    def hook(p):
                        p.grad.div_(self.world_size)
                        self.param_states[p].grad_handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                    return hook

                for p in mod.parameters(recurse=False):
                    if p.requires_grad:
                        p.register_post_accumulate_grad_hook(make_grad_hook())

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    def forward(self, *inputs, **kwargs):
        if self.has_recorded_order and self.fwd_execution_order:
            self._gather_module(self.fwd_execution_order[0], async_op=True)
        
        output = self.module(*inputs, **kwargs)

        # derive fwd / bwd ordering
        if not self.has_recorded_order:
            for mod, nxt in zip(self.fwd_execution_order[:-1], self.fwd_execution_order[1:]):
                self.next_fwd_mod[mod] = nxt
                self.next_bwd_mod[nxt] = mod
            self.has_recorded_order = True
       
        return output

    def finish_gradient_synchronization(self):
        """Wait all gradient hooks done"""
        for p, ps in self.param_states.items():
            if ps.grad_handle is not None:
                ps.grad_handle.wait()
                ps.grad_handle = None

                # Gradient accumulation
                if ps.is_shardable:
                    if p.grad is None:
                        p.grad = ps.reduced_grad_shard
                    else:
                        p.grad.add_(none_throws(ps.reduced_grad_shard))

                ps.reduced_grad_shard = None
                free_tensor(ps.grad_input_buf)
                ps.grad_input_buf = None
               

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Collect all ranks' params and restore the original shape"""
        res = {}
        for name, p in self.module.named_parameters():
            ps = self.param_states[p]
            if ps.is_shardable:
                # all_gather
                full_data = torch.empty(self.world_size * ps.shard_size, dtype=p.dtype, device=p.device)
                dist.all_gather_into_tensor(full_data, p.data, async_op=False)  # sync
                res[name] = full_data[: ps.numel].reshape(ps.shape)
            else:
                res[name] = p.data
        return res
