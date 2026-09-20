"""
Naive implementaiton, But it does not really save param memory added by Autograd
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear


SHARDED_MODULE_TYPES = (Embedding, Linear)


def is_shardable(module: torch.nn.Module):
    return isinstance(module, SHARDED_MODULE_TYPES)


def use_compute_dtype(module: torch.nn.Module) -> bool:
    return isinstance(module, Linear)


def free(data: torch.Tensor | None):
    if data is None:
        return
    if data.untyped_storage().size() > 0:
        data.untyped_storage().resize_(0)


def get_pad_size(size: int, world_size: int) -> int:
    return (world_size - (size % world_size)) % world_size


@dataclass
class ParamState:
    # static
    is_shardable: bool
    shape: torch.Size | None = None
    numel: int = 0
    pad_size: int = 0
    shard_size: int = 0

    # dynamic
    sharded_data: torch.Tensor | None = None
    all_gather_handle = None
    all_gather_data: torch.Tensor | None = None  # should be compute_dtype
    grad_handle = None
    flat_grad: torch.Tensor | None = None


@dataclass
class ModuleInfo:
    """For sharded Modules"""

    first_fwd: bool = True
    first_bwd: bool = True
    fwd_prefetch = None
    bwd_prefetch = None


class FSDPModule(torch.nn.Module):
    def __init__(
        self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.module = module
        self.storage_dtype = torch.float32  # hardcode now
        self.compute_dtype = compute_dtype

        if not dist.is_initialized():
            raise RuntimeError("torch distributed not initialized")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sync_module()
        self.shard_params()
        self.attach_fsdp_hooks()

    def sync_module(self):
        self.rank0_print("Syncing module...")
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def shard_params(self):
        """
        Build ParamStates for each shardable param
        """
        self.rank0_print("Sharding params...")
        self.param_states: dict[torch.nn.Parameter, ParamState] = {}
        for module in self.module.modules():
            if is_shardable(module):
                for pname, param in module.named_parameters(recurse=False):
                    orig_size = param.untyped_storage().size()
                    # flat & padding
                    # orig_data = param.data
                    flat_data = param.data.detach().flatten()
                    pad_size = get_pad_size(
                        size=flat_data.numel(), world_size=self.world_size
                    )
                    if pad_size > 0:
                        flat_data = torch.cat(
                            [
                                flat_data,
                                torch.zeros(
                                    pad_size,
                                    dtype=flat_data.dtype,
                                    device=flat_data.device,
                                ),
                            ]
                        )

                    # sharding
                    shard_size = flat_data.numel() // self.world_size
                    self.param_states[param] = ParamState(
                        is_shardable=True,
                        shape=param.shape,
                        numel=param.numel(),
                        pad_size=pad_size,
                        shard_size=shard_size,
                    )
                    start_idx = self.rank * shard_size
                    end_idx = start_idx + shard_size
                    sharded_data = flat_data[start_idx:end_idx].clone()

                    # swap data and release
                    param.data = sharded_data
                    free(flat_data)
                    # free(orig_data)

                    after_size = param.untyped_storage().size()
                    print(f"Size comparison {pname=}  {orig_size=}  {after_size=}")
            else:
                for param in module.parameters(recurse=False):
                    self.param_states[param] = ParamState(is_shardable=False)

    def gather_param(
        self, mod: torch.nn.Module, param: torch.nn.Parameter, async_op: bool
    ):
        """
        Store all gathered param's data on ParamState.data
        """
        ps = self.param_states[param]
        dtype = self.compute_dtype if use_compute_dtype(mod) else param.dtype
        ps.all_gather_data = torch.empty(
            self.world_size * ps.shard_size, dtype=dtype, device=param.device
        )
        dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=async_op)

    def attach_fsdp_hooks(self):
        self.module_infos: dict[torch.nn.Module, ModuleInfo] = {}
        self.fwd_modules = []
        self.bwd_modules = []
        self.fwd_prefetches = []  # use in top forward

        for mod in self.module.modules():
            if is_shardable(mod):
                if mod not in self.module_infos:
                    self.module_infos[mod] = ModuleInfo()

                ####### pre foward
                def make_fwd_pre(dt):
                    def hook(mod, inp):
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            # wait all_gather handle
                            if ps.all_gather_handle is not None:
                                ps.all_gather_handle.wait()
                                ps.all_gather_handle = None
                            else:  # If missing, do all_gather on params
                                self.gather_param(mod, param, async_op=False)
                            # ps.all_gather_data needs unpad and unflaten
                            ps.sharded_data = param.data
                            param.data = ps.all_gather_data  # compute_dtype now
                            assert param.data is not None
                            assert ps.shape is not None
                            param.data = param.data[: ps.numel].reshape(ps.shape)
                            ps.all_gather_data = None

                    return hook

                mod.register_forward_pre_hook(make_fwd_pre(self.compute_dtype))

                ####### post forward
                def make_fwd_post():
                    def hook(m, inp, out):
                        # swap param.data, and release orig data
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            orig_data = param.data
                            param.data = ps.sharded_data  # storage_type now
                            ps.sharded_data = None
                            # free(orig_data)  # TODO BUG: cannot release memory due to Autograd

                        mi = self.module_infos[m]
                        if mi.first_fwd:  # record for first fwd
                            mi.first_fwd = False
                            idx = len(self.fwd_modules)
                            if idx > 1:
                                self.module_infos[
                                    self.fwd_modules[idx - 2]
                                ].fwd_prefetch = m
                            else:
                                self.fwd_prefetches.append(m)
                            self.fwd_modules.append(m)
                        else:
                            # prefetch
                            pmod = mi.fwd_prefetch
                            if pmod is not None:
                                for param in pmod.parameters(recurse=False):
                                    self.gather_param(pmod, param, async_op=True)

                    return hook

                mod.register_forward_hook(make_fwd_post())

                ####### pre backward
                def make_bwd_pre(dt):
                    def hook(m, grad_output):
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            # wait all_gather handle
                            if ps.all_gather_handle is not None:
                                ps.all_gather_handle.wait()
                                ps.all_gather_handle = None
                            else:  # If missing, do all_gather on params
                                self.gather_param(mod, param, async_op=True)

                            # ps.all_gather_data needs unpad and unflaten
                            ps.sharded_data = param.data
                            param.data = ps.all_gather_data
                            assert param.data is not None
                            assert ps.shape is not None
                            param.data = param.data[: ps.numel].reshape(ps.shape)
                            ps.all_gather_data = None

                    return hook

                mod.register_full_backward_pre_hook(make_bwd_pre(self.compute_dtype))

                ####### post backward
                def make_bwd_post(dt):
                    def hook(m, inp, out):
                        """Only do record and prefetch. Sharding is done in grad hook"""
                        mi = self.module_infos[m]
                        if mi.first_bwd:  # record for first bwd
                            mi.first_bwd = False
                            idx = len(self.bwd_modules)
                            if idx > 1:
                                self.module_infos[
                                    self.bwd_modules[idx - 2]
                                ].bwd_prefetch = m
                            self.bwd_modules.append(m)
                        else:
                            # prefetch
                            pmod = mi.bwd_prefetch  # prefetch module
                            if pmod is not None:
                                for param in pmod.parameters(recurse=False):
                                    self.gather_param(pmod, param, async_op=True)

                    return hook

                mod.register_full_backward_hook(make_bwd_post(self.compute_dtype))

                def make_grad_hook():
                    def hook(p):
                        ps = self.param_states[p]

                        # TODO: shape mismatch
                        # swap param.data, and release orig data
                        orig_data = p.data
                        p.data = ps.sharded_data
                        ps.sharded_data = None
                        free(orig_data)

                        # reduce scatter gradient
                        # grad need to be storage_dtype
                        grad = p.grad.to(
                            self.storage_dtype
                        )  # TODO: no-op when storage_dtype == compute_dtype
                        grad.div_(self.world_size)

                        ps.flat_grad = grad.detach().flatten()
                        if ps.pad_size > 0:
                            ps.flat_grad = torch.cat(
                                [
                                    ps.flat_grad,
                                    torch.zeros(
                                        ps.pad_size,
                                        dtype=grad.dtype,
                                        device=grad.device,
                                    ),
                                ]
                            )
                        p.grad = torch.empty(
                            ps.shard_size, dtype=grad.dtype, device=grad.device
                        )  # need to match p.data shape!

                        ps.grad_handle = dist.reduce_scatter_tensor(
                            output=p.grad,
                            input=ps.flat_grad,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )
                        # ps.flat_grad = flat_grad

                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(make_grad_hook())

            else:
                # Non-sharded module -- only need grad hook for all_reduce
                def make_grad_hook():
                    def hook(p):
                        p.grad.div_(self.world_size)
                        self.param_states[p].grad_handle = dist.all_reduce(
                            p.grad, op=dist.ReduceOp.SUM, async_op=True
                        )

                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(make_grad_hook())

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    def forward(self, *inputs, **kwargs):
        # prefetch first two layers
        for pm in self.fwd_prefetches:  # must be shardable: (Linear, Embedding)
            for p in mod.parameters(recurse=False):
                pps = self.param_states[p]
                self.compute_dtype
                # TODO: check
                # ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                if isinstance(pm, Linear):
                    pps.all_gather_data = torch.empty(
                        self.world_size * ps.shard_size,
                        dtype=self.compute_dtype,
                        device=p.device,
                    )
                    pps.all_gather_handle = dist.all_gather_into_tensor(
                        ps.all_gather_data, p.data.to(self.compute_dtype), async_op=True
                    )
                else:  # Embedding
                    pps.all_gather_data = torch.empty(
                        self.world_size * pps.shard_size, dtype=p.dtype, device=p.device
                    )
                    pps.all_gather_handle = dist.all_gather_into_tensor(
                        pps.all_gather_data, p.data, async_op=True
                    )  # async

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param, ps in self.param_states.items():
            if ps.grad_handle is not None:
                ps.grad_handle.wait()
                ps.grad_handle = None
                assert param.grad is not None
                # param.grad = param.grad.to(self.storage_dtype)  # TODO: check
                free(ps.flat_grad)
                ps.flat_grad = None
                # free(ps.orig_grad)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        res = {}
        for name, param in self.module.named_parameters():
            ps = self.param_states[param]
            if ps.is_shardable:
                # all_gather
                gathered_data = torch.empty(
                    self.world_size * ps.shard_size,
                    dtype=param.dtype,
                    device=param.device,
                )
                dist.all_gather_into_tensor(
                    gathered_data, param.data, async_op=False
                )  # sync
                assert ps.shape is not None
                res[name] = gathered_data[: ps.numel].reshape(ps.shape)
            else:
                res[name] = param.data
        return res
