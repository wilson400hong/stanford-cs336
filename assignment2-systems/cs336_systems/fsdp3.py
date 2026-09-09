"""
TODO
FSDP implementation usees save_tensor_hooks
"""

from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn.functional as F
from collections import defaultdict
# Placeholder imports based on your code
# from cs336_basics.model import Embedding, Linear, RMSNorm
# SHARDED_MODULE_TYPES = (Embedding, Linear)

SHARDED_MODULE_TYPES = (torch.nn.Embedding, torch.nn.Linear)  # Adjusted for generic run


def is_shardable(module: torch.nn.Module):
    return isinstance(module, SHARDED_MODULE_TYPES)


@dataclass
class ParamState:
    """For all params"""

    # static
    is_shardable: bool
    shape: torch.Size | None = None
    numel: int = 0
    pad_size: int = 0
    shard_size: int = 0

    # dynamic
    sharded_data: torch.Tensor | None = None
    all_gather_handle: Any = None
    all_gather_data: torch.Tensor | None = None
    grad_handle: Any = None

    flat_grad: torch.Tensor | None = None
    orig_grad: torch.Tensor | None = None


class ModuleInfo:
    """For FSDP Modules"""

    def __init__(self):
        self.first_fwd: bool = True
        self.first_bwd: bool = True
        self.fwd_prefetch: torch.nn.Module | None = None
        self.bwd_prefetch: torch.nn.Module | None = None


class FSDPModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()

        self.module = module
        self.compute_dtype = compute_dtype

        if not dist.is_initialized():
            raise RuntimeError("torch distributed not initialized")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.sync_module()
        self.shard_params()

        # Map parameter Python IDs back to the actual parameter object for autograd hooks
        self._param_id_to_param = {id(p): p for p in self.param_states.keys()}

        self.attach_fsdp_hooks()

    def sync_module(self):
        self.rank0_print("Syncing module, broadcasting params...")
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def shard_params(self):
        """Build ParamStates for each shardable param"""
        self.param_states: dict[torch.nn.Parameter, ParamState] = {}
        for module in self.module.modules():
            if is_shardable(module):
                for param in module.parameters(recurse=False):
                    orig_data = param.data
                    flat_data = orig_data.detach().flatten()

                    # Pad
                    pad_size = (self.world_size - (flat_data.numel() % self.world_size)) % self.world_size
                    if pad_size > 0:
                        flat_data = F.pad(flat_data, (0, pad_size))

                    # Shard
                    shard_size = flat_data.numel() // self.world_size
                    start_idx = self.rank * shard_size
                    end_idx = start_idx + shard_size

                    # Clone creates a new leaf tensor.
                    sharded_data = flat_data[start_idx:end_idx].clone()

                    self.param_states[param] = ParamState(is_shardable=True, shape=param.shape, numel=param.numel(), pad_size=pad_size, shard_size=shard_size)

                    param.data = sharded_data

                    # Automatically freed by Python GC / PyTorch Caching Allocator
                    del orig_data, flat_data
            else:
                for param in module.parameters(recurse=False):
                    self.param_states[param] = ParamState(is_shardable=False)

    # =========================================================================
    # AUTOGRAD MEMORY HOOKS
    # =========================================================================
    def _pack_hook(self, tensor):
        tid = id(tensor)
        if tid in self._param_id_to_param:
            return (True, tid)
        if hasattr(tensor, "_fsdp_param_id"):
            return (False, tensor._fsdp_param_id)
        return tensor

    def _unpack_hook(self, packed):
        if isinstance(packed, tuple):
            is_param, param_id = packed
            param = self._param_id_to_param[param_id]
            return param if is_param else param.data
        return packed

    # =========================================================================
    # FSDP EXECUTION HOOKS
    # =========================================================================
    def attach_fsdp_hooks(self):
        self.module_infos: dict[torch.nn.Module, ModuleInfo] = {}
        self.fwd_modules = []
        self.bwd_modules = []
        self.fwd_prefetches = []

        for mod in self.module.modules():
            if is_shardable(mod):
                if mod not in self.module_infos:
                    self.module_infos[mod] = ModuleInfo()

                ####### PRE-FORWARD
                def make_fwd_pre(dt):
                    def hook(m, inp):
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            if ps.all_gather_handle is not None:
                                ps.all_gather_handle.wait()
                                ps.all_gather_handle = None
                            else:
                                ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                                dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=False)

                            ps.sharded_data = param.data

                            # Create a view, tag it for autograd, and assign it
                            gathered_view = ps.all_gather_data[: ps.numel].view(ps.shape)
                            gathered_view._fsdp_param_id = id(param)
                            param.data = gathered_view

                            # Drop the hard reference; the view keeps the storage alive
                            ps.all_gather_data = None

                    return hook

                mod.register_forward_pre_hook(make_fwd_pre(self.compute_dtype))

                ####### POST-FORWARD
                def make_fwd_post():
                    def hook(m, inp, out):
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            # Restoring sharded_data removes the last reference to the gathered
                            # view, allowing PyTorch's Caching Allocator to instantly recycle it!
                            param.data = ps.sharded_data
                            ps.sharded_data = None

                        mi = self.module_infos[m]
                        if mi.first_fwd:
                            mi.first_fwd = False
                            idx = len(self.fwd_modules)
                            if idx > 1:
                                self.module_infos[self.fwd_modules[idx - 2]].fwd_prefetch = m
                            else:
                                self.fwd_prefetches.append(m)
                            self.fwd_modules.append(m)
                        else:
                            pm = mi.fwd_prefetch
                            if pm is not None:
                                for p in pm.parameters(recurse=False):
                                    pps = self.param_states[p]
                                    pps.all_gather_data = torch.empty(self.world_size * pps.shard_size, dtype=p.dtype, device=p.device)
                                    pps.all_gather_handle = dist.all_gather_into_tensor(pps.all_gather_data, p.data, async_op=True)

                    return hook

                mod.register_forward_hook(make_fwd_post())

                ####### PRE-BACKWARD
                def make_bwd_pre(dt):
                    def hook(m, grad_output):
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            if ps.all_gather_handle is not None:
                                ps.all_gather_handle.wait()
                                ps.all_gather_handle = None
                            else:
                                ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                                dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=False)

                            ps.sharded_data = param.data
                            gathered_view = ps.all_gather_data[: ps.numel].view(ps.shape)
                            param.data = gathered_view
                            ps.all_gather_data = None

                    return hook

                mod.register_full_backward_pre_hook(make_bwd_pre(self.compute_dtype))

                ####### POST-BACKWARD
                def make_bwd_post(dt):
                    def hook(m, inp, out):
                        mi = self.module_infos[m]
                        if mi.first_bwd:
                            mi.first_bwd = False
                            idx = len(self.bwd_modules)
                            if idx > 1:
                                self.module_infos[self.bwd_modules[idx - 2]].bwd_prefetch = m
                            self.bwd_modules.append(m)
                        else:
                            pm = mi.bwd_prefetch
                            if pm is not None:
                                for p in pm.parameters(recurse=False):
                                    pps = self.param_states[p]
                                    pps.all_gather_data = torch.empty(self.world_size * pps.shard_size, dtype=p.dtype, device=p.device)
                                    pps.all_gather_handle = dist.all_gather_into_tensor(pps.all_gather_data, p.data, async_op=True)

                    return hook

                mod.register_full_backward_hook(make_bwd_post(self.compute_dtype))

                ####### GRADIENT HOOK
                def make_grad_hook():
                    def hook(p):
                        ps = self.param_states[p]

                        # 1. Swap back to sharded weights
                        p.data = ps.sharded_data
                        ps.sharded_data = None

                        # 2. Process gradients
                        orig_grad = p.grad
                        orig_grad.div_(self.world_size)

                        flat_grad = orig_grad.view(-1)
                        if ps.pad_size > 0:
                            flat_grad = F.pad(flat_grad, (0, ps.pad_size))

                        # Pre-allocate buffer for the incoming reduced shard
                        p.grad = torch.empty(ps.shard_size, dtype=orig_grad.dtype, device=orig_grad.device)

                        ps.grad_handle = dist.reduce_scatter_tensor(output=p.grad, input=flat_grad, op=dist.ReduceOp.SUM, async_op=True)

                        # Keep references alive until async NCCL completes
                        ps.flat_grad = flat_grad
                        ps.orig_grad = orig_grad

                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(make_grad_hook())

            else:
                # Non-sharded module
                def make_grad_hook():
                    def hook(p):
                        p.grad.div_(self.world_size)
                        self.param_states[p].grad_handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)

                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(make_grad_hook())

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    def forward(self, *inputs, **kwargs):
        for mod in self.fwd_prefetches:
            for param in mod.parameters(recurse=False):
                ps = self.param_states[param]
                ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                ps.all_gather_handle = dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=True)

        # Wrap in autograd hooks to prevent memory leaks during backward tracking
        with torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook):
            return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param_state in self.param_states.values():
            if param_state.grad_handle is not None:
                param_state.grad_handle.wait()
                param_state.grad_handle = None

                # Async op finished, drop references to free memory instantly
                param_state.flat_grad = None
                param_state.orig_grad = None

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        res = {}
        for name, param in self.module.named_parameters():
            ps = self.param_states[param]
            if ps.is_shardable:
                gathered_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                dist.all_gather_into_tensor(gathered_data, param.data, async_op=False)
                res[name] = gathered_data[: ps.numel].view(ps.shape)
            else:
                res[name] = param.data
        return res
