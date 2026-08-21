from dataclasses import dataclass
import torch
import torch.distributed as dist
from collections import defaultdict

from cs336_basics.model import Embedding, Linear, RMSNorm


SHARDED_MODULE_TYPES = (Embedding, Linear)


def is_shardable(module: torch.nn.Module):
    return isinstance(module, SHARDED_MODULE_TYPES)


def free(data: torch.Tensor | None):
    if data is None:
        return
    if data.untyped_storage().size() > 0:
        data.untyped_storage().resize_(0)
    del data


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
    sharded_data: torch.Tensor | None = None  # for swap purpose
    all_gather_handle = None
    all_gather_data: torch.Tensor | None = None
    grad_handle = None

    flat_grad: torch.Tensor | None = None
    orig_grad: torch.Tensor | None = None


class ModuleInfo:
    """For FSDP Modules"""

    first_fwd: bool = True
    first_bwd: bool = True

    fwd_prefetch = None
    bwd_prefetch = None

    def __init__(self):
        pass


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
        self.attach_fsdp_hooks()

    def sync_module(self):
        self.rank0_print("sync module, broadcast params...")
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def shard_params(self):
        """
        Build ParamStates for each shardable param
        """
        self.param_states: dict[torch.nn.Parameter, ParamState] = {}
        for module in self.module.modules():
            if is_shardable(module):
                for param in module.parameters(recurse=False):
                    orig_size = param.untyped_storage().size()
                    # flat & padding
                    orig_data = param.data
                    flat_data = orig_data.detach().flatten()
                    pad_size = (self.world_size - (flat_data.numel() % self.world_size)) % self.world_size
                    if pad_size > 0:
                        flat_data = torch.cat([flat_data, torch.zeros(pad_size, dtype=flat_data.dtype, device=flat_data.device)])

                    # shard
                    shard_size = flat_data.numel() // self.world_size
                    start_idx = self.rank * shard_size
                    end_idx = start_idx + shard_size
                    sharded_data = flat_data[start_idx:end_idx].clone()

                    # param infos
                    self.param_states[param] = ParamState(is_shardable=True, shape=param.shape, numel=param.numel(), pad_size=pad_size, shard_size=shard_size)

                    # swap data and release
                    param.data = sharded_data
                    free(flat_data)
                    free(orig_data)

                    after_size = param.untyped_storage().size()
                    print(f"Size comparison {orig_size=}   {after_size=}")
            else:
                for param in module.parameters(recurse=False):
                    self.param_states[param] = ParamState(is_shardable=False)

    # TODO
    # [ ] debug
    # [ ] compute_dtype support
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
                    def hook(m, inp):
                        for param in m.parameters(recurse=False):
                            ps = self.param_states[param]
                            # wait all_gather handle
                            if ps.all_gather_handle is not None:
                                ps.all_gather_handle.wait()
                                ps.all_gather_handle = None
                            else:
                                # If missing, do all_gather on params
                                ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                                dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=False)  # sync

                            # ps.all_gather_data is the flat & pad
                            ps.sharded_data = param.data
                            param.data = ps.all_gather_data
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
                            param.data = ps.sharded_data
                            ps.sharded_data = None
                            free(orig_data)

                        mi = self.module_infos[m]
                        if mi.first_fwd:
                            # record ordering in fwd_modules
                            mi.first_fwd = False
                            idx = len(self.fwd_modules)
                            if idx > 1:
                                self.module_infos[self.fwd_modules[idx - 2]].fwd_prefetch = m
                            else:
                                self.fwd_prefetches.append(m)
                            self.fwd_modules.append(m)
                        else:
                            # prefetch
                            pm = mi.fwd_prefetch  # prefetch module
                            if pm is not None:
                                for p in pm.parameters(recurse=False):
                                    pps = self.param_states[p]
                                    pps.all_gather_data = torch.empty(self.world_size * pps.shard_size, dtype=p.dtype, device=p.device)
                                    pps.all_gather_handle = dist.all_gather_into_tensor(pps.all_gather_data, p.data, async_op=True)  # async

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
                            else:
                                # If missing, do all_gather on params
                                ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                                dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=False)  # sync

                            # ps.all_gather_data is the flat & pad
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
                        if mi.first_bwd:
                            # record ordering in bwd_modules
                            mi.first_bwd = False
                            idx = len(self.bwd_modules)
                            if idx > 1:
                                self.module_infos[self.bwd_modules[idx - 2]].bwd_prefetch = m
                            # else:
                            # self.fwd_prefetches.append(m)
                            self.bwd_modules.append(m)
                        else:
                            # prefetch
                            pm = mi.bwd_prefetch  # prefetch module
                            if pm is not None:
                                for p in pm.parameters(recurse=False):
                                    pps = self.param_states[p]
                                    pps.all_gather_data = torch.empty(self.world_size * pps.shard_size, dtype=p.dtype, device=p.device)
                                    pps.all_gather_handle = dist.all_gather_into_tensor(pps.all_gather_data, p.data, async_op=True)  # async

                    return hook

                mod.register_full_backward_hook(make_bwd_post(self.compute_dtype))

                def make_grad_hook():
                    def hook(p):
                        # swap param.data, and release orig data
                        ps = self.param_states[p]
                        orig_data = p.data
                        p.data = ps.sharded_data
                        ps.sharded_data = None
                        free(orig_data)

                        # reduce scatter gradient
                        ps = self.param_states[p]
                        orig_grad = p.grad
                        p.grad.div_(self.world_size)

                        flat_grad = p.grad.detach().flatten()
                        if ps.pad_size > 0:
                            flat_grad = torch.cat([flat_grad, torch.zeros(ps.pad_size, dtype=p.grad.dtype, device=p.grad.device)])
                        p.grad = torch.empty(ps.shard_size, dtype=p.grad.dtype, device=p.grad.device)
                        # TODO: sync
                        # dist.reduce_scatter_tensor(output=p.grad, input=flat_grad, op=dist.ReduceOp.SUM, async_op=True)

                        ps.grad_handle = dist.reduce_scatter_tensor(output=p.grad, input=flat_grad, op=dist.ReduceOp.SUM, async_op=True)
                        ps.flat_grad = flat_grad
                        ps.orig_grad = orig_grad

                        # free(flat_grad)
                        # free(orig_grad)

                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(make_grad_hook())

            else:
                # Non-sharded module -- only need grad hook for all_reduce
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
        # prefetch L0, L1 if has
        for mod in self.fwd_prefetches:
            for param in mod.parameters(recurse=False):
                ps = self.param_states[param]
                ps.all_gather_data = torch.empty(self.world_size * ps.shard_size, dtype=param.dtype, device=param.device)
                ps.all_gather_handle = dist.all_gather_into_tensor(ps.all_gather_data, param.data, async_op=True)

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param_state in self.param_states.values():
            if param_state.grad_handle is not None:
                param_state.grad_handle.wait()
                param_state.grad_handle = None
                free(param_state.flat_grad)
                free(param_state.orig_grad)
