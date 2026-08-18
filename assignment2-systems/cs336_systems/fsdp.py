from dataclasses import dataclass
import torch
import torch.distributed as dist
from collections import defaultdict

from cs336_basics.model import Embedding, Linear, RMSNorm


SHARDED_MODULE_TYPES = (Embedding, Linear)


def should_shard(module: torch.nn.Module):
    return isinstance(module, SHARDED_MODULE_TYPES)


def free(data: torch.Tensor):
    if data.untyped_storage().size() > 0:
        data.untyped_storage().resize_(0)
    del data


@dataclass
class ParamInfo:
    is_sharded: bool = False
    shape: torch.Size | None = None
    numel: int | None = None
    pad_size: int | None = None
    shard_size: int | None = None
    sharded_data: torch.Tensor | None = None

    grad_handle = None

    # def __init__(self, is_sharded: bool = False, shape: torch.Size | None = None, numel: int | None = None, pad_size: int | None = None, shard_size: int | None = None):
    #     self.is_sharded = is_sharded
    #     self.shape = shape
    #     self.numel = numel
    #     self.pad_size = pad_size
    #     self.shard_size = shard_size


class ModuleInfo:
    first_fwd: bool = True
    first_bwd: bool = True

    fwd_prefetch: torch.nn.Module | None = None
    bwd_prefetch: torch.nn.Module | None = None

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
        self.broadcast_module()

        self.shard_params()

        # self.apply_hooks()

    def broadcast_module(self):
        self.rank0_print("broadcast params...")
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def shard_params(self):
        """
        Build ParamInfos for each param, and do sharding if necessary
        """
        self.param_infos: dict[torch.nn.Parameter, ParamInfo] = {}
        for module in self.module.modules():
            if should_shard(module):
                for param in module.parameters(recurse=False):
                    before_size = param.untyped_storage().size()
                    # flat
                    orig_data = param.data
                    flat_data = orig_data.detach().flatten()
                    # padding
                    pad_size = (self.world_size - (flat_data.numel() % self.world_size)) % self.world_size
                    if pad_size > 0:
                        flat_data = torch.cat([flat_data, torch.zeros(pad_size, dtype=flat_data.dtype, device=flat_data.device)])

                    # shard
                    shard_size = flat_data.numel() // self.world_size
                    start_idx = self.rank * shard_size
                    end_idx = start_idx + shard_size
                    sharded_data = flat_data[start_idx:end_idx].clone()

                    # param infos
                    self.param_infos[param] = ParamInfo(is_sharded=True, shape=param.shape, numel=param.numel(), pad_size=pad_size, shard_size=shard_size)

                    # swap data and release
                    param.data = sharded_data
                    free(flat_data)
                    free(orig_data)

                    after_size = param.untyped_storage().size()
                    print(f"Size comparison {before_size=}   {after_size=}")
            else:
                for param in module.parameters(recurse=False):
                    self.param_infos[param] = ParamInfo()   # TODO: probably not needed?

    # TODO #2
    # 1. [X] record fwd and prefetch
    # 2. [ ] reocrd bwd and prefetch
    # 3. [ ] pre-fwd all gather weights
    # 4. [ ] post-fwd release weights
    # 5. [ ] pre-bwd all gather weights
    # 6. [ ] post-bwd reduce-scatter gradients
    # 7. [ ] post-bwd release weights
    # 8. [ ] compute_dtype support
    def apply_hooks(self):
        self.module_infos: dict[torch.nn.Module, ModuleInfo] = {}
        self.fwd_modules = []
        self.bwd_modules = []

        self.fwd_prefetch = {}  # Module -> Module
        self.bwd_prefetch = {}

        # sharded modules:
        for mod in self.module.modules():
            if not should_shard(mod):
                continue
            
            if mod not in self.module_infos:
                self.module_infos[mod] = ModuleInfo()

            # TODO
            def make_fwd_pre(dt):
                def hook(m, inp):
                    pass

                return hook

            # TODO
            def make_fwd_post():
                def hook(m, inp, out):
                    mi = self.module_infos[mod]
                    if mi.first_fwd:
                        # record in fwd_modules, and update L-2 prefetch
                        idx = len(self.fwd_modules)
                        # TODO: optimize idx == 1
                        if idx > 1:
                            mi.fwd_prefetch = self.fwd_modules[idx - 2]
                        if idx 
                        fwd_modules.append(m)
                        setattr(m, "first_fwd_post", True)
                    else:
                        # TODO
                return hook

            mod.register_forward_pre_hook(make_fwd_pre(self.compute_dtype))
            mod.register_forward_hook(make_fwd_post())

            # TODO
            def make_bwd_pre(dt):
                def hook(m, grad_output):
                    if not hasattr(m, "first_bwd_pre"):
                        idx = len(bwd_modules)
                        # TODO: optimize idx == 1
                        if idx > 1:
                            bwd_prefetch[bwd_modules[idx - 2]] = m
                        bwd_modules.append(m)
                        setattr(m, "first_bwd_pre", True)

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(self.compute_dtype))


            # TODO
            def make_bwd_post(dt):
                def hook(m, grad_output):
                    if not hasattr(m, "first_bwd_pre"):
                        idx = len(bwd_modules)
                        # TODO: optimize idx == 1
                        if idx > 1:
                            bwd_prefetch[bwd_modules[idx - 2]] = m
                        bwd_modules.append(m)
                        setattr(m, "first_bwd_pre", True)

                return hook

            mod.register_backward_hook(make_bwd_post())

            # TODO
            def make_grad_hook(m):
                def hook(param):
                    # TODO: reduce scatter

                return hook

            mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod))

        # replicate modules only need to all_reduce on gradient
        for mod in self.module.modules():
            if should_shard(mod):
                continue

            def make_grad_hook(m):
                def hook(p):
                    p.grad.div_(self.world_size)
                    self.param_infos[p].grad_handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)

                return hook

            for param in mod.parameters(recurse=False):
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(make_grad_hook(mod))

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    def forward(self, *inputs, **kwargs):
        # TODO: prefetch L0, L1

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param_info in self.param_infos.values():
            if param_info.grad_handle is not None:
                param_info.grad_handle.wait()
                param_info.grad_handle = None
