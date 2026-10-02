from dataclasses import dataclass
import torch
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear

SHARDED_MODULE_TYPES = (Embedding, Linear)


def is_shardable(m: torch.nn.Module) -> bool:
    return isinstance(m, SHARDED_MODULE_TYPES)


def use_compute_dtype(m: torch.nn.Module) -> bool:
    return isinstance(m, Linear)


def free_tensor(t: torch.Tensor | None):
    if t is not None and t.untyped_storage().size() > 0:
        data.untyped_storage().resize_(0)


######### TODO ########


def get_pad_size(data_size: int, world_size: int) -> int:
    return (world_size - (data_size % world_size)) % world_size


def pad_tensor(x: torch.Tensor, pad_size: int) -> torch.Tensor:
    return torch.cat([x, torch.zeros(pad_size, dtype=x.dtype, device=x.device)])


@dataclass
class ParamState:
    # 靜態形狀與型別配置
    is_shardable: bool
    shape: torch.Size
    dtype: torch.dtype
    compute_dtype: torch.dtype
    numel: int = 0
    pad_size: int = 0
    shard_size: int = 0

    # 共享底層 UntypedStorage 的固定張量（用 resize_ 控制顯存）
    unsharded_flat_buffer: torch.Tensor | None = None
    unsharded_param_view: torch.Tensor | None = None

    # 本 Rank 負責的 Shard 副本
    sharded_data: torch.Tensor | None = None

    # 通訊 handle 與暫存區
    all_gather_handle: dist.Work | None = None
    grad_handle: dist.Work | None = None
    full_grad_buffer: torch.Tensor | None = None
    reduced_grad_shard: torch.Tensor | None = None


@dataclass
class ModuleInfo:
    first_fwd: bool = True
    fwd_prefetch: torch.nn.Module | None = None


class FSDPModule(torch.nn.Module):
    def __init__(
        self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None
    ):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("torch distributed not initialized")

        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.sync_module()
        self.shard_params()
        self.attach_fsdp_hooks()

    def rank0_print(self, msg: str):
        if self.rank == 0:
            print(msg)

    def sync_module(self):
        self.rank0_print("Syncing module...")
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def shard_params(self):
        """為可切分的參數配置 Shard 與固定的 Unsharded Buffers"""
        self.rank0_print("Sharding params...")
        self.param_states: dict[torch.nn.Parameter, ParamState] = {}

        for module in self.module.modules():
            if is_shardable(module):
                for pname, param in module.named_parameters(recurse=False):
                    orig_size = param.untyped_storage().size()
                    flat_data = param.data.detach().flatten()
                    pad_size = get_pad_size(
                        data_size=flat_data.numel(), world_size=self.world_size
                    )
                    if pad_size > 0:
                        flat_data = pad_tensor(flat_data, pad_size)

                    shard_size = flat_data.numel() // self.world_size
                    start_idx = self.rank * shard_size
                    end_idx = start_idx + shard_size
                    sharded_data = flat_data[start_idx:end_idx].clone()

                    comp_dtype = (
                        self.compute_dtype
                        if (use_compute_dtype(module) and self.compute_dtype)
                        else param.dtype
                    )

                    # 建立 unsharded flat buffer 與對應的模型形狀 view
                    # 兩者共享底層同一個 UntypedStorage
                    unsharded_flat = torch.empty(
                        flat_data.numel(), dtype=comp_dtype, device=param.device
                    )
                    unsharded_view = unsharded_flat[: param.numel()].view(param.shape)

                    # 初始化完畢後立刻將 storage 縮為 0（不佔額外顯存）
                    free_tensor(unsharded_flat)

                    self.param_states[param] = ParamState(
                        is_shardable=True,
                        shape=param.shape,
                        dtype=param.dtype,
                        compute_dtype=comp_dtype,
                        numel=param.numel(),
                        pad_size=pad_size,
                        shard_size=shard_size,
                        unsharded_flat_buffer=unsharded_flat,
                        unsharded_param_view=unsharded_view,
                        sharded_data=sharded_data,
                    )

                    param.data = sharded_data
                    free_tensor(flat_data)

                    after_size = param.untyped_storage().size()
                    self.rank0_print(
                        f"Size comparison {pname=}  {orig_size=}  {after_size=}"
                    )
            else:
                for param in module.parameters(recurse=False):
                    self.param_states[param] = ParamState(
                        is_shardable=False,
                        shape=param.shape,
                        dtype=param.dtype,
                        compute_dtype=param.dtype,
                    )

    def allocate_and_gather_param(
        self, param: torch.nn.Parameter, async_op: bool = False
    ):
        """將固定的 unsharded_flat_buffer 擴展回完整大小並執行 all-gather"""
        ps = self.param_states[param]
        buf = ps.unsharded_flat_buffer
        assert buf is not None

        # 若已在通訊中或顯存已展開則跳過
        if ps.all_gather_handle is not None or buf.untyped_storage().size() > 0:
            return

        full_nbytes = self.world_size * ps.shard_size * buf.element_size()
        buf.untyped_storage().resize_(full_nbytes)

        ps.all_gather_handle = dist.all_gather_into_tensor(
            buf, param.data.to(ps.compute_dtype), async_op=async_op
        )

    def attach_fsdp_hooks(self):
        self.module_infos: dict[torch.nn.Module, ModuleInfo] = {}
        self.fwd_modules: list[torch.nn.Module] = []
        self.fwd_prefetches: list[torch.nn.Module] = []

        for mod in self.module.modules():
            if mod not in self.module_infos:
                self.module_infos[mod] = ModuleInfo()

            if is_shardable(mod):
                # ----------------- 1. Forward Pre Hook -----------------
                def make_fwd_pre():
                    def hook(module, inp):
                        for param in module.parameters(recurse=False):
                            ps = self.param_states[param]
                            if ps.all_gather_handle is not None:
                                ps.all_gather_handle.wait()
                                ps.all_gather_handle = None
                            else:
                                self.allocate_and_gather_param(param, async_op=False)
                            # 賦值為同一個 View，維持 Autograd 追蹤的 Storage 一致
                            param.data = ps.unsharded_param_view
                    return hook

                mod.register_forward_pre_hook(make_fwd_pre())

                # ----------------- 2. Forward Post Hook -----------------
                def make_fwd_post():
                    def hook(module, inp, out):
                        for param in module.parameters(recurse=False):
                            ps = self.param_states[param]
                            # 換回 shard，並釋放完整權重的顯存空間
                            param.data = ps.sharded_data
                            free_tensor(ps.unsharded_flat_buffer)

                        mi = self.module_infos[module]
                        if mi.first_fwd:
                            mi.first_fwd = False
                            idx = len(self.fwd_modules)
                            if idx > 1:
                                self.module_infos[
                                    self.fwd_modules[idx - 2]
                                ].fwd_prefetch = module
                            else:
                                self.fwd_prefetches.append(module)
                            self.fwd_modules.append(module)
                        else:
                            pmod = mi.fwd_prefetch
                            if pmod is not None:
                                for param in pmod.parameters(recurse=False):
                                    self.allocate_and_gather_param(param, async_op=True)
                    return hook

                mod.register_forward_hook(make_fwd_post())

                # ----------------- 3. Backward Pre Hook -----------------
                def make_bwd_pre():
                    def hook(module, grad_output):
                        # Embedding 的反向計算只需要 token IDs，不需要權重數值，可直接跳過通訊
                        if isinstance(module, Linear):
                            for param in module.parameters(recurse=False):
                                ps = self.param_states[param]
                                if ps.all_gather_handle is not None:
                                    ps.all_gather_handle.wait()
                                    ps.all_gather_handle = None
                                else:
                                    self.allocate_and_gather_param(param, async_op=False)
                                param.data = ps.unsharded_param_view
                    return hook

                mod.register_full_backward_pre_hook(make_bwd_pre())

                # ----------------- 4. Post Accumulate Grad Hook -----------------
                def make_grad_hook():
                    def hook(p: torch.nn.Parameter):
                        ps = self.param_states[p]
                        assert p.grad is not None

                        # 先取出完整梯度，並置空 p.grad 避免觸發形狀與型別檢查錯誤
                        raw_grad = p.grad
                        p.grad = None

                        # 恢復成 shard 狀態並釋放暫存顯存
                        p.data = ps.sharded_data
                        free_tensor(ps.unsharded_flat_buffer)

                        # 轉換型別、展平與補齊 Padding
                        flat_grad = raw_grad.to(ps.dtype).detach().flatten()
                        del raw_grad
                        if ps.pad_size > 0:
                            flat_grad = pad_tensor(flat_grad, ps.pad_size)
                        flat_grad.div_(self.world_size)

                        ps.full_grad_buffer = flat_grad
                        ps.reduced_grad_shard = torch.empty(
                            ps.shard_size, dtype=ps.dtype, device=p.device
                        )
                        ps.grad_handle = dist.reduce_scatter_tensor(
                            output=ps.reduced_grad_shard,
                            input=ps.full_grad_buffer,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )
                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(make_grad_hook())

            else:
                # 非切分模組（維持 replicate 狀態），只需進行梯度 all-reduce
                def make_replicated_grad_hook():
                    def hook(p: torch.nn.Parameter):
                        assert p.grad is not None
                        p.grad.div_(self.world_size)
                        self.param_states[p].grad_handle = dist.all_reduce(
                            p.grad, op=dist.ReduceOp.SUM, async_op=True
                        )
                    return hook

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(
                            make_replicated_grad_hook()
                        )

    def forward(self, *inputs, **kwargs):
        # 預先發起前兩個模組的 prefetch
        for mod in self.fwd_prefetches:
            for param in mod.parameters(recurse=False):
                self.allocate_and_gather_param(param, async_op=True)

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有梯度的集合通訊結束，並將結果寫入 param.grad"""
        for param, ps in self.param_states.items():
            if ps.grad_handle is not None:
                ps.grad_handle.wait()
                ps.grad_handle = None

                if ps.is_shardable:
                    if param.grad is None:
                        param.grad = ps.reduced_grad_shard
                    else:
                        param.grad.add_(ps.reduced_grad_shard)  # add_ to deal with gradient accumulation!
                    ps.reduced_grad_shard = None
                    free_tensor(ps.full_grad_buffer)
                    ps.full_grad_buffer = None

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """收集所有 Rank 的參數，還原出完整的模型參數字典（主要用於評估或存檔）"""
        res = {}
        for name, param in self.module.named_parameters():
            ps = self.param_states[param]
            if ps.is_shardable:
                gathered_data = torch.empty(
                    self.world_size * ps.shard_size,
                    dtype=param.dtype,
                    device=param.device,
                )
                dist.all_gather_into_tensor(gathered_data, param.data, async_op=False)
                res[name] = gathered_data[: ps.numel].reshape(ps.shape)
            else:
                res[name] = param.data.clone()
        return res
