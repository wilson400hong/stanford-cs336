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
        t.untyped_storage().resize_(0)


def pad_tensor(t: torch.Tensor, pad_size: int) -> torch.Tensor:
    return (
        torch.cat([t, torch.zeros(pad_size, dtype=t.dtype, device=t.device)])
        if pad_size > 0
        else t
    )


@dataclass
class ParamState:
    is_shardable: bool
    shape: torch.Size
    dtype: torch.dtype          # 儲存精度 (storage_dtype，通常為 float32)
    compute_dtype: torch.dtype  # 計算精度 (例如 bfloat16 / float16)
    numel: int = 0
    pad_size: int = 0
    shard_size: int = 0

    # 共享底層 UntypedStorage 的固定張量，用於維持 Autograd SavedVariable 的記憶體連續
    unsharded_flat: torch.Tensor | None = None
    unsharded_view: torch.Tensor | None = None
    sharded_data: torch.Tensor | None = None

    # 非同步通訊 handle 與緩衝區
    all_gather_handle: dist.Work | None = None
    grad_handle: dist.Work | None = None
    grad_input_buf: torch.Tensor | None = None
    grad_output_shard: torch.Tensor | None = None


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

        # 廣播參數與 buffer 確保 rank 0 與其他 rank 初始一致
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0)

        self.param_states: dict[torch.nn.Parameter, ParamState] = {}
        self.shardable_modules: list[torch.nn.Module] = []
        self._shard_params()
        self._attach_hooks()

    def _shard_params(self):
        for mod in self.module.modules():
            if is_shardable(mod):
                self.shardable_modules.append(mod)
                for p in mod.parameters(recurse=False):
                    flat = p.data.detach().flatten()
                    pad_size = (self.world_size - (flat.numel() % self.world_size)) % self.world_size
                    flat = pad_tensor(flat, pad_size)
                    shard_size = flat.numel() // self.world_size

                    # 切割出當前 rank 負責的 shard
                    sharded = flat[self.rank * shard_size : (self.rank + 1) * shard_size].clone()

                    comp_dtype = (
                        self.compute_dtype
                        if (use_compute_dtype(mod) and self.compute_dtype is not None)
                        else p.dtype
                    )

                    # 建立 1D flat buffer 與對應原始形狀的視圖（共享同一個 Storage）
                    unsharded_flat = torch.empty(flat.numel(), dtype=comp_dtype, device=p.device)
                    unsharded_view = unsharded_flat[: p.numel()].view(p.shape)
                    free_tensor(unsharded_flat)  # 初始將 storage resize 為 0，不佔額外顯存

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
                        sharded_data=sharded,
                    )
                    p.data = sharded
                    free_tensor(flat)
            else:
                for p in mod.parameters(recurse=False):
                    self.param_states[p] = ParamState(
                        is_shardable=False,
                        shape=p.shape,
                        dtype=p.dtype,
                        compute_dtype=p.dtype,
                    )

    def _gather_module(self, mod: torch.nn.Module, async_op: bool = False):
        """將模組內參數的 buffer storage 擴展回完整大小，並發起 All-Gather（轉換為 compute_dtype）"""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            if ps.all_gather_handle is None and ps.unsharded_flat.untyped_storage().size() == 0:
                full_bytes = self.world_size * ps.shard_size * ps.unsharded_flat.element_size()
                ps.unsharded_flat.untyped_storage().resize_(full_bytes)
                # Downcast/Upcast 到 compute_dtype 進行通訊與後續計算
                ps.all_gather_handle = dist.all_gather_into_tensor(
                    ps.unsharded_flat, p.data.to(ps.compute_dtype), async_op=async_op
                )

    def _wait_and_assign_module(self, mod: torch.nn.Module):
        """等待通訊完成並將 param.data 指回 compute_dtype 的視圖"""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            if ps.all_gather_handle is not None:
                ps.all_gather_handle.wait()
                ps.all_gather_handle = None
            p.data = ps.unsharded_view

    def _release_module(self, mod: torch.nn.Module):
        """將 param.data 換回 shard，並釋放完整權重的顯存"""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            p.data = ps.sharded_data
            free_tensor(ps.unsharded_flat)

    def _attach_hooks(self):
        # 建立前向 prefetch 關係 (mod[i] -> mod[i + 1])
        next_fwd_mod = {
            self.shardable_modules[i]: self.shardable_modules[i + 1]
            for i in range(len(self.shardable_modules) - 1)
        }

        # 建立反向 prefetch 關係 (僅限需要 all-gather 的 Linear: linear[i] -> linear[i - 1])
        linear_modules = [m for m in self.shardable_modules if isinstance(m, Linear)]
        next_bwd_linear = {
            linear_modules[i]: linear_modules[i - 1]
            for i in range(len(linear_modules) - 1, 0, -1)
        }

        for mod in self.module.modules():
            if is_shardable(mod):
                # 1. Forward Pre Hook
                def fwd_pre(m, inp):
                    if m.parameters(recurse=False):
                        self._gather_module(m, async_op=False)
                        self._wait_and_assign_module(m)
                    # 發起下一個模組的非同步 All-Gather (Forward Prefetch)
                    nxt = next_fwd_mod.get(m)
                    if nxt is not None:
                        self._gather_module(nxt, async_op=True)

                mod.register_forward_pre_hook(fwd_pre)

                # 2. Forward Post Hook
                mod.register_forward_hook(lambda m, inp, out: self._release_module(m))

                # 3. Backward Pre Hook
                def bwd_pre(m, grad_out):
                    if isinstance(m, Linear):
                        # 等待自身參數到位
                        self._gather_module(m, async_op=False)
                        self._wait_and_assign_module(m)
                        # 發起上一層 Linear 的非同步 All-Gather (Backward Prefetch)
                        prev_linear = next_bwd_linear.get(m)
                        if prev_linear is not None:
                            self._gather_module(prev_linear, async_op=True)

                mod.register_full_backward_pre_hook(bwd_pre)

                # 4. Post Accumulate Grad Hook (處理權重恢復、反向梯度型別轉換與 Reduce-Scatter)
                def make_grad_hook():
                    def hook(p: torch.nn.Parameter):
                        ps = self.param_states[p]
                        raw_grad = p.grad
                        p.grad = None  # 置空，避免切換 p.data 時型別與形狀檢查報錯

                        # 恢復成 shard 狀態並清空展開的全量顯存
                        p.data = ps.sharded_data
                        free_tensor(ps.unsharded_flat)

                        # 將梯度轉回儲存精度 (ps.dtype)、補 padding 並均分 world_size
                        flat_grad = pad_tensor(raw_grad.to(ps.dtype).detach().flatten(), ps.pad_size)
                        flat_grad.div_(self.world_size)
                        del raw_grad

                        ps.grad_input_buf = flat_grad
                        ps.grad_output_shard = torch.empty(ps.shard_size, dtype=ps.dtype, device=p.device)
                        ps.grad_handle = dist.reduce_scatter_tensor(
                            output=ps.grad_output_shard,
                            input=ps.grad_input_buf,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )
                    return hook

                for p in mod.parameters(recurse=False):
                    if p.requires_grad:
                        param_hook = make_grad_hook()
                        p.register_post_accumulate_grad_hook(param_hook)
            else:
                # 非切分模組（Replicated）：直接對梯度進行 All-Reduce
                def make_replicated_hook():
                    def hook(p: torch.nn.Parameter):
                        p.grad.div_(self.world_size)
                        self.param_states[p].grad_handle = dist.all_reduce(
                            p.grad, op=dist.ReduceOp.SUM, async_op=True
                        )
                    return hook

                for p in mod.parameters(recurse=False):
                    if p.requires_grad:
                        p.register_post_accumulate_grad_hook(make_replicated_hook())

    def forward(self, *inputs, **kwargs):
        # 預先發起第 1 個切分模組的 All-Gather
        if self.shardable_modules:
            self._gather_module(self.shardable_modules[0], async_op=True)
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有梯度的集合通訊結束，並將結果累加或賦值回 param.grad"""
        for p, ps in self.param_states.items():
            if ps.grad_handle is not None:
                ps.grad_handle.wait()
                ps.grad_handle = None

                if ps.is_shardable:
                    if p.grad is None:
                        p.grad = ps.grad_output_shard
                    else:
                        p.grad.add_(ps.grad_output_shard)
                    ps.grad_output_shard = None
                    free_tensor(ps.grad_input_buf)
                    ps.grad_input_buf = None

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """收集各 Rank 參數並還原完整形狀（以 storage_dtype 輸出）"""
        res = {}
        for name, p in self.module.named_parameters():
            ps = self.param_states[p]
            if ps.is_shardable:
                full_data = torch.empty(self.world_size * ps.shard_size, dtype=p.dtype, device=p.device)
                dist.all_gather_into_tensor(full_data, p.data, async_op=False)
                res[name] = full_data[: ps.numel].reshape(ps.shape)
            else:
                res[name] = p.data.clone()
        return res
