# Fully-Sharded Data Parallel (FSDP) Implementation Notes

Comprehensive implementation guide, pitfalls, autograd engine internals, and production-grade solutions for the CS336 assignment.

---

## 1. Core Architecture & Mental Model

FSDP shards the data-parallel axis (ZeRO-3 style): each rank retains strictly $1 / \text{world\_size}$ of every shardable weight (`Linear`, `Embedding`), its matching gradient slice, and optimizer state. Full weights are gathered just-in-time via collective communications (`all_gather_into_tensor`) and scattered after backward execution (`reduce_scatter_tensor`). Replicated parameters (e.g., LayerNorm/RMSNorm scale vectors) remain unsharded to eliminate collective communication latency on tiny tensors.

```
                     FORWARD PASS                                         BACKWARD PASS
  ┌──────────────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
  │ [fwd_pre_hook]                               │     │ [bwd_pre_hook]                               │
  │ 1. Wait/Trigger All-Gather (comp_dtype)      │     │ 1. Linear only: All-Gather (comp_dtype)      │
  │ 2. Point param.data -> unsharded_view        │     │ 2. Point param.data -> unsharded_view        │
  │ 3. Async Prefetch Layer (i + 1)              │     │ 3. Async Prefetch Layer (i - 1)              │
  └──────────────────────┬───────────────────────┘     └──────────────────────┬───────────────────────┘
                         │                                                    │
                 [Layer Forward]                                      [Layer Backward]
                         │                                                    │
  ┌──────────────────────▼───────────────────────┐     ┌──────────────────────▼───────────────────────┐
  │ [fwd_post_hook]                              │     │ [post_accumulate_grad_hook]                  │
  │ 1. Restore param.data -> sharded_data        │     │ 1. Cache raw grad & set p.grad = None        │
  │ 2. In-place free: storage.resize_(0)         │     │ 2. Restore param.data -> sharded_data        │
  └──────────────────────────────────────────────┘     │ 3. In-place free: storage.resize_(0)         │
                                                       │ 4. Cast to storage_dtype & pad               │
                                                       │ 5. Async Reduce-Scatter gradient slice       │
                                                       └──────────────────────────────────────────────┘

```

---

## 2. In-Depth Pitfalls & Failure Modes

### Pitfall 1: Autograd `SavedVariable` Storage Severance

* **The Root Cause:** When PyTorch executes forward operations (e.g., `y = x @ W.T`), Autograd wraps the parameter in a `SavedVariable` referencing the exact underlying `UntypedStorage` instance.
* **The Flaw:** If the gathered buffer is allocated dynamically via `torch.empty(...)` on every forward pass and then freed with `storage.resize_(0)` in `fwd_post_hook`, re-gathering into a *new* storage in `bwd_pre_hook` fails. The backward graph still references the original storage (which is now zero-sized), triggering fatal memory access violations in `MmBackward0`.
* **The Fix:** Maintain a persistent `unsharded_flat_buffer` and `unsharded_param_view` per parameter across the module lifetime. During forward and backward stages, expand and collapse the **same underlying storage instance** using `unsharded_flat.untyped_storage().resize_(target_bytes)` and `unsharded_flat.untyped_storage().resize_(0)`.

### Pitfall 2: Hook Execution Inversion (`Linear` vs. `Embedding`)

Autograd does not operate on modules; it operates on directed acyclic computation graphs (DAGs). Module hooks (`register_full_backward_hook`) insert identity backward nodes around tensors.

#### Case A: `Linear` Layer

$$\nabla_y \longrightarrow [\text{bwd\_pre\_hook}] \longrightarrow [\text{MmBackward0}] \longrightarrow (\nabla_x, \nabla_W)$$

* $\nabla_x$ leads to the module input backward hook (`full_backward_hook`).
* $\nabla_W$ leads to `AccumulateGrad` $\rightarrow$ `post_accumulate_grad_hook`.
* **The Race Condition:** Both branches are siblings. The PyTorch ready-queue prioritizes `AccumulateGrad` to flush gradient sinks immediately. Consequently, `post_accumulate_grad_hook` frequently executes **before** `full_backward_hook`.

#### Case B: `Embedding` Layer

* Embedding input is integer tensor IDs (`dtype=torch.long`), which cannot require gradients.
* Because no input gradient exists, PyTorch triggers `full_backward_hook` **immediately** at the module output node prior to computing $\nabla_W$.
* **The Impact:** Relying on `full_backward_hook` to reshard parameters causes non-deterministic ordering and size mismatch exceptions.

### Pitfall 3: C++ Type & Dimension Assertion on `p.grad` Assignment

Assigning directly to `p.grad` in Python invokes the internal `THPVariable_set_grad` C++ method, which enforces:


$$\text{shape}(p.\text{grad}) \equiv \text{shape}(p.\text{data}) \quad \land \quad \text{dtype}(p.\text{grad}) \equiv \text{dtype}(p.\text{data})$$


If `p.data` is still pointing to the full compute-precision view (`(out_dim, in_dim)`, `bfloat16`) when the grad hook assigns a flattened shard (`(shard_size,)`, `float32`), PyTorch throws `RuntimeError: assigned grad has data of a different size / dtype`.

### Pitfall 4: Wasteful Backward Gather on `Embedding`

$$\nabla_{\text{Embedding}} = \text{scatter\_add}(\nabla_y, \text{indices})$$


Calculating embedding weight gradients requires solely token indices and output gradient values. It never reads existing weight values. Performing an All-Gather on the vocab weight table during backward pass squanders network interconnect bandwidth without functional purpose.

---

## 3. Structural Solutions: Dual-Buffer Strategy

To satisfy both collective communications and tensor operations without breaking Autograd references, maintain two paired tensor objects sharing identical storage:

| Buffer | Geometry & Properties | Primary Objective |
| --- | --- | --- |
| **`unsharded_flat`** | 1D vector, length padded to multiple of `world_size`, `compute_dtype`. | Direct target for `dist.all_gather_into_tensor`. |
| **`unsharded_view`** | Multi-dimensional matrix (`shape == ps.shape`), zero padding, `compute_dtype`. | Assigned to `param.data` for forward and backward compute. |

```python
# Shared memory instantiation
unsharded_flat = torch.empty(padded_numel, dtype=comp_dtype, device=p.device)
unsharded_view = unsharded_flat[: p.numel()].view(p.shape)
# Collapse storage to zero bytes to eliminate memory consumption at rest
unsharded_flat.untyped_storage().resize_(0)

```

---

## 4. End-to-End Implementation

```python
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
    """Collapse tensor storage to zero bytes without modifying metadata."""
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
    dtype: torch.dtype          # Parameter master/storage precision (float32)
    compute_dtype: torch.dtype  # Communication & execution precision (bfloat16/float16)
    numel: int = 0
    pad_size: int = 0
    shard_size: int = 0

    # Persistent dual-buffers sharing single UntypedStorage
    unsharded_flat: torch.Tensor | None = None
    unsharded_view: torch.Tensor | None = None
    sharded_data: torch.Tensor | None = None

    # Async communication handles & buffers
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
            raise RuntimeError("Torch distributed environment must be initialized.")

        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Synchronize module baseline across all ranks
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

                    # Slice local rank partition
                    sharded = flat[self.rank * shard_size : (self.rank + 1) * shard_size].clone()

                    comp_dtype = (
                        self.compute_dtype
                        if (use_compute_dtype(mod) and self.compute_dtype is not None)
                        else p.dtype
                    )

                    # Initialize persistent dual-buffers
                    unsharded_flat = torch.empty(flat.numel(), dtype=comp_dtype, device=p.device)
                    unsharded_view = unsharded_flat[: p.numel()].view(p.shape)
                    free_tensor(unsharded_flat)

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
        """Expands storage in-place and triggers all_gather into unsharded_flat."""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            if ps.all_gather_handle is None and ps.unsharded_flat.untyped_storage().size() == 0:
                full_bytes = self.world_size * ps.shard_size * ps.unsharded_flat.element_size()
                ps.unsharded_flat.untyped_storage().resize_(full_bytes)
                ps.all_gather_handle = dist.all_gather_into_tensor(
                    ps.unsharded_flat, p.data.to(ps.compute_dtype), async_op=async_op
                )

    def _wait_and_assign_module(self, mod: torch.nn.Module):
        """Awaits collective communication and assigns view to param.data."""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            if ps.all_gather_handle is not None:
                ps.all_gather_handle.wait()
                ps.all_gather_handle = None
            p.data = ps.unsharded_view

    def _release_module(self, mod: torch.nn.Module):
        """Restores sharded parameter representation and frees unsharded memory."""
        for p in mod.parameters(recurse=False):
            ps = self.param_states[p]
            p.data = ps.sharded_data
            free_tensor(ps.unsharded_flat)

    def _attach_hooks(self):
        # Forward prefetch topology: mod[i] -> mod[i + 1]
        next_fwd_mod = {
            self.shardable_modules[i]: self.shardable_modules[i + 1]
            for i in range(len(self.shardable_modules) - 1)
        }

        # Backward prefetch topology: Linear modules only: linear[i] -> linear[i - 1]
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
                    nxt = next_fwd_mod.get(m)
                    if nxt is not None:
                        self._gather_module(nxt, async_op=True)

                mod.register_forward_pre_hook(fwd_pre)

                # 2. Forward Post Hook
                mod.register_forward_hook(lambda m, inp, out: self._release_module(m))

                # 3. Backward Pre Hook
                def bwd_pre(m, grad_out):
                    if isinstance(m, Linear):
                        self._gather_module(m, async_op=False)
                        self._wait_and_assign_module(m)
                        prev_linear = next_bwd_linear.get(m)
                        if prev_linear is not None:
                            self._gather_module(prev_linear, async_op=True)

                mod.register_full_backward_pre_hook(bwd_pre)

                # 4. Post Accumulate Grad Hook
                def make_grad_hook():
                    def hook(p: torch.nn.Parameter):
                        ps = self.param_states[p]
                        raw_grad = p.grad
                        p.grad = None  # Evade shape/dtype checks during p.data swap

                        # Restore shard and immediately collapse unsharded storage
                        p.data = ps.sharded_data
                        free_tensor(ps.unsharded_flat)

                        # Cast to storage_dtype (float32), pad, and normalize by world_size
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
                        p.register_post_accumulate_grad_hook(make_grad_hook())
            else:
                # Replicated non-sharded parameters (e.g. Norm layers)
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
        # Prefetch first module before initiating compute graph
        if self.shardable_modules:
            self._gather_module(self.shardable_modules[0], async_op=True)
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Awaits all outstanding collective communications and accumulates gradients."""
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
        """Gathers parameters across ranks for evaluation/checkpointing (storage_dtype)."""
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

```

---

## 5. Corner Cases & Verification Checklist

### Corner Case 1: Gradient Accumulation & In-Flight Tensors

Never configure the output of an asynchronous `reduce_scatter_tensor` directly to `p.grad`. If multiple backward passes execute before an optimizer step:

1. Direct assignment overwrites previous accumulations.
2. An asynchronous collective writes directly to in-use tensors, producing race conditions.
Holding the output in `ps.grad_output_shard` and applying `p.grad.add_()` upon `.wait()` inside `finish_gradient_synchronization()` ensures correctness.

### Corner Case 2: Tied Weights (`lm_head.weight == tok_embeddings.weight`)

If token embeddings and projection heads share identical parameters:

* The post-accumulate grad hook fires twice within a single backward iteration.
* Staging communications independently prevents in-flight communication buffer collisions.

### Corner Case 3: Mixed Precision (`compute_dtype` Up/Down Casting)

* **Weights:** Stored as `torch.float32`. When gathering, slice `p.data` is downcast via `.to(ps.compute_dtype)` before `dist.all_gather_into_tensor`. Compute runs in low precision.
* **Gradients:** `raw_grad` arrives in `compute_dtype`. It must be converted back to `storage_dtype` (`.to(ps.dtype)`) before Reduce-Scatter to match optimizer master weight expectations.

### Verification Run Commands

```bash
# 1. Base FP32 correctness verification
pytest tests/test_fsdp.py -k "correctness and fp32" -q -s

# 2. Mixed precision (compute_dtype=bfloat16/float16) verification
pytest tests/test_fsdp.py -k "correctness and compute_dtype" -q -s

# 3. Gradient sync, dtypes, and shapes validation
pytest tests/test_fsdp.py -k "gradient_sync" -q

# 4. Multi-iteration race condition stress test
for i in {1..5}; do
  echo "=== Test Iteration $i ==="
  pytest tests/test_fsdp.py -x -q || break
done

```
