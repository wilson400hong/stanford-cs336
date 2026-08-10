from typing import Any, Type

import torch
import torch.distributed as dist
from collections.abc import Callable, Iterable
import torch.nn.functional as F


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch distributed not initialized!")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls
        self.optim = None  # Lazy creation
        self.rank_to_params = [[] for _ in range(self.world_size)]
        self.rank_sizes = [0 for _ in range(self.world_size)]
        self.max_size = 0  # max(size of rank_sizes)

        self.kwargs = kwargs
        super().__init__(params, kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        valid_params = [p for p in param_group["params"] if p.requires_grad]
        if not valid_params:
            return
        super().add_param_group(param_group)
        total_elements = sum(p.numel() for p in valid_params)
        chunk_size = total_elements / self.world_size

        cur_size = 0
        cur_rank = 0
        local_params = []
        for p in valid_params:
            cur_size += p.numel()
            self.rank_to_params[cur_rank].append(p)
            if cur_rank == self.rank:
                local_params.append(p)
            self.rank_sizes[cur_rank] += p.numel()
            if cur_size >= chunk_size and cur_rank + 1 < self.world_size:
                cur_size = 0
                cur_rank += 1
        self.max_size = max(self.rank_sizes)

        if min(self.rank_sizes) == 0:
            raise RuntimeError("Empty shard")
        if self.optim is None:
            self.optim = self.optimizer_cls(local_params, **self.kwargs)
            # print(f"[Rank {self.rank}] Add params {local_params} with args: {self.kwargs}")
        else:
            self.optim.add_param_group({"params": local_params})

    @torch.no_grad  # CRITICAL: since we modify Tensors that requires_grad=True
    def step(self, closure: Callable | None = None):
        # sync first param group's fields to all sub optimizer
        for idx, pg in enumerate(self.param_groups):
            for k, v in pg.items():
                if k != "params":
                    self.optim.param_groups[idx][k] = v

        loss = self.optim.step(closure)

        local_params = self.rank_to_params[self.rank]
        local_flat = torch._utils._flatten_dense_tensors(local_params)

        pad_size = self.max_size - self.rank_sizes[self.rank]
        if pad_size > 0:
            local_flat = F.pad(local_flat, (0, pad_size))

        gathered = [torch.zeros_like(local_flat) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_flat)

        for rank, pad_flat in enumerate(gathered):
            params = self.rank_to_params[rank]
            unpad_flat = pad_flat[: self.rank_sizes[rank]]
            unflat_params = torch._utils._unflatten_dense_tensors(unpad_flat, params)

            for p, new_p in zip(params, unflat_params):
                p.copy_(new_p)

        return loss

    # TODO: state_dict() and load_state_dict() for chedckpoint S/L
