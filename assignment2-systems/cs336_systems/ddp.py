import torch
import torch.distributed as dist


class DDPModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        if not dist.is_initialized():
            raise RuntimeError("torch distributed not initialized!")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.rank0_print("Broadcasting module parameters and buffers...")

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        self.rank0_print("Broadcast done")

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    @torch.no_grad
    def sync_all_gradients(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.div_(self.world_size)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
