import torch
import torch.distributed as dist


class NaiveDDPModule(torch.nn.Module):
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
    def reduce_all_gradients(self):
        # BAD: each param callls one all_reduce
        # for param in self.module.parameters():
        #     if param.requires_grad:
        #         param.grad.div_(self.world_size)
        #         dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)

        # GOOD: all params share one all_reduce
        grads = [p.grad for p in self.module.parameters() if p.requires_grad]
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        flat_grads.div_(self.world_size)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)

        unflatten_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for orig_grad, new_grad in zip(grads, unflatten_grads):
            orig_grad.copy_(new_grad)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


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

        self.handles = []

        def grad_hook(p):
            p.grad.div_(self.world_size)
            handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(grad_hook)

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    @torch.no_grad
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
