import torch
import torch.distributed as dist


from cs336_basics.model import Embedding, Linear, RMSNorm

# TODO
class FSDPModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype):
        super().__init__()

        self.module = module
        self.fwd_orders = []
        self.bwd_orders = []
        self.record_orders()

    @torch.no_grad
    def record_orders(self):
        # TODO: dummy forward and backward to record layers ordering
        for mod in self.module.modules():
        


    def apply_hooks(self, compute_dtype: torch.dtype):
        # TODO
        raise RuntimeError("")

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

    def forward(self, *args, **kwargs):
        # TODO
        return self.module(*args, **kwargs)
