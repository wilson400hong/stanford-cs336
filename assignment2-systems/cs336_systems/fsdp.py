import torch
import torch.distributed as dist


# TODO
class FSDPModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype):
        super().__init__()
        # TODO
        

    def rank0_print(self, msg):
        if self.rank == 0:
            print(msg)

  

    def forward(self, *args, **kwargs):
        # return self.module(*args, **kwargs)
