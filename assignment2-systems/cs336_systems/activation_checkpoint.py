import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
# num_layers for this model is 32

device = "cuda"

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048

block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length)).to(device)

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True, device=device)


total_size_bytes = 0


def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):  # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return


# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = block(x)


def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x


# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     total_size_bytes = 0
#     y = four_blocks(x)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")

from torch.utils.checkpoint import checkpoint


def two_blocks(x):
    x = block(x)
    x = block(x)
    return x


def four_blocks_checkoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    total_size_bytes = 0
    y = four_blocks_checkoint(x)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")
