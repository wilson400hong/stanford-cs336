import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    size = len(dataset)
    seqs = []
    targets = []

    starts = np.random.randint(0, size - context_length, size=batch_size)
    offsets = np.arange(context_length)
    idx = starts[:, None] + offsets[None, :]
    x = dataset[idx]
    y = dataset[idx + 1]

    x = torch.from_numpy(x).to(device, dtype=torch.long)
    y = torch.from_numpy(y).to(device, dtype=torch.long)
    return x, y

    # non-vectorized version (not efficient)
    # for _ in range(batch_size):
    #     start = random.randint(0, size - context_length - 1)
    #     window = dataset[start : start + context_length + 1]

    #     seqs.append(torch.tensor(window[:-1], dtype=torch.long, device=device))
    #     targets.append(torch.tensor(window[1:], dtype=torch.long, device=device))

    # return torch.stack(seqs), torch.stack(targets)
