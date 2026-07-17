import collections
import multiprocessing
import os
import time
from typing import BinaryIO, Collection

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def find_file_chunk_boundaries(
    file_path: str,
    num_processes: int,
) -> list[int]:
    with open(file_path, "rb") as f:
        return find_chunk_boundaries(f, num_processes, b"<|endoftext|>")


def pretokenize(file_path: str, start: int, end: int) -> dict[str, int]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        counts = collections.Counter()
        for m in re.finditer(PAT, chunk):
            counts[m.group()] += 1
        return dict(counts)


def parallel_pretokenize(file_path: str, num_processes: int) -> dict[str, int]:
    """
    Pre-tokenize a file into chunks that can be counted independently.
    """
    t0 = time.perf_counter()
    boundaries = find_file_chunk_boundaries(file_path, num_processes)
    jobs = [
        (file_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with multiprocessing.Pool(num_processes) as pool:
        partial_counts = pool.starmap(pretokenize, jobs)

    total_counts = collections.Counter()
    for partial_count in partial_counts:
        total_counts.update(partial_count)

    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0}s")
    print("First 10 Pretoken Counts:")
    from itertools import islice

    first_10 = dict(islice(total_counts.items(), 10))
    for pretoken, count in first_10.items():
        print(f"{pretoken!r}: {count}")

    return dict(total_counts)


## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
