import collections
import multiprocessing
import os
import time
from typing import BinaryIO, Collection

import regex as re

SPLIT_TOKEN = "<|endoftext|>"

PRETOKENIZE_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_chunk_boundaries(
    input_path: str,
    desired_num_chunks: int,
    split_token: str,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    split_token_bytes = split_token.encode("utf-8")
    assert isinstance(
        split_token_bytes, bytes
    ), "Must represent special token as a bytestring"

    with open(input_path, "rb") as file:
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
                found_at = mini_chunk.find(split_token_bytes)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


def pretokenize_str(s: str, special_tokens: list[str]) -> dict[str, int]:
    pattern = "|".join([re.escape(token) for token in special_tokens])
    counts = collections.Counter()
    parts = re.split(pattern, s)  # use s.split() if memory bomb
    for part in parts:
        for m in re.finditer(PRETOKENIZE_PAT, part):
            counts[m.group()] += 1
    return dict(counts)


def pretokenize_chunk(
    file_path: str, start: int, end: int, special_tokens: list[str]
) -> dict[str, int]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize_str(chunk, special_tokens)


def parallel_pretokenize(
    input_path: str, num_processes: int, special_tokens: list[str]
) -> dict[str, int]:
    """
    Pre-tokenize a file into chunks that can be counted independently.
    """
    t0 = time.perf_counter()
    boundaries = get_chunk_boundaries(
        input_path, num_processes, split_token=SPLIT_TOKEN
    )
    chunks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with multiprocessing.Pool(num_processes) as pool:
        partial_counts = pool.starmap(pretokenize_chunk, chunks)

    total_counts = collections.Counter()
    for partial_count in partial_counts:
        total_counts.update(partial_count)

    total_counts = dict(total_counts)

    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0}s")

    # DEBUG
    print("First 10 Pretoken Counts:")
    from itertools import islice

    first_10 = dict(islice(total_counts.items(), 10))
    for pretoken, count in first_10.items():
        print(f"{pretoken!r}: {count}")

    return total_counts  # TODO: sorted to ensure correctness


BytesTuple = tuple[bytes, ...]
BytesPair = tuple[bytes, bytes]


def str_to_bt(s: str) -> BytesTuple:
    return tuple(bytes([b]) for b in s.encode("utf-8"))


def counts_to_bt(counts: dict[str, int]) -> dict[tuple[bytes], int]:
    return {str_to_bt(k): v for k, v in counts.items()}


def bt_to_bps(bt: BytesTuple) -> list[BytesPair]:
    return list(zip(bt[:-1], bt[1:]))


def get_max_bp(bp_counts: dict[BytesPair, int]) -> BytesPair:
    ans = (bytes([0]), bytes([0]))
    max = -1
    for bp, count in bp_counts.items():
        if count > max or (count == max and bp > ans):
            max = count
            ans = bp
    # print(f"Max BP: {ans}, count: {max}")
    return ans


def merge_bp(bt: BytesTuple, bp: BytesPair) -> BytesTuple:
    merged: list[bytes] = []
    n = len(bt)
    i = 0
    while i < n:
        if i < n - 1 and (bt[i], bt[i + 1]) == bp:
            merged.append(bt[i] + bt[i + 1])
            i += 2
        else:
            merged.append(bt[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[BytesPair]]:
    # 1. get pretoken counts
    num_processes = 16
    pretoken_to_count = parallel_pretokenize(input_path, num_processes, special_tokens)

    # 2. initialize vacab
    vocab: dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])

    # NIT: merge into one dict to simplify
    pretoken_to_bt = {pretoken: str_to_bt(pretoken) for pretoken in pretoken_to_count}
    # NIT: merge into one dict to simplify
    bp_to_count: dict[BytesPair, int] = collections.defaultdict(int)
    bp_to_pretokens: dict[BytesPair, set[str]] = collections.defaultdict(set)

    # 3. train & merge
    merges: list[BytesPair] = []

    for pretoken, bt in pretoken_to_bt.items():
        cnt = pretoken_to_count[pretoken]
        bps = bt_to_bps(bt)
        for bp in bps:
            bp_to_count[bp] += cnt
            bp_to_pretokens[bp].add(pretoken)

    while len(vocab) < vocab_size:
        # 1. get max BP to merge
        bp_to_merge = get_max_bp(bp_to_count)

        # 2. update merges and vocab
        merges.append(bp_to_merge)
        vocab[len(vocab)] = bp_to_merge[0] + bp_to_merge[1]

        affected_pretokens = bp_to_pretokens[bp_to_merge].copy()
        for pretoken in affected_pretokens:
            cnt = pretoken_to_count[pretoken]
            old_bt = pretoken_to_bt[pretoken]
            old_bps = bt_to_bps(old_bt)

            new_bt = merge_bp(old_bt, bp_to_merge)
            new_bps = bt_to_bps(new_bt)

            pretoken_to_bt[pretoken] = new_bt

            # update bp_to_count to bp_to_pretokens

            for bp in old_bps:
                if pretoken in bp_to_pretokens[bp]:
                    bp_to_pretokens[bp].remove(pretoken)
                bp_to_count[bp] -= cnt

            for bp in new_bps:
                bp_to_pretokens[bp].add(pretoken)
                bp_to_count[bp] += cnt

    return vocab, merges


# class BPETokenizer:
#     def __init__(self):
#         pass

#     def encode(self):
#         pass

#     def decode(self):
#         pass
