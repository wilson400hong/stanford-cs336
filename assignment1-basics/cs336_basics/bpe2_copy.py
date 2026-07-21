import collections
import heapq
import itertools

# from typing import BinaryIO, Collection
import json
import math
import multiprocessing
import os
import pickle
import time

import regex as re

from .bp_index_builder import BPIndexBuilder, parallel_build_bp_index
from .common import Timer

SPLIT_TOKEN = "<|endoftext|>"

PRETOKENIZE_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

NPROC = 16


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
    parts = re.split(pattern, s)  # use str.split() if memory bomb
    for part in parts:
        for m in re.finditer(PRETOKENIZE_PAT, part):
            counts[m.group()] += 1
    return dict(counts)


def pretokenize_chunk(
    file_path: str, start: int, end: int, special_tokens: list[str]
) -> dict[str, int]:
    with open(file_path, "rb") as f:
        f.seek(start)
        string = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize_str(string, special_tokens)


def parallel_pretokenize(
    input_path: str, num_processes: int, special_tokens: list[str]
) -> dict[str, int]:
    """
    Pre-tokenize a file into chunks that can be counted independently.
    """
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

    # DEBUG
    # print("First 10 Pretoken Counts:")
    # from itertools import islice

    # first_10 = dict(islice(total_counts.items(), 10))
    # for pretoken, count in first_10.items():
    #     print(f"{pretoken!r}: {count}")
    return total_counts


BytesTuple = tuple[bytes, ...]
BytesPair = tuple[bytes, bytes]


def str_to_bt(s: str) -> BytesTuple:
    # return tuple(bytes([b]) for b in s.encode("utf-8"))  # slower?
    b = s.encode("utf-8")
    return tuple(b[i : i + 1] for i in range(len(b)))


def bt_to_bps(bt: BytesTuple) -> list[BytesPair]:
    return list(zip(bt, bt[1:]))  # same as zip(bt[:-1], bt[1:])


# def get_max_bp(bp_counts: dict[BytesPair, int]) -> BytesPair:
#     return max(bp_counts, key=lambda bp: (bp_counts[bp], bp))


# TODO: Level 2 optimize
def merge_bp(bt: BytesTuple, bp: BytesPair) -> BytesTuple:
    merged: list[bytes] = []
    n = len(bt)
    i = 0
    while i < n:
        if i < n - 1 and bt[i] == bp[0] and bt[i + 1] == bp[1]:
            merged.append(bt[i] + bt[i + 1])
            i += 2
        else:
            merged.append(bt[i])
            i += 1
    return tuple(merged)


class BPHeapNode:
    __slots__ = ("count", "bp")

    def __init__(self, bp: BytesPair, count: int):
        self.bp = bp
        self.count = count

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.bp > other.bp


class BPHeap:
    def __init__(self):
        self.heap: list[BPHeapNode] = []

    def push(self, bp: BytesPair, count: int):
        heapq.heappush(self.heap, BPHeapNode(bp, count))

    def pop(self, bp_counts: dict[BytesPair, int]) -> BytesPair:
        while self.heap:
            node = heapq.heappop(self.heap)
            if bp_counts.get(node.bp, -1) == node.count:
                return node.bp
        return (bytes([0]), bytes([0]))  # should not happen


# TODO: can be optimized by C++
def build_bp_index(
    pretoken_counts: list[int], pretoken_to_bt: list[tuple[bytes, ...]]
) -> tuple[dict[BytesPair, int], dict[BytesPair, set[int]]]:
    bp_to_count: dict[BytesPair, int] = collections.defaultdict(int)
    bp_to_pretokens: dict[BytesPair, set[int]] = collections.defaultdict(set)
    for pretoken, bt in enumerate(pretoken_to_bt):
        if len(bt) < 2:
            continue
        cnt = pretoken_counts[pretoken]

        for bp in zip(bt, bt[1:]):
            bp_to_count[bp] += cnt
            bp_to_pretokens[bp].add(pretoken)
    return bp_to_count, bp_to_pretokens


def build_vocab_merges(
    pretoken_to_count: dict[str, int],
    vocab_size: int,
    special_tokens: list[str],
    use_cache: bool = True,  # TODO
) -> tuple[dict[int, bytes], list[BytesPair]]:
    # initialize vocab
    vocab: dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])

    with Timer("init pretoken_counts and pretoken_to_bt"):
        pretoken_counts = list(pretoken_to_count.values())
        pretokens = list(pretoken_to_count.keys())
        with multiprocessing.Pool(NPROC) as pool:
            pretoken_to_bt = pool.map(str_to_bt, pretokens, chunksize=10000)

    TMP_BP_CNT_FILE = "/data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/tmp/bp_to_count.json"
    TMP_BP_PT_FILE = "/data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/tmp/bp_to_pretokens.json"

    if use_cache:
        with open(TMP_BP_CNT_FILE, "rb") as f:
            bp_to_count = pickle.load(f)
        with open(TMP_BP_PT_FILE, "rb") as f:
            bp_to_pretokens = pickle.load(f)

    else:
        with Timer("build_bp_index"):
            bp_to_count, bp_to_pretokens = build_bp_index(
                pretoken_counts,
                pretoken_to_bt,
            )

        with open(TMP_BP_CNT_FILE, "wb") as f:
            pickle.dump(bp_to_count, f)
        with open(TMP_BP_PT_FILE, "wb") as f:
            pickle.dump(bp_to_pretokens, f)

    with Timer("init bp_heap"):
        bp_heap = BPHeap()
        for bp, count in bp_to_count.items():
            bp_heap.push(bp, count)

    print("#bp=", len(bp_to_count))

    # raise RuntimeError("stop here")

    # mergeing
    merges: list[BytesPair] = []
    print("start merging...")
    while len(vocab) < vocab_size:
        with Timer(f"round #{len(merges)}", False) as all_timer:
            # get max BP to merge
            with Timer("    get_max_bp"):
                bp_to_merge = bp_heap.pop(bp_to_count)

            # update merges and vocab
            merges.append(bp_to_merge)
            vocab[len(vocab)] = bp_to_merge[0] + bp_to_merge[1]

            # affected_pretokens = bp_to_pretokens[bp_to_merge].copy()
            affected_pretokens = bp_to_pretokens.pop(bp_to_merge)  # TODO: save a little

            latencies = collections.defaultdict(float)
            # collect affected pretokens' updates
            bps_to_update = set()

            with Timer("    update affected_pretokens"):
                for pretoken in affected_pretokens:
                    t0 = time.perf_counter()
                    cnt = pretoken_counts[pretoken]
                    old_bt = pretoken_to_bt[pretoken]
                    old_bps = bt_to_bps(old_bt)
                    t1 = time.perf_counter()
                    latencies["old_bps"] += t1 - t0

                    new_bt = merge_bp(old_bt, bp_to_merge)
                    t2 = time.perf_counter()
                    latencies["new_bt"] += t2 - t1

                    new_bps = bt_to_bps(new_bt)
                    t3 = time.perf_counter()
                    latencies["new_bps"] += t3 - t2

                    pretoken_to_bt[pretoken] = new_bt

                    # update bp_to_count
                    old_counts = collections.Counter(old_bps)
                    new_counts = collections.Counter(new_bps)

                    remove_bps = old_counts - new_counts
                    add_bps = new_counts - old_counts
                    for bp, diff in add_bps.items():
                        bps_to_update.add(bp)
                        bp_to_count[bp] += diff * cnt
                    for bp, diff in remove_bps.items():
                        bps_to_update.add(bp)
                        bp_to_count[bp] -= diff * cnt

                        if bp_to_count[bp] <= 0:
                            bp_to_count.pop(bp)
                    t4 = time.perf_counter()
                    latencies["bp_to_count"] += t4 - t3

                    # update bp_to_pretokens
                    old_bps_set = set(old_bps)
                    new_bps_set = set(new_bps)
                    to_remove = old_bps_set - new_bps_set
                    to_add = new_bps_set - old_bps_set
                    for bp in to_remove:
                        if bp != bp_to_merge and pretoken in bp_to_pretokens[bp]:
                            bp_to_pretokens[bp].remove(pretoken)
                    for bp in to_add:
                        bp_to_pretokens[bp].add(pretoken)

                    t5 = time.perf_counter()
                    latencies["bp_to_pretokens"] += t5 - t4

            print(f"    latency: {json.dumps(latencies, indent=8)}")
            # update bp_heap
            with Timer("    update bp_heap"):
                for bp in bps_to_update:
                    if bp in bp_to_count:
                        bp_heap.push(bp, bp_to_count[bp])
        print(
            f"Round #{len(merges)-1}. elapsed: {all_timer.elapsed}, merge={bp_to_merge}, #pretokens={len(affected_pretokens)}"
        )

    print(f"vocab size: {len(vocab)}, merges size: {len(merges)}")
    return vocab, merges


def build_vocab_merges_from_file(
    file_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[BytesPair]]:
    with Timer("load pretoken_to_count from file"):
        with open(file_path, "r") as f:
            pretoken_to_count = json.load(f)
    return build_vocab_merges(pretoken_to_count, vocab_size, special_tokens)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[BytesPair]]:
    with Timer("parallel_pretokenize"):
        pretoken_to_count = parallel_pretokenize(input_path, NPROC, special_tokens)
    with Timer("build_vocab_merges"):
        vocab, merges = build_vocab_merges(
            pretoken_to_count, vocab_size, special_tokens
        )
    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.index: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab = json.load(open(vocab_filepath, "r"))
        merges = json.load(open(merges_filepath, "r"))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # 1. split
        raise NotImplementedError("")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("")
