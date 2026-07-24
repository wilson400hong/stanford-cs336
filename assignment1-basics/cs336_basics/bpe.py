import collections
import heapq
import itertools

# from typing import BinaryIO, Collection
import json
import multiprocessing
import os
import pickle
import shutil
import time
from typing import Iterable, Iterator

import numpy as np
import regex as re
from torch.nn.utils._expanded_weights.conv_utils import THRESHOLD

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
    if special_tokens:
        pattern = "|".join([re.escape(token) for token in special_tokens])
        parts = re.split(pattern, s)  # use str.split() if memory bomb
    else:
        parts = [s]

    counts = collections.Counter()
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
    b = s.encode("utf-8")
    return tuple(b[i : i + 1] for i in range(len(b)))


def bt_to_bps(bt: BytesTuple) -> list[BytesPair]:
    return list(zip(bt, bt[1:]))  # same as zip(bt[:-1], bt[1:])


# def get_max_bp(bp_counts: dict[BytesPair, int]) -> BytesPair:
#     return max(bp_counts, key=lambda bp: (bp_counts[bp], bp))


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


def merge_bp(bt: BytesTuple, bp: BytesPair):
    M = bp[0] + bp[1]
    res: list[bytes] = []
    delta: dict[BytesPair, int] = collections.defaultdict(int)

    n = len(bt)
    i = 0
    while i < n:
        if i < n - 1 and bt[i] == bp[0] and bt[i + 1] == bp[1]:
            delta[bp] -= 1
            if len(res) > 0:
                L_b0 = (res[-1], bt[i])
                delta[L_b0] -= 1
                L_M = (res[-1], M)
                delta[L_M] += 1
            if i + 2 < n:
                b1_R = (bt[i + 1], bt[i + 2])
                delta[b1_R] -= 1
                M_R = (M, bt[i + 2])
                delta[M_R] += 1
            res.append(M)
            i += 2
        else:
            res.append(bt[i])
            i += 1

    # old_present = set(itertools.pairwise(bt))
    new_present = set(itertools.pairwise(res))

    return tuple(res), delta, new_present


def build_vocab_merges(
    pretoken_to_count: dict[str, int],
    vocab_size: int,
    special_tokens: list[str],
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

    with Timer("build_bp_index"):
        bp_to_count, bp_to_pretokens = build_bp_index(
            pretoken_counts,
            pretoken_to_bt,
        )

    with Timer("init bp_heap"):
        bp_heap = BPHeap()
        for bp, count in bp_to_count.items():
            bp_heap.push(bp, count)

    print("#bp=", len(bp_to_count))

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

                    new_bt, delta, new_present = merge_bp(old_bt, bp_to_merge)
                    pretoken_to_bt[pretoken] = new_bt
                    t1 = time.perf_counter()
                    latencies["merge_bp"] += t1 - t0

                    # update bp_to_count
                    for bp, diff in delta.items():
                        if diff == 0:
                            continue
                        bps_to_update.add(bp)
                        bp_to_count[bp] += diff * cnt
                        if bp_to_count[bp] <= 0:
                            bp_to_count.pop(bp)
                        if diff > 0:
                            bp_to_pretokens[bp].add(pretoken)
                        if diff < 0 and bp not in new_present:
                            if bp != bp_to_merge and pretoken in bp_to_pretokens[bp]:
                                bp_to_pretokens[bp].remove(pretoken)

                    t2 = time.perf_counter()
                    latencies["bp_to_count and bp_to_pretokens"] += t2 - t1

                    # update bp_to_pretokens

                    # to_remove = old_present - new_present
                    # to_add = new_present - old_present
                    # for bp in to_remove:
                    #     if bp != bp_to_merge and pretoken in bp_to_pretokens[bp]:
                    #         bp_to_pretokens[bp].remove(pretoken)
                    # for bp in to_add:
                    #     bp_to_pretokens[bp].add(pretoken)

                    # t3 = time.perf_counter()
                    # latencies["bp_to_pretokens"] += t3 - t2

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
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens_pattern = (
                "("
                + "|".join([f"{re.escape(token)}" for token in self.special_tokens])
                + ")"
            )
        else:
            self.special_tokens = []
            self.special_tokens_pattern = None
        # print(self.special_tokens_pattern)
        self.bytes_index: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {m: rank for rank, m in enumerate(self.merges)}
        self.pretoken_cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab = pickle.load(open(vocab_filepath, "rb"))
        merges = pickle.load(open(merges_filepath, "rb"))
        return cls(vocab, merges, special_tokens)

    # TODO: need ot optimize. too slow
    def encode(self, text: str) -> list[int]:
        res = []
        if self.special_tokens_pattern:
            parts = re.split(self.special_tokens_pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:  # TODO: this might can be faster
                res.append(self.bytes_index[part.encode("utf-8")])
            else:
                for m in re.finditer(PRETOKENIZE_PAT, part):
                    w = m.group()
                    if w in self.pretoken_cache:
                        res.extend(self.pretoken_cache[w])
                    else:
                        res.extend(self.encode_pretoken(w))
        return res

    def encode_pretoken(self, pretoken: str) -> list[int]:
        """pretoken should never be a special token"""
        # assert pretoken not in self.special_tokens
        bt = str_to_bt(pretoken)
        done = False

        BIG = len(self.merges) + 10000  # much larger than |vocab|

        while len(bt) >= 2 and not done:
            done = True
            bp_to_merge = None
            merge_rank = BIG  # very big

            for bp in itertools.pairwise(bt):
                if (rank := self.merge_rank.get(bp, BIG)) < merge_rank:
                    done = False
                    merge_rank = rank
                    bp_to_merge = bp

            if bp_to_merge:
                bt, _, _ = merge_bp(bt, bp_to_merge)

        tokens = [self.bytes_index[b] for b in bt]
        self.pretoken_cache[pretoken] = tokens
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        THRESHOLD = 1e5
        batch = []
        size = 0
        has_split_token = False
        for line in iterable:
            batch.append(line)
            has_split_token |= SPLIT_TOKEN in line
            size += len(line)
            if size > THRESHOLD and has_split_token:
                text = "".join(batch)
                pos = text.rfind(SPLIT_TOKEN)
                pos += len(SPLIT_TOKEN)
                to_encode = text[:pos]
                remain = text[pos:]
                has_split_token = SPLIT_TOKEN in remain
                batch = [remain]
                size = len(remain)

                yield from self.encode(to_encode)
        if batch:
            yield from self.encode("".join(batch))

    def encode_file(self, in_file: str, out_file: str, num_processes: int = 1):
        # write as np array binary format, uint16
        buffer = []
        BUFFER_MAX = 1e6
        total = 0
        t0 = time.perf_counter()
        with open(in_file, "r") as f, open(out_file, "wb") as g:
            for token in self.encode_iterable(f):
                # assert token < 2**16, "token id > uint16"
                buffer.append(token)
                if len(buffer) >= BUFFER_MAX:
                    total += len(buffer)
                    elapsed = time.perf_counter() - t0
                    print(f"elapsed: {elapsed}, tps: {len(buffer) / elapsed}")
                    t0 = time.perf_counter()
                    print(f"write to buffer: {len(buffer)}. total: {total}")
                    # append buffer to file
                    ndarray = np.array(buffer, dtype=np.uint16)
                    ndarray.tofile(g)
                    buffer = []
            if buffer:
                ndarray = np.array(buffer, dtype=np.uint16)
                ndarray.tofile(g)
                total += len(buffer)

        print(f"total written: {total}")

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[id] for id in ids)
        return b.decode("utf-8", errors="replace")


def bpe_encode_chunk(
    idx: int,
    in_file: str,
    out_file_prefix: str,  # use for prefix
    start: int,
    end: int,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None,
):
    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    out_file = f"{out_file_prefix}_{idx}.bin"

    with open(in_file, "rb") as f, open(out_file, "wb") as g:
        t0 = time.perf_counter()

        f.seek(start)
        string = f.read(end - start).decode("utf-8", errors="ignore")

        tokens = tokenizer.encode(string)
        ndarray = np.array(tokens, dtype=np.uint16)
        ndarray.tofile(g)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(
            f"chunk {idx}, elapsed: {elapsed:.6f}, encode: {len(tokens)}, tps: {len(tokens) / elapsed:.6f}"
        )

    return len(tokens)


def parallel_bpe_encode_file(
    in_file: str,
    out_file: str,
    num_processes: int,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None,
):
    boundaries = get_chunk_boundaries(in_file, num_processes, split_token=SPLIT_TOKEN)

    chunks = [
        (
            idx,
            in_file,
            out_file,
            start,
            end,
            vocab_filepath,
            merges_filepath,
            special_tokens,
        )
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    with multiprocessing.Pool(num_processes) as pool:
        totals = pool.starmap(bpe_encode_chunk, chunks)

    total = sum(totals)
    print(f"total encode {total} tokens")

    print(f"merge to file: {out_file}")
    with open(out_file, "wb") as f:
        for idx in range(len(chunks)):
            shard_file = f"{out_file}_{idx}.bin"
            with open(shard_file, "rb") as shard:
                shutil.copyfileobj(shard, f)

            os.remove(shard_file)


# TESTING

# vocab1, merges1 = build_vocab_merges_from_file("/data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/tmp/owt_pretoken_counts.json", 400, ["<|endoftext|>"], True)

# build_vocab_merges({"low":5, "lower":2, "newest":6, "widest":3}, 261, [], use_cache=False)
# # A → [(b's',b't'), (b'e',b'st'), (b'o',b'w'), (b'l',b'ow'), (b'w',b'est')]

# build_vocab_merges({"aaaa":3, "xx":5}, 258, [], use_cache=False)
# # B → [(b'a',b'a'), (b'x',b'x')]

# build_vocab_merges({"ababab":4}, 259, [], use_cache=False)
# # C → [(b'a',b'b'), (b'ab',b'ab'), (b'abab',b'ab')]


# tokenizer = BPETokenizer.from_files("/data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/tmp/owt_vocab.pickle", "/data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/tmp/owt_merges.pickle", ["<|endoftext|>"])


# Reading (for training):
#   - np.memmap(path, dtype=np.uint16, mode="r") — on-disk array, flat 1-D. Docs: numpy.memmap
#   (https://numpy.org/doc/stable/reference/generated/numpy.memmap.html). Pass the same dtype you
#   wrote.
#   - (alt) np.fromfile(path, dtype=np.uint16) — reads the whole thing into RAM; fine for a quick
#   sanity check, not for the big file.
#   - torch.from_numpy(...) then .long() — cast the sampled batch (small), not the whole array.

#   Your generator side (already built):
#   - tokenizer.encode_iterable(f) → Iterator[int].

# from cs336_basics.bpe import BPETokenizer

# tokenizer = BPETokenizer.from_files(
#     "data/tiny_stories/ts_vocab.pickle",
#     "data/tiny_stories/ts_merges.pickle",
#     ["<|endoftext|>"],
# )
# tokenizer.encode_file(
#     "data/TinyStoriesV2-GPT4-valid.txt", "data/tiny_stories/ts_valid.bin"
# )
