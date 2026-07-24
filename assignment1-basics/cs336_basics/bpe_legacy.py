import collections
import heapq
import json
import multiprocessing
import os
import pickle
import time
from typing import Iterable, Iterator

import regex as re

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
    print("parallel_pretokenize start...")
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

    # DEBUG
    # print("First 10 Pretoken Counts:")
    # from itertools import islice

    # first_10 = dict(islice(total_counts.items(), 10))
    # for pretoken, count in first_10.items():
    #     print(f"{pretoken!r}: {count}")
    t1 = time.perf_counter()
    print(f"parallel_pretokenize done. elapsed: {t1 - t0}s")
    return total_counts


BytesTuple = tuple[bytes, ...]
BytesPair = tuple[bytes, bytes]


def str_to_bt(s: str) -> BytesTuple:
    # return tuple(bytes([b]) for b in s.encode("utf-8"))  # slower?
    b = s.encode("utf-8")
    return tuple(b[i : i + 1] for i in range(len(b)))


def bt_to_bps(bt: BytesTuple) -> list[BytesPair]:
    return list(zip(bt, bt[1:]))  # same as zip(bt[:-1], bt[1:])


def get_max_bp(bp_counts: dict[BytesPair, int]) -> BytesPair:
    return max(bp_counts, key=lambda bp: (bp_counts[bp], bp))
    # ans = (bytes([0]), bytes([0]))
    # max = -1
    # for bp, count in bp_counts.items():
    #     if count > max or (count == max and bp > ans):
    #         max = count
    #         ans = bp
    # # print(f"Max BP: {ans}, count: {max}")
    # return ans


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


def build_vocab_merges_from_file(
    file_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[BytesPair]]:
    with open(file_path, "r") as f:
        pretoken_to_count = json.load(f)
    print(f"load pretoken_to_count from file {file_path} done")
    return build_vocab_merges(pretoken_to_count, vocab_size, special_tokens)


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


# TODO: optimization
def build_vocab_merges(
    pretoken_to_count: dict[str, int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[BytesPair]]:
    # initialize vacab
    t0 = time.perf_counter()
    vocab: dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])

    t1 = time.perf_counter()
    print(f"init vocab done. elapsed: {t1-t0:.6f}s")

    # NIT: merge into one dict to simplify
    # NOTE: now pretoken use idx (int)

    pretoken_counts = [pretoken_to_count[pretoken] for pretoken in pretoken_to_count]
    pretoken_to_bt = [str_to_bt(pretoken) for pretoken in pretoken_to_count]

    # NIT: merge into one dict to simplify
    bp_to_count: dict[BytesPair, int] = collections.defaultdict(int)
    bp_to_pretokens: dict[BytesPair, set[int]] = collections.defaultdict(set)

    t2 = time.perf_counter()
    print(f"init pretoken_to_bt done. elapsed: {t2-t1:.6f}s")

    merges: list[BytesPair] = []

    for pretoken, bt in enumerate(pretoken_to_bt):
        cnt = pretoken_counts[pretoken]
        bps = bt_to_bps(bt)
        for bp in bps:
            bp_to_count[bp] += cnt
            bp_to_pretokens[bp].add(pretoken)

    # TODO:
    bp_heap = BPHeap()
    for bp, count in bp_to_count.items():
        bp_heap.push(bp, count)

    t3 = time.perf_counter()
    print(f"init bp_to_count and bp_to_pretokens done. elapsed: {t3-t2:.6f}s")

    # mergeing
    print("start merging...")
    while len(vocab) < vocab_size:
        # get max BP to merge
        t10 = time.perf_counter()

        # TODO: use bp_heap
        # bp_to_merge = get_max_bp(bp_to_count)
        bp_to_merge = bp_heap.pop(bp_to_count)
        t11 = time.perf_counter()

        print(f"  round #{len(merges)} merge {bp_to_merge}")
        print(f"  get_max_bp: elapsed {t11-t10:.6f}s")

        # update merges and vocab
        merges.append(bp_to_merge)
        vocab[len(vocab)] = bp_to_merge[0] + bp_to_merge[1]
        t12 = time.perf_counter()
        # print(f"  update merges and vocab: elapsed {t12-t11:.6f}s")

        affected_pretokens = bp_to_pretokens[bp_to_merge].copy()
        latencies = collections.defaultdict(float)

        # collect affected pretokens' updates
        bps_to_update = set()
        for pretoken in affected_pretokens:
            ta = time.perf_counter()
            cnt = pretoken_counts[pretoken]
            old_bt = pretoken_to_bt[pretoken]
            old_bps = bt_to_bps(old_bt)
            tb = time.perf_counter()

            latencies["old_bps"] += tb - ta

            new_bt = merge_bp(old_bt, bp_to_merge)
            tc = time.perf_counter()
            latencies["new_bt"] += tc - tb

            new_bps = bt_to_bps(new_bt)
            td = time.perf_counter()

            latencies["new_bps"] += td - tc

            pretoken_to_bt[pretoken] = new_bt

            # update bp_to_count
            old_counts = collections.Counter(old_bps)
            new_counts = collections.Counter(new_bps)

            remove_bps = old_counts - new_counts
            add_bps = new_counts - old_counts
            for bp, diff in add_bps.items():
                bps_to_update.add(bp)
                bp_to_count[bp] += diff * cnt
                # bp_heap.push(bp, bp_to_count[bp])  # TODO
            for bp, diff in remove_bps.items():
                bps_to_update.add(bp)
                bp_to_count[bp] -= diff * cnt

                if bp_to_count[bp] <= 0:
                    bp_to_count.pop(bp)

            # update bp_to_pretokens
            te = time.perf_counter()
            latencies["bp_to_count"] += te - td

            old_bps_set = set(old_bps)
            new_bps_set = set(new_bps)
            to_remove = old_bps_set - new_bps_set
            to_add = new_bps_set - old_bps_set
            for bp in to_remove:
                if pretoken in bp_to_pretokens[bp]:
                    bp_to_pretokens[bp].remove(pretoken)
            for bp in to_add:
                bp_to_pretokens[bp].add(pretoken)

            tf = time.perf_counter()
            latencies["bp_to_pretokens"] += tf - te

        # TODO: update bp_heap
        for bp in bps_to_update:
            if bp in bp_to_count:
                bp_heap.push(bp, bp_to_count[bp])

        t13 = time.perf_counter()
        # print(f"  round takes  {len(affected_pretokens)} elapsed {t13-t10:.6f}s")
        print(
            f"  update affected_pretokens {len(affected_pretokens)}, elapsed {t13-t12:.6f}s, latencies: {latencies}"
        )

    t4 = time.perf_counter()

    print(f"all done. elapsed: {t4 - t3}s")
    print(f"vocab size: {len(vocab)}, merges size: {len(merges)}")
    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[BytesPair]]:
    pretoken_to_count = parallel_pretokenize(input_path, NPROC, special_tokens)

    return build_vocab_merges(pretoken_to_count, vocab_size, special_tokens)


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
                    res.extend(self.encode_pretoken(m.group()))
        return res

    def encode_pretoken(self, pretoken: str) -> list[int]:
        """pretoken should never be a special token"""
        assert pretoken not in self.special_tokens

        # TODO: measure
        if pretoken in self.pretoken_cache:
            return self.pretoken_cache[pretoken]

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
        for text in iterable:
            yield from self.encode(text)

    def encode_file(self, in_file: str, out_file: str):
        # write as np array binary format, uint16
        buffer = []
        BUFFER_MAX = 1e5
        count = 0
        total = 0
        t0 = time.perf_counter()
        with open(in_file, "r") as f:
            with open(out_file, "wb") as g:
                for token in self.encode_iterable(f):
                    if count < 20:
                        print(token, self.vocab[token])
                    # assert token < 2**16, "token id > uint16"
                    buffer.append(token)
                    if len(buffer) >= BUFFER_MAX:
                        total += len(buffer)
                        elapsed = time.perf_counter() - t0
                        # print(f"elapsed: {elapsed}, tps: {len(buffer) / elapsed}")
                        t0 = time.perf_counter()
                        # print(f"write to buffer: {len(buffer)}. total: {total}")
                        # append buffer to file
                        ndarray = np.array(buffer, dtype=np.uint16)
                        ndarray.tofile(g)
                        buffer = []

                    count += 1
                if buffer:
                    ndarray = np.array(buffer, dtype=np.uint16)
                    ndarray.tofile(g)
                    total += len(buffer)

        print(f"total written: {total}")

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[id] for id in ids)
        return b.decode("utf-8", errors="replace")
