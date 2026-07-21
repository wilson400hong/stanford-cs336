import collections

# from typing import BinaryIO, Collection
import json
import multiprocessing
import os
import time

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

    t3 = time.perf_counter()
    print(f"init bp_to_count and bp_to_pretokens done. elapsed: {t3-t2:.6f}s")

    # mergeing
    print("start merging...")
    while len(vocab) < vocab_size:
        # get max BP to merge
        t10 = time.perf_counter()
        bp_to_merge = get_max_bp(bp_to_count)
        print(f"  round #{len(merges)} merge {bp_to_merge}")
        t11 = time.perf_counter()
        print(f"  get_max_bp: elapsed {t11-t10:.6f}s")

        # update merges and vocab
        merges.append(bp_to_merge)
        vocab[len(vocab)] = bp_to_merge[0] + bp_to_merge[1]
        t12 = time.perf_counter()
        print(f"  update merges and vocab: elapsed {t12-t11:.6f}s")

        affected_pretokens = bp_to_pretokens[bp_to_merge].copy()
        latencies = collections.defaultdict(float)
        # TODO: this part toooooo slow
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
            for bp in new_bps:
                bp_to_count[bp] += cnt
            for bp in old_bps:
                bp_to_count[bp] -= cnt
                if bp_to_count[bp] <= 0:
                    bp_to_count.pop(bp)

            # update bp_to_pretokens
            te = time.perf_counter()
            latencies["bp_to_count"] += te - td
            # print(f"  update bp_to_count: elapsed {te-tb:.6f}s")

            # print(f"  update bp_to_pretokens: elapsed {te-tb:.6f}s")
            # print(f"  update bp_to_pretokens: elapsed {te-tb:.6f}s")
            #
            old_bps_set = set(old_bps)
            new_bps_set = set(new_bps)
            to_remove = old_bps_set - new_bps_set
            to_add = new_bps_set - old_bps_set
            for bp in to_remove:
                if pretoken in bp_to_pretokens[bp]:
                    bp_to_pretokens[bp].remove(pretoken)
            for bp in to_add:
                bp_to_pretokens[bp].add(pretoken)

            te = time.perf_counter()
            latencies["bp_to_pretokens"] += te - td

        t13 = time.perf_counter()
        # print(f"  round takes  {len(affected_pretokens)} elapsed {t13-t10:.6f}s")
        print(
            f"  update affected_pretokens {len(affected_pretokens)} elapsed {t13-t12:.6f}s, latencies: {latencies}"
        )

    t4 = time.perf_counter()

    print(f"all done . Elapsed: {t4 - t3}s")
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


# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[BytesPair]]:
#     # 1. get pretoken counts

#     print("Running pretokenization...")
#     t0 = time.perf_counter()
#     pretoken_to_count = parallel_pretokenize(input_path, NPROC, special_tokens)
#     t1 = time.perf_counter()
#     print(f"Parallel pretokenization done. Elapsed: {t1 - t0}s")

#     # 2. initialize vacab
#     vocab: dict[int, bytes] = {}
#     for special_token in special_tokens:
#         vocab[len(vocab)] = special_token.encode("utf-8")
#     for b in range(256):
#         vocab[len(vocab)] = bytes([b])

#     # NIT: merge into one dict to simplify
#     pretoken_to_bt = {pretoken: str_to_bt(pretoken) for pretoken in pretoken_to_count}
#     # NIT: merge into one dict to simplify
#     bp_to_count: dict[BytesPair, int] = collections.defaultdict(int)
#     bp_to_pretokens: dict[BytesPair, set[str]] = collections.defaultdict(set)

#     # 3. train & merge
#     merges: list[BytesPair] = []
#     print("Running BPE training...")
#     t2 = time.perf_counter()
#     for pretoken, bt in pretoken_to_bt.items():
#         cnt = pretoken_to_count[pretoken]
#         bps = bt_to_bps(bt)
#         for bp in bps:
#             bp_to_count[bp] += cnt
#             bp_to_pretokens[bp].add(pretoken)

#     while len(vocab) < vocab_size:
#         # get max BP to merge
#         bp_to_merge = get_max_bp(bp_to_count)
#         print(f"  round #{len(merges)}: merging {bp_to_merge}")

#         # update merges and vocab
#         merges.append(bp_to_merge)
#         vocab[len(vocab)] = bp_to_merge[0] + bp_to_merge[1]

#         affected_pretokens = bp_to_pretokens[bp_to_merge].copy()
#         for pretoken in affected_pretokens:
#             cnt = pretoken_to_count[pretoken]
#             old_bt = pretoken_to_bt[pretoken]
#             old_bps = bt_to_bps(old_bt)

#             new_bt = merge_bp(old_bt, bp_to_merge)
#             new_bps = bt_to_bps(new_bt)

#             pretoken_to_bt[pretoken] = new_bt

#             # update bp_to_count to bp_to_pretokens
#             bp_delta: dict[bytes, int] = collections.defaultdict(int)
#             for bp in new_bps:
#                 bp_delta[bp] += 1
#             for bp in old_bps:
#                 bp_delta[bp] -= 1

#             for bp, delta in bp_delta.items():
#                 if delta == 0:
#                     continue
#                 bp_to_count[bp] += cnt * delta

#             old_bps_set = set(old_bps)
#             new_bps_set = set(new_bps)
#             to_remove = old_bps_set - new_bps_set
#             to_add = new_bps_set - old_bps_set
#             for bp in to_remove:
#                 if pretoken in bp_to_pretokens[bp]:
#                     bp_to_pretokens[bp].remove(pretoken)
#             for bp in to_add:
#                 bp_to_pretokens[bp].add(pretoken)

#             # for bp in old_bps:
#             #     if pretoken in bp_to_pretokens[bp]:
#             #         bp_to_pretokens[bp].remove(pretoken)
#             #     bp_to_count[bp] -= cnt

#             # for bp in new_bps:
#             #     bp_to_pretokens[bp].add(pretoken)
#             #     bp_to_count[bp] += cnt

#     t3 = time.perf_counter()
#     print(f"vocab size: {len(vocab)}, merges size: {len(merges)}")
#     print(f"BPE training done. Elapsed: {t3 - t2}s")
#     return vocab, merges


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
