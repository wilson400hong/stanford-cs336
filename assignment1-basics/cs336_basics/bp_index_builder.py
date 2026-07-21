import array
import collections
import itertools
import math
import multiprocessing

_worker_bts = None
_worker_counts = None


def _init_worker(bts, counts):
    global _worker_bts, _worker_counts
    _worker_bts = bts
    _worker_counts = counts


def _init_chunk(start_idx: int, end_idx: int):
    bts = _worker_bts
    counts = _worker_counts

    local_count = collections.defaultdict(int)
    local_pretokens = collections.defaultdict(set)
    pairwise = itertools.pairwise

    for i in range(start_idx, end_idx):
        bt = bts[i]
        if len(bt) < 2:
            continue

        cnt = counts[i]
        for bp in pairwise(bt):
            local_count[bp] += cnt
            local_pretokens[bp].add(i)

    # 🔥 核心優化：在回傳前，將龐大零碎的 set 轉成 C 型態的 array('i', ...) (i 代表整數)
    # 這樣 pickling 的時候只需要傳輸一塊連續的 bytes，速度快上千倍！
    packed_pretokens = {
        bp: array.array("i", pts) for bp, pts in local_pretokens.items()
    }

    return dict(local_count), packed_pretokens


class BPIndexBuilder:
    def __init__(
        self,
        pretoken_counts: list[int],
        pretoken_to_bt: list[tuple[bytes, ...]],
        num_workers: int,
    ):
        self.counts = pretoken_counts
        self.bts = pretoken_to_bt
        self.num_workers = num_workers

    def build(self) -> tuple[dict, dict]:
        total_len = len(self.bts)
        chunk_size = math.ceil(total_len / self.num_workers)

        tasks = [
            (i * chunk_size, min((i + 1) * chunk_size, total_len))
            for i in range(self.num_workers)
        ]

        with multiprocessing.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.bts, self.counts),
        ) as pool:
            # 這裡的速度會飆升，因為回傳的資料被極度壓縮了
            results = pool.starmap(_init_chunk, tasks)

        bp_to_count = collections.defaultdict(int)
        bp_to_pretokens = collections.defaultdict(set)

        # 主程式解包合併
        for local_count, local_pretokens in results:
            for bp, cnt in local_count.items():
                bp_to_count[bp] += cnt

            # set.update() 可以直接吃 array，底層會用 C 語言極速合併
            for bp, packed_pts in local_pretokens.items():
                bp_to_pretokens[bp].update(packed_pts)

        return dict(bp_to_count), dict(bp_to_pretokens)


def _init_chunk(start_idx: int, bts_chunk: list, counts_chunk: list):
    local_count = collections.defaultdict(int)
    local_pretokens = collections.defaultdict(set)

    for offset, bt in enumerate(bts_chunk):
        if len(bt) < 2:
            continue

        cnt = counts_chunk[offset]
        pretoken_idx = start_idx + offset  # 還原全域的 pretoken index

        for bp in zip(bt, bt[1:]):
            local_count[bp] += cnt
            local_pretokens[bp].add(pretoken_idx)

    return local_count, local_pretokens


def parallel_build_bp_index(
    pretoken_counts: list[int],
    pretoken_to_bt: list[tuple[bytes, ...]],
    num_workers: int,
) -> tuple[dict, dict]:
    total_len = len(pretoken_to_bt)
    chunk_size = math.ceil(total_len / num_workers)

    tasks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        tasks.append(
            (
                start_idx,
                pretoken_to_bt[start_idx:end_idx],
                pretoken_counts[start_idx:end_idx],
            )
        )

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(_init_chunk, tasks)

    bp_to_count = collections.defaultdict(int)
    bp_to_pretokens = collections.defaultdict(set)

    for local_count, local_pretokens in results:
        for bp, cnt in local_count.items():
            bp_to_count[bp] += cnt

        for bp, pts in local_pretokens.items():
            bp_to_pretokens[bp].update(pts)

    return bp_to_count, bp_to_pretokens
