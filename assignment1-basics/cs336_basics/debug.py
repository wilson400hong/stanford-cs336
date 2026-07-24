import numpy as np
from cs336_basics.bpe import BPETokenizer, parallel_bpe_encode_file

TS_VOCAB = "data/tiny_stories/ts_vocab.pickle"
TS_MERGES = "data/tiny_stories/ts_merges.pickle"

TS_VALID_TXT = "data/TinyStoriesV2-GPT4-valid.txt"
TS_TRAIN_TXT = "data/TinyStoriesV2-GPT4-train.txt"

TS_VALID_TOKENS = "data/tiny_stories/ts_valid.bin"
TS_TRAIN_TOKENS = "data/tiny_stories/ts_train.bin"
TS_TRAIN_TOKENS_2 = "data/tiny_stories/ts_train2.bin"

"""
tokenizer = BPETokenizer.from_files(TS_VOCAB, TS_MERGES, ["<|endoftext|>"])

parallel_bpe_encode_file(TS_TRAIN_TXT, TS_TRAIN_TOKENS, 16, TS_VOCAB, TS_MERGES, ["<|endoftext|>"])
tokenizer.encode_file(TS_TRAIN_TXT,  TS_TRAIN_TOKENS_2)
"""


"""
a = np.fromfile("data/tiny_stories/ts_train.bin", dtype=np.uint16)
b = np.fromfile("data/tiny_stories/ts_train2.bin", dtype=np.uint16)

print(np.array_equal(a, b))


"""

"""
uv run python -m cs336_basics.train -t data/tiny_stories/ts_train.bin  -e data/tiny_stories/ts_valid.bin -o /tmp/ckpt_smoke --train_steps 50 --eval_interval 10 --eval_steps 5 --batch_size 16


uv run python -m cs336_basics.train -t /data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/data/tiny_stories/ts_train.bin  -e /data/users/wilsonhong/projects/stanford-cs336/assignment1-basics/data/tiny_stories/ts_valid.bin -o /tmp/ckpt_smoke --train_steps 50 --eval_interval 10 --eval_steps 5 --batch_size 16
"""
