from cs336_basics.bpe import BPETokenizer, parallel_bpe_encode_file


TS_VOCAB = "data/tiny_stories/ts_vocab.pickle"
TS_MERGES = "data/tiny_stories/ts_merges.pickle"

TS_VALID_TXT = "data/TinyStoriesV2-GPT4-valid.txt"
TS_TRAIN_TXT = "data/TinyStoriesV2-GPT4-train.txt"

TS_VALID_TOKENS = "data/tiny_stories/ts_valid.bin"
TS_TRAIN_TOKENS = "data/tiny_stories/ts_train.bin"

"""
tokenizer = BPETokenizer.from_files(TS_VOCAB, TS_MERGES, ["<|endoftext|>"],)

parallel_bpe_encode_file(TS_TRAIN_TXT, TS_TRAIN_TOKENS, 16, TS_VOCAB, TS_MERGES, ["<|endoftext|>"])
tokenizer.encode_file(TS_VALID_TXT,  TS_ENCODE_BIN)
"""


"""
a = np.fromfile("data/tiny_stories/ts_valid1.bin", dtype=np.uint16)
b = np.fromfile("data/tiny_stories/ts_valid2.bin", dtype=np.uint16)
"""
