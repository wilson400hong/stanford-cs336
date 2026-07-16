# TODO: incomplete

"""
Input
input_path: str  Path to a text file with BPE tokenizer training data.
vocab_size: int  A positive integer that defines the maximum final vocabulary size (including
the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat
them as hard boundaries that prevent merges across their spans, but do not include them when
computing merge statistics.
Your BPE training function should return the resulting vocabulary and merges:
Output
vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the
vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list
item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.
"""


class BPETokenizer:
    def __init__(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass
