import argparse
import os
import pickle
import time

import numpy as np
import torch

from .bpe import BPETokenizer, SPLIT_TOKEN
from .data import get_batch
from .model import TransformerLM
from .nn_utils import cross_entropy, gradient_clipping
from .optimizer import AdamW, lr_cosine_schedule
from .serialization import load_checkpoint, save_checkpoint


TS_VOCAB = "data/tiny_stories/ts_vocab.pickle"
TS_MERGES = "data/tiny_stories/ts_merges.pickle"
CKPT = "tmp/ts_ckpt_5000.pt"


class LLM:
    def __init__(
        self,
        *,
        # tokenizer
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
        # model
        checkpoint_path: str,
        context_length: int = 256,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        d_ff: int = 1344,
        rope_theta: float = 10000,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Init tokenizer...")
        special_tokens = [SPLIT_TOKEN] if special_tokens is None else special_tokens
        self.tokenizer = BPETokenizer.from_files(
            vocab_path, merges_path, special_tokens
        )
        vocab_size = len(self.tokenizer.vocab)

        self.eot_token = self.tokenizer.eot_token
        print("Init model...")
        self.model = TransformerLM(
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            1e-5,
            torch.device(device),
            torch.float32,
        )
        print("Load checkpoint...")
        load_checkpoint(checkpoint_path, self.model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,  # something very big
        temp: float | None = None,
        top_p: float | None = None,
    ) -> str:
        prompt_tokens = self.tokenizer.encode(prompt)
        # print(prompt_tokens)
        gen_tokens = self.model.decode(
            prompt_tokens, self.eot_token, max_tokens, temp, top_p
        )
        # print(gen_tokens)
        return self.tokenizer.decode(gen_tokens)


"""
Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big
black one. One day, Lily’s mom asked her to help cook dinner. Lily was so excited! She loved to
help her mom. Lily’s mom made a big pot of soup for dinner. Lily was so happy and said, “Thank
you, Mommy! I love you.” She helped her mom pour the soup into a big bowl. After dinner, Lily’s
mom made some yummy soup. Lily loved it! She said, “Thank you, Mommy! This soup is so
yummy!” Her mom smiled and said, “I’m glad you like it, Lily.” They finished cooking and
continued to cook together. The end.
"""
