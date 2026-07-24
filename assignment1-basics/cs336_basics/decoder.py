import argparse
import os
import pickle
import time

import numpy as np
import torch

from .bpe import BPETokenizer, SPLIT_TOKEN
from .model import TransformerLM
from .serialization import load_checkpoint




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
        vocab_size: int = 10000,  # TinyStories
        context_length: int = 256,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        d_ff: int = 1344,
        rope_theta: float = 10000
    ):

        self.special_tokens |= [SPLIT_TOKEN]

        self.tokenizer = BPETokenizer(
            vocab_path,
            merges_path,
            special_tokens,
        )
        vocab_size = len(self.tokenizer.vocab)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_length = context_length

        self.model = TransformerLM(
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            1e-5,
            torch.device(self.device),
            torch.float32,
        )

        load_checkpoint(checkpoint_path, self.model, self.optimizer)
    

    def decode(self, tokens: list[int], max_tokens: int, temp: float | None = None, p: float | None = None) -> list[int]:
        


    
