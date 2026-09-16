"""
Tokenizer package for the encoder-decoder transformer.

Exports the main SpiderTokenizer class for use across all downstream tasks
including data preprocessing, model training, and inference.
"""

from tokenizer.tokenizer import SpiderTokenizer

__all__ = ["SpiderTokenizer"]
