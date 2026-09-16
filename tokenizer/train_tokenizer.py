"""
BPE tokenizer training script for the Spider text-to-SQL transformer.

Trains a Byte-Pair Encoding (BPE) tokenizer on the prepared corpus using
the HuggingFace tokenizers library. The tokenizer learns subword units
from both natural language and SQL text, preserving case.

Usage:
    python -m tokenizer.train_tokenizer
    python -m tokenizer.train_tokenizer --vocab_size 8000 --corpus data/tokenizer_corpus.txt
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace as MetaspacePretokenizer
from tokenizers.normalizers import NFC
from tokenizers.decoders import Metaspace as MetaspaceDecoder


# ---- Configuration defaults ----
DEFAULT_CORPUS_PATH = Path("data/tokenizer_corpus.txt")
DEFAULT_OUTPUT_PATH = Path("tokenizer/trained_tokenizer.json")
DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MIN_FREQUENCY = 2

# Special tokens — order matters: IDs are assigned in this order (0, 1, 2, 3, 4)
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]


def train_tokenizer(
    corpus_path: Path | None = None,
    output_path: Path | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> Path:
    """
    Train a BPE tokenizer on the prepared Spider corpus.

    Architecture decisions:
    - BPE (Byte-Pair Encoding): Best balance of vocabulary efficiency and
      handling of rare words. Widely used in transformers (GPT, RoBERTa).
    - No lowercasing: Preserves case for SQL keywords and identifiers.
    - NFC normalization: Consistent Unicode representation without changing case.
    - Metaspace pre-tokenizer: Splits on whitespace and marks word boundaries
      with a ▁ prefix, which allows the decoder to correctly restore spaces.
      Configuration is explicit (replacement='▁', prepend_scheme='always')
      for reproducibility across tokenizers library versions.
    - UNK token: Set as the default unknown token in the BPE model so
      input sequences that cannot be represented with the learned vocabulary
      are mapped to [UNK] instead of causing errors.

    Args:
        corpus_path: Path to the training corpus file (one sentence per line).
        output_path: Path to save the trained tokenizer JSON.
        vocab_size: Target vocabulary size (default: 8000).
        min_frequency: Minimum frequency for a pair to be merged (default: 2).

    Returns:
        Path to the saved tokenizer JSON file.
    """
    corpus_path = corpus_path or DEFAULT_CORPUS_PATH
    output_path = output_path or DEFAULT_OUTPUT_PATH

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found at {corpus_path}. "
            "Run `python -m tokenizer.prepare_data` first."
        )

    # --- 1. Initialize BPE tokenizer ---
    print(f"Initializing BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # --- 2. Configure normalizer ---
    # NFC: canonical Unicode decomposition followed by composition.
    # Does NOT change case — preserves SELECT vs select distinction.
    tokenizer.normalizer = NFC()

    # --- 3. Configure pre-tokenizer ---
    # Metaspace splits on whitespace and prepends ▁ to mark word starts.
    # This lets the decoder correctly restore spaces between words.
    # Configuration is explicit for reproducibility across library versions.
    tokenizer.pre_tokenizer = MetaspacePretokenizer(
        replacement="▁",
        prepend_scheme="always",
    )

    # --- 4. Configure decoder ---
    # Metaspace decoder restores spaces by interpreting ▁ markers.
    tokenizer.decoder = MetaspaceDecoder(
        replacement="▁",
        prepend_scheme="always",
    )

    # --- 5. Configure trainer ---
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # --- 6. Train ---
    print(f"Training on corpus: {corpus_path}")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    # --- 7. Verify special token IDs ---
    print("\nSpecial token assignments:")
    for i, token in enumerate(SPECIAL_TOKENS):
        assigned_id = tokenizer.token_to_id(token)
        print(f"  {token} -> ID {assigned_id}")
        assert assigned_id == i, (
            f"Expected {token} at ID {i}, got {assigned_id}. "
            "Special tokens were not assigned in the expected order."
        )

    # --- 8. Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    final_vocab_size = tokenizer.get_vocab_size()
    print(f"\nTokenizer trained successfully!")
    print(f"  Vocabulary size: {final_vocab_size}")
    print(f"  Saved to: {output_path}")

    # --- 9. Quick sanity check ---
    print("\n--- Quick sanity check ---")
    test_inputs = [
        "How many departments are there?",
        "SELECT count(*) FROM department",
        "Find all students whose age is greater than 20",
        "SELECT name FROM student WHERE age > 20 ORDER BY name ASC",
    ]
    for text in test_inputs:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Input:   {text}")
        print(f"  Tokens:  {encoded.tokens}")
        print(f"  IDs:     {encoded.ids}")
        print(f"  Decoded: {decoded}")
        print()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on the Spider corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Path to corpus file (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output tokenizer path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help=f"Minimum token frequency (default: {DEFAULT_MIN_FREQUENCY})",
    )
    args = parser.parse_args()

    train_tokenizer(
        corpus_path=args.corpus,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
