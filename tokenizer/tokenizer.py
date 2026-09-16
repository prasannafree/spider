"""
SpiderTokenizer — the main tokenizer interface for the encoder-decoder transformer.

This class wraps a trained HuggingFace BPE tokenizer and provides a clean,
consistent API for all downstream tasks:
  - Data preprocessing (dataset creation)
  - Model embedding layers (vocab_size, pad_token_id)
  - Training loops (encode_for_model)
  - Inference (encode, decode)

Usage:
    from tokenizer import SpiderTokenizer

    tok = SpiderTokenizer("tokenizer/trained_tokenizer.json")
    ids = tok.encode("How many departments are there?")
    text = tok.decode(ids)

    # For model training — adds [BOS]/[EOS] framing
    src_ids, tgt_ids = tok.encode_for_model(
        source="How many departments are there?",
        target="SELECT count(*) FROM department"
    )
"""

from pathlib import Path
from tokenizers import Tokenizer


# Special token definitions — shared across training and inference
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]

# Expected IDs (set during training, verified on load)
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"


class SpiderTokenizer:
    """
    BPE tokenizer for the Spider text-to-SQL encoder-decoder transformer.

    Wraps a HuggingFace tokenizers.Tokenizer with a domain-specific API
    that handles special token management, model-ready encoding, and
    batch operations.

    Attributes:
        vocab_size: Total vocabulary size including special tokens.
        pad_token_id: Token ID for [PAD] (always 0).
        unk_token_id: Token ID for [UNK].
        bos_token_id: Token ID for [BOS].
        eos_token_id: Token ID for [EOS].
        sep_token_id: Token ID for [SEP].
    """

    def __init__(self, tokenizer_path: str | Path):
        """
        Load a trained tokenizer from a JSON file.

        Args:
            tokenizer_path: Path to the trained tokenizer JSON file
                            (produced by train_tokenizer.py).

        Raises:
            FileNotFoundError: If the tokenizer file does not exist.
            ValueError: If required special tokens are missing from the vocabulary.
        """
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Trained tokenizer not found at {tokenizer_path}. "
                "Run `python -m tokenizer.train_tokenizer` first."
            )

        self._tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer_path = tokenizer_path

        # Validate that all special tokens are present
        self._validate_special_tokens()

    def _validate_special_tokens(self) -> None:
        """Verify all required special tokens exist in the vocabulary."""
        for token in SPECIAL_TOKENS:
            token_id = self._tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(
                    f"Special token '{token}' not found in tokenizer vocabulary. "
                    "The tokenizer may be corrupted or was trained incorrectly."
                )

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encode text into a list of token IDs.

        This is a raw encoding — no special tokens ([BOS], [EOS]) are added.
        Use encode_for_model() for training/inference-ready encoding.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            ids: List of integer token IDs.
            skip_special_tokens: If True, special tokens are omitted from output.

        Returns:
            Decoded text string.
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        Encode a batch of texts into lists of token IDs.

        Args:
            texts: List of input text strings.

        Returns:
            List of lists of integer token IDs.
        """
        encodings = self._tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def decode_batch(
        self, ids_list: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        """
        Decode a batch of token ID lists back into texts.

        Args:
            ids_list: List of lists of integer token IDs.
            skip_special_tokens: If True, special tokens are omitted from output.

        Returns:
            List of decoded text strings.
        """
        return self._tokenizer.decode_batch(
            ids_list, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Model-ready encoding (adds special tokens)
    # ------------------------------------------------------------------

    def encode_for_model(
        self, source: str, target: str | None = None) -> dict[str, list[int]]:
        """
        Encode source (and optionally target) with special token framing
        for the encoder-decoder transformer.

        Encoding format:
            encoder_input:  [BOS] <source_tokens> [EOS]
            decoder_input:  [BOS] <target_tokens>          (teacher forcing input)
            decoder_target: <target_tokens> [EOS]          (shifted labels)

        During inference, pass target=None to get only encoder_input.

        Args:
            source: Natural language question (encoder input).
            target: SQL query (decoder target). None during inference.

        Returns:
            Dict with keys:
                - "encoder_input": list[int] — [BOS] + source + [EOS]
                - "decoder_input": list[int] — [BOS] + target (if target given)
                - "decoder_target": list[int] — target + [EOS] (if target given)
        """
        source_ids = self.encode(source)
        result = {
            "encoder_input": [self.bos_token_id] + source_ids + [self.eos_token_id],
        }

        if target is not None:
            target_ids = self.encode(target)
            result["decoder_input"] = [self.bos_token_id] + target_ids
            result["decoder_target"] = target_ids + [self.eos_token_id]

        return result

    # ------------------------------------------------------------------
    # Token ↔ ID conversion utilities
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> int | None:
        """Convert a token string to its ID, or None if not in vocab."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        """Convert a token ID to its string, or None if out of range."""
        return self._tokenizer.id_to_token(id)

    # ------------------------------------------------------------------
    # Properties for downstream model configuration
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """Token ID for [PAD] — used for padding sequences in batches."""
        return self._tokenizer.token_to_id(PAD_TOKEN)

    @property
    def unk_token_id(self) -> int:
        """Token ID for [UNK] — used when input cannot be represented with the learned vocabulary."""
        return self._tokenizer.token_to_id(UNK_TOKEN)

    @property
    def bos_token_id(self) -> int:
        """Token ID for [BOS] — beginning of sequence marker."""
        return self._tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_token_id(self) -> int:
        """Token ID for [EOS] — end of sequence marker."""
        return self._tokenizer.token_to_id(EOS_TOKEN)

    @property
    def sep_token_id(self) -> int:
        """Token ID for [SEP] — separator token."""
        return self._tokenizer.token_to_id(SEP_TOKEN)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save the tokenizer to a JSON file.

        Args:
            path: Output path. Defaults to the original load path.

        Returns:
            Path where the tokenizer was saved.
        """
        save_path = Path(path) if path else self._tokenizer_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(save_path))
        return save_path

    @classmethod
    def load(cls, tokenizer_path: str | Path) -> "SpiderTokenizer":
        """
        Load a trained tokenizer from a JSON file.

        This is an alias for the constructor, provided for API clarity.

        Args:
            tokenizer_path: Path to the trained tokenizer JSON file.

        Returns:
            Initialized SpiderTokenizer instance.
        """
        return cls(tokenizer_path)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpiderTokenizer(vocab_size={self.vocab_size}, "
            f"path='{self._tokenizer_path}')"
        )

    def __len__(self) -> int:
        """Returns vocab_size for convenience."""
        return self.vocab_size
