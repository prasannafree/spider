"""
Comprehensive test suite for the SpiderTokenizer.

Validates that the trained tokenizer works correctly for all downstream tasks:
roundtrip encoding, special token handling, SQL/NL tokenization, batch ops,
model-ready formatting, and edge cases.

Usage:
    python -m tokenizer.test_tokenizer
    python -m tokenizer.test_tokenizer --tokenizer_path tokenizer/trained_tokenizer.json
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.tokenizer import SpiderTokenizer


DEFAULT_TOKENIZER_PATH = Path("tokenizer/trained_tokenizer.json")


class TokenizerTestSuite:
    """Test suite for SpiderTokenizer validation."""

    def __init__(self, tokenizer_path: Path):
        self.tokenizer_path = tokenizer_path
        self.tok: SpiderTokenizer | None = None
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def _assert(self, condition: bool, test_name: str, detail: str = ""):
        """Record test result."""
        if condition:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            msg = f"  ✗ {test_name}"
            if detail:
                msg += f" — {detail}"
            print(msg)
            self.errors.append(f"{test_name}: {detail}")

    def run_all(self) -> bool:
        """Run all tests and return True if all passed."""
        print("=" * 60)
        print("SpiderTokenizer Test Suite")
        print("=" * 60)

        self.test_load()
        if self.tok is None:
            print("\n✗ Cannot proceed — tokenizer failed to load.")
            return False

        self.test_special_token_ids()
        self.test_pad_token_is_zero()
        self.test_vocab_size()
        self.test_roundtrip_nl()
        self.test_roundtrip_sql()
        self.test_sql_keywords()
        self.test_sql_operators()
        self.test_nested_sql()
        self.test_nl_questions()
        self.test_batch_encode()
        self.test_batch_decode()
        self.test_encode_for_model_with_target()
        self.test_encode_for_model_inference()
        self.test_skip_special_tokens()
        self.test_empty_string()
        self.test_token_id_conversion()
        self.test_repr()
        self.test_len()
        self.test_save_and_reload()
        self.test_case_preservation()

        print("\n" + "=" * 60)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 60)

        if self.errors:
            print("\nFailures:")
            for err in self.errors:
                print(f"  • {err}")

        return self.failed == 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def test_load(self):
        """Test that the tokenizer loads successfully."""
        print("\n[Loading]")
        try:
            self.tok = SpiderTokenizer(self.tokenizer_path)
            self._assert(True, "Tokenizer loads from file")
        except Exception as e:
            self._assert(False, "Tokenizer loads from file", str(e))

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def test_special_token_ids(self):
        """Test that all special tokens have valid IDs."""
        print("\n[Special Tokens]")
        self._assert(
            self.tok.pad_token_id is not None,
            "[PAD] has an ID",
            f"got {self.tok.pad_token_id}",
        )
        self._assert(
            self.tok.unk_token_id is not None,
            "[UNK] has an ID",
            f"got {self.tok.unk_token_id}",
        )
        self._assert(
            self.tok.bos_token_id is not None,
            "[BOS] has an ID",
            f"got {self.tok.bos_token_id}",
        )
        self._assert(
            self.tok.eos_token_id is not None,
            "[EOS] has an ID",
            f"got {self.tok.eos_token_id}",
        )
        self._assert(
            self.tok.sep_token_id is not None,
            "[SEP] has an ID",
            f"got {self.tok.sep_token_id}",
        )

    def test_pad_token_is_zero(self):
        """Test that [PAD] is at ID 0 for efficient masking."""
        print("\n[Pad Token Position]")
        self._assert(
            self.tok.pad_token_id == 0,
            "[PAD] is at ID 0",
            f"got ID {self.tok.pad_token_id}",
        )

    # ------------------------------------------------------------------
    # Vocab size
    # ------------------------------------------------------------------

    def test_vocab_size(self):
        """Test that vocab size is within expected range."""
        print("\n[Vocabulary]")
        vs = self.tok.vocab_size
        self._assert(
            1000 <= vs <= 10000,
            f"Vocab size is reasonable ({vs})",
            f"expected 1000-10000, got {vs}",
        )

    # ------------------------------------------------------------------
    # Roundtrip tests
    # ------------------------------------------------------------------

    def test_roundtrip_nl(self):
        """Test encode→decode roundtrip for natural language."""
        print("\n[Roundtrip - Natural Language]")
        texts = [
            "How many departments are there?",
            "Find the names of all students older than 20.",
            "What is the average salary of employees in each department?",
        ]
        for text in texts:
            ids = self.tok.encode(text)
            decoded = self.tok.decode(ids)
            # BPE roundtrip should preserve content (whitespace may differ slightly)
            self._assert(
                decoded.strip().lower() == text.strip().lower(),
                f"Roundtrip NL: '{text[:40]}...'",
                f"got '{decoded[:50]}'",
            )

    def test_roundtrip_sql(self):
        """Test encode→decode roundtrip for SQL queries."""
        print("\n[Roundtrip - SQL]")
        queries = [
            "SELECT count(*) FROM department",
            "SELECT name FROM student WHERE age > 20",
            "SELECT T1.name FROM student AS T1 JOIN enrollment AS T2 ON T1.id = T2.student_id",
        ]
        for query in queries:
            ids = self.tok.encode(query)
            decoded = self.tok.decode(ids)
            self._assert(
                decoded.strip().lower() == query.strip().lower(),
                f"Roundtrip SQL: '{query[:40]}...'",
                f"got '{decoded[:50]}'",
            )

    # ------------------------------------------------------------------
    # SQL-specific tests
    # ------------------------------------------------------------------

    def test_sql_keywords(self):
        """Test that common SQL keywords tokenize correctly and roundtrip."""
        print("\n[SQL Keywords]")
        keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER", "HAVING"]
        for kw in keywords:
            ids = self.tok.encode(kw)
            decoded = self.tok.decode(ids).strip()
            # Hard check: keyword must roundtrip correctly
            self._assert(
                decoded == kw,
                f"'{kw}' roundtrips correctly ({len(ids)} token{'s' if len(ids) != 1 else ''})",
                f"decoded as '{decoded}'",
            )
            # Informational: report if a keyword takes unusually many tokens
            if len(ids) > 3:
                print(f"    ⚠ Note: '{kw}' uses {len(ids)} tokens (expected ≤3 for efficiency)")

    def test_sql_operators(self):
        """Test that SQL operators are handled."""
        print("\n[SQL Operators]")
        operators = ["=", ">", "<", ">=", "<=", "!=", "*", "(", ")"]
        for op in operators:
            ids = self.tok.encode(op)
            self._assert(
                len(ids) >= 1,
                f"Operator '{op}' produces tokens",
                f"got {len(ids)} tokens",
            )

    def test_nested_sql(self):
        """Test tokenization of a complex nested SQL query."""
        print("\n[Nested SQL]")
        query = (
            "SELECT name FROM student WHERE id IN "
            "(SELECT student_id FROM enrollment WHERE course_id = 101)"
        )
        ids = self.tok.encode(query)
        decoded = self.tok.decode(ids)
        self._assert(
            len(ids) > 5,
            "Nested SQL produces multiple tokens",
            f"got {len(ids)} tokens",
        )
        self._assert(
            "SELECT" in decoded or "select" in decoded.lower(),
            "Decoded nested SQL contains SELECT",
            f"decoded: '{decoded[:60]}'",
        )

    # ------------------------------------------------------------------
    # NL-specific tests
    # ------------------------------------------------------------------

    def test_nl_questions(self):
        """Test that natural language questions tokenize reasonably."""
        print("\n[NL Questions]")
        questions = [
            "How many heads of the departments are older than 56?",
            "List the name and building of all departments sorted by budget.",
        ]
        for q in questions:
            ids = self.tok.encode(q)
            self._assert(
                len(ids) >= 3,
                f"NL question tokenizes: '{q[:40]}...'",
                f"got {len(ids)} tokens",
            )

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def test_batch_encode(self):
        """Test batch encoding produces correct results."""
        print("\n[Batch Encode]")
        texts = [
            "How many students are there?",
            "SELECT count(*) FROM student",
        ]
        batch_result = self.tok.encode_batch(texts)
        single_results = [self.tok.encode(t) for t in texts]

        self._assert(
            len(batch_result) == len(texts),
            "Batch encode returns correct count",
            f"expected {len(texts)}, got {len(batch_result)}",
        )
        self._assert(
            batch_result == single_results,
            "Batch encode matches individual encode",
        )

    def test_batch_decode(self):
        """Test batch decoding produces correct results."""
        print("\n[Batch Decode]")
        texts = ["Hello world", "SELECT * FROM test"]
        ids_list = self.tok.encode_batch(texts)
        decoded = self.tok.decode_batch(ids_list)

        self._assert(
            len(decoded) == len(texts),
            "Batch decode returns correct count",
        )
        for orig, dec in zip(texts, decoded):
            self._assert(
                dec.strip().lower() == orig.strip().lower(),
                f"Batch decode roundtrip: '{orig}'",
                f"got '{dec}'",
            )

    # ------------------------------------------------------------------
    # Model-ready encoding
    # ------------------------------------------------------------------

    def test_encode_for_model_with_target(self):
        """Test encode_for_model with both source and target."""
        print("\n[Model Encoding - Training]")
        result = self.tok.encode_for_model(
            source="How many students?",
            target="SELECT count(*) FROM student",
        )

        # Check all keys present
        self._assert(
            "encoder_input" in result,
            "Result has 'encoder_input'",
        )
        self._assert(
            "decoder_input" in result,
            "Result has 'decoder_input'",
        )
        self._assert(
            "decoder_target" in result,
            "Result has 'decoder_target'",
        )

        # Check BOS/EOS framing on encoder
        enc_input = result["encoder_input"]
        self._assert(
            enc_input[0] == self.tok.bos_token_id,
            "Encoder input starts with [BOS]",
            f"first token ID: {enc_input[0]}",
        )
        self._assert(
            enc_input[-1] == self.tok.eos_token_id,
            "Encoder input ends with [EOS]",
            f"last token ID: {enc_input[-1]}",
        )

        # Check decoder input starts with BOS
        dec_input = result["decoder_input"]
        self._assert(
            dec_input[0] == self.tok.bos_token_id,
            "Decoder input starts with [BOS]",
            f"first token ID: {dec_input[0]}",
        )

        # Check decoder target ends with EOS
        dec_target = result["decoder_target"]
        self._assert(
            dec_target[-1] == self.tok.eos_token_id,
            "Decoder target ends with [EOS]",
            f"last token ID: {dec_target[-1]}",
        )

        # Check decoder_input and decoder_target have same content shifted
        self._assert(
            dec_input[1:] == dec_target[:-1],
            "Decoder input/target are properly shifted",
            f"input[1:]={dec_input[1:][:5]}... target[:-1]={dec_target[:-1][:5]}...",
        )

    def test_encode_for_model_inference(self):
        """Test encode_for_model with source only (inference mode)."""
        print("\n[Model Encoding - Inference]")
        result = self.tok.encode_for_model(
            source="How many students?",
            target=None,
        )

        self._assert(
            "encoder_input" in result,
            "Inference result has 'encoder_input'",
        )
        self._assert(
            "decoder_input" not in result,
            "Inference result has no 'decoder_input'",
        )
        self._assert(
            "decoder_target" not in result,
            "Inference result has no 'decoder_target'",
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_skip_special_tokens(self):
        """Test that skip_special_tokens works in decode."""
        print("\n[Skip Special Tokens]")
        ids = [self.tok.bos_token_id] + self.tok.encode("hello") + [self.tok.eos_token_id]
        with_special = self.tok.decode(ids, skip_special_tokens=False)
        without_special = self.tok.decode(ids, skip_special_tokens=True)

        self._assert(
            "[BOS]" in with_special or "[EOS]" in with_special,
            "Decode with special tokens includes them",
            f"got '{with_special}'",
        )
        self._assert(
            "[BOS]" not in without_special and "[EOS]" not in without_special,
            "Decode skip_special_tokens removes them",
            f"got '{without_special}'",
        )

    def test_empty_string(self):
        """Test encoding an empty string doesn't crash."""
        print("\n[Edge Cases]")
        try:
            ids = self.tok.encode("")
            decoded = self.tok.decode(ids)
            self._assert(True, "Empty string encodes without error")
        except Exception as e:
            self._assert(False, "Empty string encodes without error", str(e))

    def test_token_id_conversion(self):
        """Test token_to_id and id_to_token."""
        print("\n[Token ↔ ID Conversion]")
        # Known token
        pad_id = self.tok.token_to_id("[PAD]")
        pad_token = self.tok.id_to_token(0)
        self._assert(
            pad_id == 0,
            "token_to_id('[PAD]') == 0",
            f"got {pad_id}",
        )
        self._assert(
            pad_token == "[PAD]",
            "id_to_token(0) == '[PAD]'",
            f"got {pad_token}",
        )

    def test_repr(self):
        """Test __repr__ produces readable output."""
        print("\n[Repr]")
        r = repr(self.tok)
        self._assert(
            "SpiderTokenizer" in r and "vocab_size" in r,
            f"repr is informative: {r}",
        )

    def test_len(self):
        """Test __len__ returns vocab_size."""
        print("\n[Len]")
        self._assert(
            len(self.tok) == self.tok.vocab_size,
            f"len(tok) == vocab_size ({len(self.tok)})",
        )

    def test_save_and_reload(self):
        """Test save/load roundtrip."""
        print("\n[Save & Reload]")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tokenizer.json"
            self.tok.save(save_path)

            reloaded = SpiderTokenizer.load(save_path)
            text = "SELECT name FROM student"
            orig_ids = self.tok.encode(text)
            reload_ids = reloaded.encode(text)

            self._assert(
                orig_ids == reload_ids,
                "Reloaded tokenizer produces same encoding",
            )

    def test_case_preservation(self):
        """Test that case is preserved (no lowercasing)."""
        print("\n[Case Preservation]")
        text = "SELECT Name FROM Student"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self._assert(
            "SELECT" in decoded and "Name" in decoded,
            "Case is preserved in encode→decode",
            f"got '{decoded}'",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Test the trained SpiderTokenizer."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Path to trained tokenizer (default: {DEFAULT_TOKENIZER_PATH})",
    )
    args = parser.parse_args()

    suite = TokenizerTestSuite(args.tokenizer_path)
    success = suite.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
