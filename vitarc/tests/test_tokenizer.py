# tests/test_tokenizer.py
import pytest
from vitarc.tokenizers.arc_tokenizer import get_or_build_arc_tokenizer

def test_arc_tokenizer_vocab_size():
    """
    Builds/loads the tokenizer and asserts the vocabulary size is 22.
    """
    hf_tokenizer = get_or_build_arc_tokenizer("arc_tokenizer_v1")
    vocab_size = len(hf_tokenizer)
    print(f"[TEST] Vocab size: {vocab_size}")
    assert vocab_size == 22, f"Expected 22 tokens, but got {vocab_size}"

def test_arc_tokenizer_encode_decode():
    """
    Simple encode/decode check with a sample input string.
    """
    hf_tokenizer = get_or_build_arc_tokenizer("arc_tokenizer_v1")
    test_input = "<arc_0><arc_1><pad><sep><arc_pad><arc_nl>"
    encoded_ids = hf_tokenizer.encode(test_input)
    tokens = hf_tokenizer.convert_ids_to_tokens(encoded_ids)
    decoded = hf_tokenizer.decode(encoded_ids)

    print(f"\n[TEST] Input: {test_input}")
    print(f"[TEST] Encoded IDs: {encoded_ids}")
    print(f"[TEST] Encoded tokens: {tokens}")
    print(f"[TEST] Decoded: {decoded}")

    # Basic sanity check
    assert isinstance(tokens, list)
    assert decoded is not None
