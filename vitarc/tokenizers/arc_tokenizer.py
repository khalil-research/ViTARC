#!/usr/bin/env python3
"""
arc_tokenizer.py

Utility to get or build a custom Byte-Pair Encoding (BPE) tokenizer
for the ARC project, with special ARC tokens. If the tokenizer 
already exists, it loads it rather than building anew.

Usage example:
----------------------------------------------------------------
from vitarc.tokenizers.arc_tokenizer import get_or_build_arc_tokenizer, demo_arc_tokenizer_test

tokenizer = get_or_build_arc_tokenizer()
demo_arc_tokenizer_test()
"""

import os
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

def get_or_build_arc_tokenizer(save_dir: str = "arc_tokenizer_v1") -> PreTrainedTokenizerFast:
    """
    Returns a HuggingFace-compatible ARC tokenizer.
    Always stores/loads the tokenizer in a subdirectory under
    the folder containing arc_tokenizer.py, e.g.:
      vitarc/tokenizers/arc_tokenizer_v1/

    :param save_dir: Name of the tokenizer subfolder (relative to arc_tokenizer.py).
    :return: A PreTrainedTokenizerFast instance.
    """
    # Build an absolute path to a subdir named `save_dir` under this file's directory
    current_dir = os.path.dirname(__file__)
    absolute_save_path = os.path.join(current_dir, save_dir)

    # If tokenizer directory exists, load it
    if os.path.isdir(absolute_save_path):
        print(f"[INFO] Loading existing tokenizer from: {absolute_save_path}")
        hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(absolute_save_path)
        return hf_tokenizer

    print(f"[INFO] No existing tokenizer found at '{absolute_save_path}'. Building a new one...")

    # Initialize a tokenizer with the Byte-Pair Encoding (BPE) model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer and PreTokenizer setup
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define custom ARC tokens
    arc_tokens = [
        "<arc_0>", "<arc_1>", "<arc_2>", "<arc_3>", "<arc_4>",
        "<arc_5>", "<arc_6>", "<arc_7>", "<arc_8>", "<arc_9>",
        "<arc_endxgrid>", "<arc_endygrid>", "<arc_endxygrid>",
        "<arc_pad>", "<arc_nl>"
    ]

    # Define special tokens
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "cls_token": "<cls>",
        "sep_token": "<sep>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    }

    # Convert the special tokens dict to a list
    special_tokens_list = list(special_tokens.values())

    # Add special and ARC tokens
    tokenizer.add_special_tokens(special_tokens_list)
    tokenizer.add_tokens(arc_tokens)

    # BPE trainer
    trainer = trainers.BpeTrainer(special_tokens=special_tokens_list)

    # Train on a trivial iterator to finalize config
    # (In real usage, pass actual text data from a file or dataset)
    tokenizer.train_from_iterator([""], trainer=trainer)

    # Wrap with HF-compatible tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=special_tokens["bos_token"],
        eos_token=special_tokens["eos_token"],
        unk_token=special_tokens["unk_token"],
        cls_token=special_tokens["cls_token"],
        sep_token=special_tokens["sep_token"],
        pad_token=special_tokens["pad_token"],
        mask_token=special_tokens["mask_token"],
        padding_side="right",
        truncation_side="right"
    )

    # Ensure the save directory exists
    os.makedirs(absolute_save_path, exist_ok=True)

    # Save the raw Tokenizer JSON
    raw_json_path = os.path.join(absolute_save_path, "arc_tokenizer.json")
    tokenizer.save(raw_json_path)

    # Save the HF-compatible tokenizer
    hf_tokenizer.save_pretrained(absolute_save_path)

    print(f"[INFO] New tokenizer built and saved to: {absolute_save_path}")
    print(f"[INFO] Vocabulary size: {len(hf_tokenizer)}")

    return hf_tokenizer


def demo_arc_tokenizer_test(test_input: str = "<arc_0><arc_1><pad><sep><arc_pad><arc_nl>"):
    """Simple demo function to encode/decode a sample input string."""
    hf_tokenizer = get_or_build_arc_tokenizer()  # Default: arc_tokenizer_v1 in this file's directory
    encoded_ids = hf_tokenizer.encode(test_input)
    tokens = hf_tokenizer.convert_ids_to_tokens(encoded_ids)
    decoded_text = hf_tokenizer.decode(encoded_ids)

    print("\n[DEMO] Input:", test_input)
    print("[DEMO] Encoded IDs:", encoded_ids)
    print("[DEMO] Encoded tokens:", tokens)
    print("[DEMO] Decoded:", decoded_text)
