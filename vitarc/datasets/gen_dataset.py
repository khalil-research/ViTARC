#!/usr/bin/env python3
"""
gen_dataset.py

Generates ARC-style datasets using a local copy of the `re_arc` library for grid
transformations, applies custom tokenizer encoding, and splits into train/val/test.

Example usage:
  python gen_dataset.py --task_idx 0 --num_examples 10 --test_size 1 --seed 1230 --output_dir "./out_10_trial"

  # This will create a dataset for task 0, with 10 total examples, 1 test example,
  # and 1 validation example, then save it under ./out_10_trial.

Note:
  - `re_arc` code is vendored in: vitarc/external/re_arc/
  - `arc_tokenizer.py` in vitarc/tokenizers/ is used to load or build an ARC tokenizer.
"""

import os
import sys
import time
import random
import argparse
import json
import tqdm
import numpy as np
import torch

# Vendored re_arc imports
from vitarc.external.re_arc.main import get_generators, get_verifiers
from vitarc.datasets.obj_idx_utils import generate_input_type_ids_multi

# If you use other re_arc modules/functions, import them similarly:
from vitarc.external.re_arc.utils import *
from vitarc.external.re_arc.generators import *
from vitarc.external.re_arc.verifiers import *

# HF Datasets
from datasets import Dataset, DatasetDict

# HF Transformers
from transformers import AutoTokenizer, set_seed

# Tools for table printing
from prettytable import PrettyTable

# Local ARC tokenizer
from vitarc.tokenizers.arc_tokenizer import get_or_build_arc_tokenizer

# ------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------
MAX_INPUT_LENGTH = 1124
MAX_TARGET_LENGTH = 1124
TARGET_SHAPE = (33, 34)

def set_random_seed(seed: int):
    """
    Set random seeds for reproducibility (Python, NumPy, Torch).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# ------------------------------------------------------------------------------------
# Utility / Helper Functions
# ------------------------------------------------------------------------------------
def repad_2d_list(grid, pad_token='<arc_pad>', target_size=32):
    """
    Re-pads a 2D list of tokens to (target_size+1) rows, each row padded to target_size + 1 columns.
    Adds boundary tokens <arc_endxgrid>, <arc_endygrid>, <arc_endxygrid>.
    """
    row_len = len(grid[0])
    pad_len = target_size - row_len
    padded_grid = [
        row + ['<arc_endxgrid>'] + [pad_token]*pad_len + ['<arc_nl>']
        for row in grid
    ]

    # Add one extra row for <arc_endygrid>
    padding_row_y = (['<arc_endygrid>']*row_len + ['<arc_endxygrid>']
                     + ['<arc_endygrid>']*pad_len + ['<arc_nl>'])
    padding_row = ([pad_token]*row_len + ['<arc_endxgrid>']
                   + [pad_token]*pad_len + ['<arc_nl>'])
    padded_grid.append(padding_row_y)
    while len(padded_grid) < target_size + 1:
        padded_grid.append(padding_row)
    return padded_grid

def reshape_to_32x32(input_list):
    """
    Reshapes a 1D list of length 32*32 into a 32x32 2D list.
    Raises ValueError if length != 32*32.
    """
    if len(input_list) != 32 * 32:
        raise ValueError("Input list must have exactly 32*32 elements")
    return [input_list[i*32:(i+1)*32] for i in range(32)]

def unpad_2d_list(grid):
    """
    Placeholder if you had an 'unpad_2d_list' function from HPC code.
    Not fully defined here; define if needed or remove calls to it.
    """
    return grid

def print_table(grid):
    """
    Print a 2D grid as a PrettyTable (for debugging).
    """
    table = PrettyTable()
    for row in grid:
        table.add_row(row)
    print(table)

def reformat_arc_tokens(grid, pad_token="<arc_pad>", print_grids=False):
    """
    Takes a 2D grid of tokens (ints replaced by <arc_#>).
    Re-pads them to a standard size, then flattens into a single string.
    """
    padded_tokens_2d = repad_2d_list(grid, pad_token=pad_token)
    if print_grids:
        print("\n[INFO] 2D Tokens:")
        print_table(grid)
        print("\n[INFO] Re-padded 2D Tokens:")
        print_table(padded_tokens_2d)

    flattened_tokens = [token for row in padded_tokens_2d for token in row]
    joined_tokens = ''.join(flattened_tokens)
    return joined_tokens

def replace_digits_with_arc(grid):
    """
    Example: a 2D grid of ints 0..9 => <arc_0>.. <arc_9>
    """
    return [[f'<arc_{element}>' for element in row] for row in grid]

def pad_and_flatten(input_type_ids, target_shape, final_length):
    """
    2D -> target_shape (padded with zeros) -> flatten -> pad front+back with 0 => final_length.
    Used to ensure consistent input_type_ids / output_type_ids size.
    """
    current_shape = input_type_ids.shape
    padded_array = np.zeros(target_shape, dtype=int)
    padded_array[:current_shape[0], :current_shape[1]] = input_type_ids

    flattened_array = padded_array.flatten()

    # Add 0 at the beginning and the end
    padded_and_flattened = np.pad(flattened_array, (1, 1), 'constant', constant_values=0)

    if len(padded_and_flattened) != final_length:
        raise ValueError(
            f"Final length {len(padded_and_flattened)} != expected {final_length}."
        )
    return padded_and_flattened

def is_grid_extra(grid) -> bool:
    """
    Returns True if 'grid' is a valid grid of shape (<=30, <=30) of integers [0..9].
    """
    if not isinstance(grid, tuple):
        return False
    if len(grid) == 0 or len(grid) > 30:
        return False
    if not all(isinstance(r, tuple) for r in grid):
        return False
    if not all(0 < len(r) <= 30 for r in grid):
        return False
    # consistent column size
    if not len({len(r) for r in grid}) == 1:
        return False
    # all ints
    if not all(all(isinstance(x, int) for x in r) for r in grid):
        return False
    # 0..9
    if not all(all(0 <= x <= 9 for x in r) for r in grid):
        return False
    return True

# ------------------------------------------------------------------------------------
# Preprocessing Pipeline
# ------------------------------------------------------------------------------------
def preprocess_example(example, tokenizer, max_input_length=MAX_INPUT_LENGTH, max_target_length=MAX_TARGET_LENGTH):
    """
    Convert 2D grid input/output to 1D sequences of special tokens, tokenize them,
    and attach 'labels' for training.
    """
    input_grid = replace_digits_with_arc(example['input'])
    output_grid = replace_digits_with_arc(example['output'])

    # Reformat tokens into a single string, with <s> & </s> wrappers
    result_string_input = "<s>" + reformat_arc_tokens(input_grid) + "</s>"
    result_string_output = "<s>" + reformat_arc_tokens(output_grid) + "</s>"

    # Tokenize & pad/truncate
    model_inputs = tokenizer(result_string_input, max_length=max_input_length,
                             padding="max_length", truncation=True)

    labels = tokenizer(result_string_output, max_length=max_target_length,
                       padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    model_inputs["input_text"] = result_string_input
    model_inputs["output_text"] = result_string_output

    # Generate type IDs using vendored re_arc DSL
    i_type_ids = generate_input_type_ids_multi(np.array(example['input']), visualize=False)
    o_type_ids = generate_input_type_ids_multi(np.array(example['output']), visualize=False)

    # Pad & flatten to consistent size
    model_inputs["input_type_ids"] = pad_and_flatten(i_type_ids, TARGET_SHAPE, max_input_length)
    model_inputs["output_type_ids"] = pad_and_flatten(o_type_ids, TARGET_SHAPE, max_target_length)

    return model_inputs

# ------------------------------------------------------------------------------------
# Main Dataset Generation
# ------------------------------------------------------------------------------------
def generate_single_dataset_hf(
    task_idx: int = 0,
    seed: int = 42,
    n_examples: int = 1000,
    testsize: int = 100,
    diff_lb: float = 0.0,
    diff_ub: float = 1.0,
    tokenizer=None
):
    """
    Generates a dataset using vendored re_arc (task_idx). Returns a DatasetDict.

    :param task_idx: index of the generator key from re_arc
    :param seed: random seed for reproducibility
    :param n_examples: total examples to generate
    :param testsize: number of test + val samples
    :param diff_lb: difficulty lower bound
    :param diff_ub: difficulty upper bound
    :param tokenizer: a HF tokenizer (from get_or_build_arc_tokenizer)
    :return: (task_key, DatasetDict, stats)
    """

    set_random_seed(seed)
    random.seed(seed)

    generators_mapper = get_generators()
    verifiers_mapper = get_verifiers()
    keys = sorted(generators_mapper.keys())
    key = keys[task_idx]

    print(f"[INFO] Generating dataset for Task ID: {key}")
    generator = generators_mapper[key]
    verifier = verifiers_mapper[key]

    seen = set()
    examples = []

    stats = {
        'n_generations': 0,
        'n_verified': 0,
        'n_nondegenerate': 0
    }

    max_attempts = 40 * n_examples
    attempts = 0
    start = time.time()

    with tqdm.tqdm(total=n_examples) as pbar:
        while len(examples) < n_examples and attempts < max_attempts:
            attempts += 1
            example_data = None
            success = True

            try:
                example_data = generator(diff_lb, diff_ub)
                # Validate
                assert is_grid_extra(example_data['input'])
                assert is_grid_extra(example_data['output'])
                stats['n_generations'] += 1
            except:
                success = False

            if success:
                try:
                    # Verify that the output is correct
                    assert verifier(example_data['input']) == example_data['output']
                    stats['n_verified'] += 1
                except:
                    success = False

            if success:
                try:
                    # Ensure non-degenerate input != output
                    assert example_data['input'] != example_data['output']
                    stats['n_nondegenerate'] += 1
                except:
                    success = False

            if success:
                identifier = hash(example_data['input'])
                if identifier not in seen:
                    examples.append(example_data)
                    seen.add(identifier)
                    pbar.update(1)

            pbar.set_postfix({"attempts": attempts})

    end = time.time()
    stats['runtime'] = end - start

    print("[INFO] Stats:", stats)

    # Convert to HF Dataset
    dataset = Dataset.from_list(examples)

    # If the user hasn't provided a tokenizer, load the default from arc_tokenizer
    if tokenizer is None:
        tokenizer = get_or_build_arc_tokenizer()

    def _preprocess(example):
        return preprocess_example(example, tokenizer)

    # Preprocessing
    processed_dataset = dataset.map(_preprocess, batched=False)

    # Split train/val/test
    train_test_split = processed_dataset.train_test_split(test_size=testsize*2, shuffle=True, seed=seed)
    train_dataset = train_test_split['train']
    test_val_split = train_test_split['test'].train_test_split(test_size=testsize, shuffle=True, seed=seed)
    val_dataset = test_val_split['train']
    test_dataset = test_val_split['test']

    final_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    return key, final_datasets, stats

# ------------------------------------------------------------------------------------
# Command-Line Interface
# ------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate ARC dataset with vendored re_arc.")
    parser.add_argument("--task_idx", type=int, default=0, help="Task index from re_arc.")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of total examples to generate.")
    parser.add_argument("--test_size", type=int, default=1, help="Number of test+val examples each.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="./dataset_out", help="Where to save the final dataset.")
    args = parser.parse_args()

    task_idx = args.task_idx
    num_examples = args.num_examples
    test_size = args.test_size
    seed = args.seed
    output_dir = args.output_dir

    start_time = time.time()
    task_key, final_datasets, stats = generate_single_dataset_hf(
        task_idx=task_idx,
        seed=seed,
        n_examples=num_examples,
        testsize=test_size
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{task_key}_{num_examples}_examples")
    final_datasets.save_to_disk(save_path)
    print(f"[INFO] Saved dataset to {save_path}")

    end_time = time.time()
    runtime = end_time - start_time

    print(f"Total runtime: {runtime:.2f} seconds.")
    hrs, rem = divmod(runtime, 3600)
    mins, secs = divmod(rem, 60)
    print(f"[INFO] Elapsed: {int(hrs)}h {int(mins)}m {secs:.2f}s")
    print("[INFO] Final Stats:", stats)

if __name__ == "__main__":
    main()
