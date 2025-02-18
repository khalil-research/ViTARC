import pytest
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-GUI backend for tests

from vitarc.datasets.gen_dataset import generate_single_dataset_hf
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

TEMP_DIR = "vitarc/tests/test_gen_dataset_tmp"

def ensure_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)

##############################################################################
# 1) ARC short-label map for annotation
##############################################################################
ARC_LABEL_MAP = {
    f"<arc_{i}>": str(i) for i in range(10)      # <arc_0>.. <arc_9> => "0".."9"
}
ARC_LABEL_MAP["<arc_pad>"]       = "pad"
ARC_LABEL_MAP["<arc_nl>"]        = "nl"
ARC_LABEL_MAP["<arc_endxgrid>"]  = "eX"
ARC_LABEL_MAP["<arc_endygrid>"]  = "eY"
ARC_LABEL_MAP["<arc_endxygrid>"] = "eXY"

def get_short_label(token: str) -> str:
    """Return a short label (e.g. '5', 'pad', 'nl', etc.) or '?' if unknown."""
    return ARC_LABEL_MAP.get(token, "?")

##############################################################################
# 2) Token -> color index for background
##############################################################################
ARC_COLOR_INDEX = {
    f"<arc_{i}>": i for i in range(10)
}
ARC_COLOR_INDEX["<arc_pad>"]       = 10
ARC_COLOR_INDEX["<arc_nl>"]        = 11
ARC_COLOR_INDEX["<arc_endxgrid>"]  = 12
ARC_COLOR_INDEX["<arc_endygrid>"]  = 13
ARC_COLOR_INDEX["<arc_endxygrid>"] = 14

def token_to_color_index(token: str) -> int:
    """Return 0..14, else 19 if unknown."""
    return ARC_COLOR_INDEX.get(token, 19)

##############################################################################
# 3) A color map for up to 15 special indices, plus fallback for 19
##############################################################################
arc_colors = [
    # 0..9 => arc_0.. arc_9
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25", 
    # 10..14 => arc_pad, arc_nl, eX, eY, eXY
    "#DDDDDD", "#E6B0AA", "#B2BABB", "#D2B4DE", "#58D68D",
    # fallback for 19
    "#808B96", "#999999", "#C0C0C0", "#D0D0D0", "#E0E0E0"
]
ARC_COLOR_MAP = LinearSegmentedColormap.from_list("arc_colors", arc_colors)

##############################################################################
# 4) We define a bounding-box color list to cycle through for object IDs
##############################################################################
BOX_COLORS = [
    "white", "red", "blue", "yellow", "green",
    "purple", "magenta", "cyan", "orange", "lime",
    "pink", "gold", "brown", "navy", "olive",
]

def paint_text_with_boxes(text_str: str,
                          tokenizer,
                          png_name: str,
                          input_type_ids_2d: np.ndarray = None,
                          fig_title="ARC Heatmap"):
    """
    1) Tokenize text_str -> tokens.
    2) Remove <s> / </s> if present.
    3) Force 33x34=1122 shape (cut or pad).
    4) Build numeric grid for color, short_label grid for annotation.
    5) Plot a 12x10 heatmap.
    6) If input_type_ids_2d is given, draw bounding boxes for IDs !=0 with distinct color.
    7) Save to 'png_name' in TEMP_DIR.
    """

    ensure_temp_dir()
    tokens = tokenizer.tokenize(text_str)

    if tokens and tokens[0] == "<s>":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "</s>":
        tokens = tokens[:-1]

    # Force length = 1122
    length = len(tokens)
    if length >= 1122:
        tokens = tokens[:1122]
    else:
        tokens += ["<arc_pad>"] * (1122 - len(tokens))

    arr_2d = np.array(tokens, dtype=object).reshape(33, 34)

    # Build color + label grids
    color_grid = np.zeros((33, 34), dtype=int)
    label_grid = np.empty((33, 34), dtype=object)
    for i in range(33):
        for j in range(34):
            tk = arr_2d[i, j]
            color_grid[i, j] = token_to_color_index(tk)
            label_grid[i, j] = get_short_label(tk)

    # Plot
    plt.figure(figsize=(16, 15))
    plt.title(fig_title, fontsize=16)
    df = pd.DataFrame(color_grid)
    ax = sns.heatmap(
        df,
        annot=label_grid,
        fmt="s",
        linewidths=.5,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        cmap=ARC_COLOR_MAP,
        vmin=0, vmax=19
    )

    # bounding boxes
    if input_type_ids_2d is not None:
        object_ids = np.unique(input_type_ids_2d)
        object_ids = object_ids[object_ids != 0]  # skip background=0
        for idx, obj_id in enumerate(object_ids):
            coords = np.argwhere(input_type_ids_2d == obj_id)
            if len(coords) == 0:
                continue
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)
            box_color = BOX_COLORS[idx % len(BOX_COLORS)]
            rect = patches.Rectangle(
                (min_col, min_row),
                (max_col - min_col + 1),
                (max_row - min_row + 1),
                linewidth=2,
                edgecolor=box_color,
                facecolor="none"
            )
            ax.add_patch(rect)

    out_path = os.path.join(TEMP_DIR, png_name)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {png_name} => {out_path}")

##############################################################################
# rough_print => console table
##############################################################################
def rough_print(gen_output, tokenizer):
    tokens = tokenizer.tokenize(gen_output)
    print(f"[DEBUG] Total tokens: {len(tokens)}")

    lines = []
    current_line = []
    for token in tokens:
        if token == "<arc_nl>":
            current_line.append(token)
            lines.append(current_line)
            current_line = []
        else:
            current_line.append(token)

    if current_line:
        if current_line[-1] != "</s>":
            current_line.append("<arc_nl>")
        lines.append(current_line)

    from prettytable import PrettyTable
    tbl = PrettyTable()

    if lines:
        max_len = max(len(line) for line in lines)
        print(f"[INFO] rough_print => # lines: {len(lines)}, max line length: {max_len}")
        headers = [f"T{i+1}" for i in range(max_len)]
        tbl.field_names = headers

        for line in lines:
            padded = line + [""] * (max_len - len(line))
            tbl.add_row(padded)
    else:
        print("[WARN] No tokens found. Possibly empty string?")
        tbl.field_names = ["(no tokens)"]

    print(tbl)


@pytest.mark.parametrize("num_examples", [4])
def test_generate_and_print_task(num_examples):
    """
    1) Generate a small dataset.
    2) Print first train example, partial data if big.
    3) For input_text, output_text => produce input_text.png, output_text.png, each bounding box colored differently.
    4) rough_print them
    5) Show all columns & shapes, reprint partial data if big.
    """

    task_key, final_ds, stats = generate_single_dataset_hf(
        task_idx=0,
        seed=1230,
        n_examples=num_examples,
        testsize=1
    )

    print(f"\n=== Generated dataset for task '{task_key}' with {num_examples} examples ===")
    print("Stats:", stats)
    print(final_ds)

    train_split = final_ds["train"]
    if len(train_split) == 0:
        print("[ERROR] No train samples found.")
        return

    # Attempt to load local ARC tokenizer
    try:
        local_tokenizer = AutoTokenizer.from_pretrained("vitarc/tokenizers/arc_tokenizer_v1")
    except Exception as e:
        print(f"[WARN] Could not load arc_tokenizer_v1. Fallback => {e}")
        local_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    first_sample = train_split[0]
    print("\n=== First Train Sample ===")

    # Print partial data
    for col_name, col_val in first_sample.items():
        if isinstance(col_val, list):
            ln = len(col_val)
            if ln > 10:
                preview = col_val[:10]
                print(f"{col_name} = list of length {ln}, first 10 => {preview}")
            else:
                print(f"{col_name} = {col_val}")
        else:
            if col_name in ["input_text", "output_text"]:
                print(f"{col_name} = (str length {len(col_val)}) => {col_val[:80]}...")
                tks = local_tokenizer.tokenize(col_val)
                print(f"   => tokenized length {len(tks)}, first 10 => {tks[:10]}")
            else:
                print(f"{col_name} = {col_val}")

    # Build bounding box arrays if present
    input_box_2d = None
    if "input_type_ids" in first_sample and len(first_sample["input_type_ids"]) >= 1122:
        arr = first_sample["input_type_ids"]
        arr_2d = np.array(arr[1:-1], dtype=int).reshape(33,34)  # ignoring first/last
        input_box_2d = arr_2d

    output_box_2d = None
    if "output_type_ids" in first_sample and len(first_sample["output_type_ids"]) >= 1122:
        arr = first_sample["output_type_ids"]
        arr_2d = np.array(arr[1:-1], dtype=int).reshape(33,34)
        output_box_2d = arr_2d

    # Paint input_text with bounding boxes
    if "input_text" in first_sample:
        paint_text_with_boxes(
            text_str=first_sample["input_text"],
            tokenizer=local_tokenizer,
            png_name="input_text.png",
            input_type_ids_2d=input_box_2d,
            fig_title="Input Text Heatmap"
        )

    # Paint output_text with bounding boxes
    if "output_text" in first_sample:
        paint_text_with_boxes(
            text_str=first_sample["output_text"],
            tokenizer=local_tokenizer,
            png_name="output_text.png",
            input_type_ids_2d=output_box_2d,
            fig_title="Output Text Heatmap"
        )

    # rough_print
    if "input_text" in first_sample:
        print("\n--- rough_print of input_text ---")
        rough_print(first_sample["input_text"], local_tokenizer)

    if "output_text" in first_sample:
        print("\n--- rough_print of output_text ---")
        rough_print(first_sample["output_text"], local_tokenizer)

    # Summarize columns
    print("\n=== Feature Columns / Shapes ===")
    feats = final_ds["train"].features
    for col in feats:
        cval = first_sample[col]
        if isinstance(cval, list):
            shape_str = f"list of length {len(cval)}"
        else:
            shape_str = str(type(cval))
        print(f"Column '{col}' => {shape_str}")

    # Reprint partial data for big lists
    print("\n=== REPRINT partial data for big lists ===")
    for cn, cv in first_sample.items():
        if isinstance(cv, list) and len(cv) > 10:
            print(f"{cn} => first 10 => {cv[:10]}")

    print("\n=== End of test ===")
