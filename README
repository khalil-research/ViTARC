# ViTARC: Tackling the Abstraction and Reasoning Corpus with Vision Transformers
**Paper:** [Tackling the Abstraction and Reasoning Corpus with Vision Transformers: the Importance of 2D Representation, Positions, and Objects](https://arxiv.org/abs/2410.06405)

This repository provides code to generate ARC-like datasets, build a custom T5-based model with 2D positional embeddings, and train end-to-end on those tasks. It is structured for quick experimentation with different ARC tasks, custom tokenizers, and a flexible pipeline using Hugging Face Transformers and PyTorch Lightning.

**License:** MIT.

---

## Requirements

- Python 3.10 (recommended: 3.10.12)
- Additional packages as listed in:
  - `requirements.txt` (strict pinned versions)
  - `setup_full.py` (if you need a more strongly pinned environment)

---

## Quick Start

1. **Clone the Repository & Create a Virtual Environment**

    ```bash
    git clone https://github.com/yourusername/ViTARC.git
    cd ViTARC
    python3.10 -m venv venv
    source venv/bin/activate
    ```

2. **Install the Package**

    ```bash
    pip install --upgrade pip
    pip install -e .
    ```

3. **Run the Training Script**

    ```bash
    python vitarc/training/train.py
    ```

    **Command-line arguments**:
    ```
    --task_idx (int, default=0)
    --max_input_length (int, default=1124)
    --max_target_length (int, default=1124)
    --batch_size (int, default=8)
    --epochs (int, default=1)
    --seed (int, default=1230)
    --ds_base_dir (str, default="./arc_x2y_datasets")
    --use_slurm_copy (bool, default=False)   # HPC copy to SLURM_TMPDIR
    ```

---

## Repository Layout

```
.
├── arc_x2y_datasets
├── requirements.txt
├── setup.py
├── setup_full.py
└── vitarc
    ├── datasets
    │   ├── gen_dataset.py
    │   └── obj_idx_utils.py
    ├── external
    │   └── re_arc
    │       ├── LICENSE_re_arc
    │       ├── README.md
    │       ├── dsl.py
    │       ├── generators.py
    │       ├── main.py
    │       ├── utils.py
    │       └── verifiers.py
    ├── models
    │   └── model.py
    ├── tests
    │   ├── test_gen_dataset.py
    │   └── test_tokenizer.py
    ├── tokenizers
    │   └── arc_tokenizer_v1
    └── training
        └── train.py
```

---

## Dataset Generation (re_arc Code)

We generate ARC-like datasets by leveraging code from `re_arc` (MIT licensed, [https://github.com/michaelhodel/re-arc](https://github.com/michaelhodel/re-arc)), included under `vitarc/external/re_arc`. See `LICENSE_re_arc` in that folder for more details.

A simple usage example:

```python
from vitarc.datasets.gen_dataset import generate_single_dataset_hf

task_key, final_ds, stats = generate_single_dataset_hf(
    task_idx=0,
    seed=1230,
    n_examples=1000,
    testsize=10
)
```

This returns a Hugging Face `DatasetDict` with `train`, `validation`, and `test` splits. See `tests/test_gen_dataset.py` for more detailed usage.

---

## Tokenizer Generation

We use a Hugging Face–style tokenizer for ARC tasks:

```python
from vitarc.tokenizers.arc_tokenizer import get_or_build_arc_tokenizer

tokenizer = get_or_build_arc_tokenizer("arc_tokenizer_v1")
```

This returns a tokenizer configured for ARC-like inputs/outputs. See `tests/test_tokenizer.py` for an example.

---

## Model Overview

**`ViTARCForConditionalGeneration`** is a specialized T5-based model that introduces:

- 2D absolute positional embeddings (`ape_type="SinusoidalAPE2D"`)
- Relative attention with four-slope Alibi (`rpe_type="Four-diag-slope-Alibi"`)
- Object-based embeddings (`use_OPE=True`)
- Custom embedding mixer strategies (`ape_mixer="weighted_sum_no_norm_vec"`), etc.

Example usage:

```python
from transformers import T5Config
from vitarc.models.model import ViTARCForConditionalGeneration

config = T5Config(
    vocab_size=len(tokenizer),
    d_model=128,
    num_layers=3,
    num_decoder_layers=3,
    num_heads=8,
    d_ff=256,
    dropout_rate=0.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,
    rows=33,
    cols=34,
    ape_type="SinusoidalAPE2D",
    rpe_type="Four-diag-slope-Alibi",
    rpe_abs=True,
    use_OPE=True,
    ape_mixer="weighted_sum_no_norm_vec"
)

model = ViTARCForConditionalGeneration(config)
```

See `vitarc/training/train.py` for a full training loop based on PyTorch Lightning.

---

## Citation & License

**License**: MIT.

**If you use or reference this code, please cite**:

> **Tackling the Abstraction and Reasoning Corpus with Vision Transformers: the Importance of 2D Representation, Positions, and Objects**  
> [arXiv:2410.06405](https://arxiv.org/abs/2410.06405)

This code relies on:
- Hugging Face Transformers
- PyTorch & PyTorch Lightning
- re_arc (MIT license) for generating ARC tasks
- Datasets (Hugging Face)
