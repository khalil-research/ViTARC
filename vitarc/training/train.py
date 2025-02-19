#######################################
# vitarc/training/train.py
#######################################
import os
import re
import sys
import time
import shutil
import argparse
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import deque
from functools import reduce
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# If you have wandb logging:
import wandb
os.environ["WANDB_MODE"] = "offline"

# from huggingface/transformers
from transformers import (
    T5Config,
    AutoTokenizer,
    set_seed,
    get_linear_schedule_with_warmup    
)

from datasets import load_from_disk,load_metric

# Import your new model from vitarc
from vitarc.models.model import ViTARCForConditionalGeneration
from vitarc.datasets.gen_dataset import generate_single_dataset_hf, set_random_seed
from vitarc.tokenizers.arc_tokenizer import get_or_build_arc_tokenizer
from vitarc.external.re_arc.main import get_generators
# Or any other dataset generation methods you use

##################################################
# Global / environment config
##################################################
# e.g. for HPC
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to suppress parallel warnings
seed = 1230
set_random_seed(seed)

# For BLEU
#bleu = load_metric("./hf_sacrebleu.py", trust_remote_code=True)  # local path to HF metric
bleu = load_metric("sacrebleu", trust_remote_code=True) 

# If you have a local tokenizer:
tokenizer = get_or_build_arc_tokenizer("arc_tokenizer_v1")

##################################################
# Simple PL module wrapping your new model
##################################################
class ARCTrainerModule(pl.LightningModule):
    def __init__(
        self,
        config: T5Config,
        train_dataloader_len: int,
        lr: float = 5e-5,
        num_train_epochs: int = 15,
        warmup_steps: int = 1000,
    ):
        """
        A minimal LightningModule that wraps the new ViTARCForConditionalGeneration model.
        """
        super().__init__()
        self.save_hyperparameters()

        # Create the model
        self.model = ViTARCForConditionalGeneration(config)
        self.tokenizer = tokenizer

        # If you want to ensure your vocab size matches the tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.train_dataloader_len = train_dataloader_len
        self.lr = lr
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps

    def forward(self, input_ids, attention_mask, input_type_ids=None, labels=None):
        # Basic forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            object_idx=input_type_ids,
        )
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("input_type_ids"),
            batch.get("labels"),
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        total_steps = self.num_train_epochs * self.train_dataloader_len
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

##################################################
# Helper: Evaluate Model
##################################################
def evaluate_model(
    max_input_length,
    max_target_length,
    dataset_split,  # can be 'test' or a subset of dataset
    tokenizer,
    model_name,
    model_save_path,
    eval_model,
    task_id,
    ds_type="test",
    batch_size=16,
    force_rerun=False,
    rm_ws=True,
):
    """
    Evaluate model on 'dataset_split' (list or Dataset),
    store results in a pickled DF, and also compute:
      - exact match
      - token-level sacreBLEU
    If rm_ws=True, remove <s>, </s>, <pad>, and whitespace from both
    the generated output and the reference labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = eval_model.to(device)
    eval_model.eval()

    results_pkl = os.path.join(model_save_path, f"{ds_type}_results.pkl")
    metrics_csv = os.path.join(model_save_path, f"{ds_type}_metrics.csv")

    if os.path.exists(results_pkl) and not force_rerun:
        results_df = pd.read_pickle(results_pkl)
    else:
        results_df = pd.DataFrame(columns=["input", "label", "generated_output"])

        for start_idx in tqdm(range(0, len(dataset_split), batch_size)):
            end_idx = min(start_idx + batch_size, len(dataset_split))
            batch = dataset_split[start_idx:end_idx]

            # Tokenize inputs
            inputs = tokenizer(
                batch["input_text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_input_length,
            )
            input_ids = inputs.input_ids.to(device)

            # If you have 'input_type_ids' stored in your dataset:
            if "input_type_ids" in batch:
                object_idx = torch.tensor(batch["input_type_ids"]).to(device)
            else:
                object_idx = None

            # Generate
            with torch.no_grad():
                out_seqs = eval_model.generate(
                    input_ids,
                    max_length=max_target_length + 1,  # +1 for T5's start generation token
                    object_idx=object_idx,
                )

            # Decode raw strings
            decoded = [
                tokenizer.decode(seq, skip_special_tokens=False)
                for seq in out_seqs
            ]

            for i, gen_str in enumerate(decoded):
                # Possibly strip <s> / </s> / <pad> / whitespace from the generated text
                if rm_ws:
                    gen_str = re.sub(r"<s>|</s>|<pad>|\s+", "", gen_str)
                else:
                    gen_str = re.sub(r"<s>|</s>|<pad>", "", gen_str)

                # Also possibly strip from the label text
                lbl_str = batch["output_text"][i]
                if rm_ws:
                    lbl_str = re.sub(r"<s>|</s>|<pad>|\s+", "", lbl_str)
                else:
                    lbl_str = re.sub(r"<s>|</s>|<pad>", "", lbl_str)

                row = {
                    "input": batch["input"][i],
                    "output": batch["output"][i],
                    "label": lbl_str,
                    "generated_output": gen_str,
                }
                results_df = pd.concat(
                    [results_df, pd.DataFrame([row])],
                    ignore_index=True
                )

        results_df.to_pickle(results_pkl)

    # --------------------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------------------

    # 1) Exact Match
    def exact_match(truth, pred):
        return truth == pred

    exact_matches = [
        exact_match(row["label"], row["generated_output"])
        for _, row in results_df.iterrows()
    ]
    avg_exact = sum(exact_matches) / len(exact_matches) if len(exact_matches) else 0.0

    # 2) Token-level BLEU
    #    - Tokenize each label/pred. Then join with spaces to get a single string.
    #    - sacrebleu expects a list of strings for `predictions`,
    #      and a list of list-of-strings for `references`.
    tokenized_preds = []
    tokenized_refs = []
    for _, row in results_df.iterrows():
        pred_tokens = tokenizer.tokenize(row["generated_output"])
        ref_tokens  = tokenizer.tokenize(row["label"])

        pred_str = " ".join(pred_tokens)
        ref_str  = " ".join(ref_tokens)

        tokenized_preds.append(pred_str)
        tokenized_refs.append([ref_str])

    sacre_bleu = bleu.compute(
        predictions=tokenized_preds,
        references=tokenized_refs
    )["score"]

    # Log metrics in a CSV
    if os.path.exists(metrics_csv) and not force_rerun:
        mdf = pd.read_csv(metrics_csv)
    else:
        mdf = pd.DataFrame(columns=["model_name", "exact_match", "bleu"])

    new_row = {
        "model_name": model_name,
        "exact_match": avg_exact,
        "bleu": sacre_bleu,
    }
    mdf = pd.concat([mdf, pd.DataFrame([new_row])], ignore_index=True)
    mdf.to_csv(metrics_csv, index=False)

    print(f"[INFO] {ds_type} results => exact_match={avg_exact:.3f}, BLEU={sacre_bleu:.2f}")
    print("[INFO] Detailed saved =>", results_pkl)

    # ----------------------
    # ADDING THE SUMMARY CSV
    # ----------------------
    summary_dir = os.path.abspath(os.path.join(model_save_path, os.pardir, os.pardir))
    summary_csv_path = os.path.join(summary_dir, f"{ds_type}_summary_metrics.csv")

    summary_data = {
        "task_id": [task_id],
        "model_name": [model_name],
        "exact_match": [avg_exact],
        "bleu": [sacre_bleu],
    }
    summary_df = pd.DataFrame(summary_data)

    if os.path.exists(summary_csv_path):
        existing_summary_df = pd.read_csv(summary_csv_path)
        summary_df = pd.concat([existing_summary_df, summary_df], ignore_index=True)

    summary_df.to_csv(summary_csv_path, index=False)
    print("Summary metrics saved to:", summary_csv_path)


################################################################
# (Optional) HPC dataset existence checks
################################################################
def check_dataset_exists(folder):    
    if not os.path.exists(folder):
        return False
    # e.g. check subfiles
    subfolders = ["test", "train", "validation", "dataset_dict.json"]
    for sub in subfolders:
        if not os.path.exists(os.path.join(folder, sub)):
            return False
    return True

def copy_dataset_to_tmpdir(ds_src_dir, tmpdir):
    """
    Copy data to node-local scratch if needed.
    """
    task_folder = os.path.basename(ds_src_dir)
    dst = os.path.join(tmpdir, task_folder)
    if not os.path.exists(dst):
        shutil.copytree(ds_src_dir, dst)
    return dst

##################################################
# Main training flow
##################################################
def main():
    parser = argparse.ArgumentParser("ViTARC training script")
    parser.add_argument("--task_idx", type=int, default=0, help="Which re_arc task index to train on")
    parser.add_argument("--max_input_length", type=int, default=1124, help="Max input length")
    parser.add_argument("--max_target_length", type=int, default=1124, help="Max target length")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="max epochs")
    parser.add_argument("--seed", type=int, default=1230, help="seed")
    parser.add_argument("--ds_base_dir", type=str, default="./arc_x2y_datasets", help="Base folder for ARC datasets")
    parser.add_argument("--use_slurm_copy", action="store_true", help="If set, attempt to copy to SLURM_TMPDIR first") # This is for HPC env    
    args = parser.parse_args()    

    set_random_seed(args.seed)

    # Basic environment, load or create dataset
    ds_base_dir = args.ds_base_dir
    task_idx = args.task_idx
    generators_mapper = get_generators()
    task_keys = sorted(generators_mapper.keys())
    task_id = task_keys[task_idx]
    
    dataset_path = os.path.join(ds_base_dir, f"task{task_idx}_{task_id}_1M")
    num_examples = 1000000
    test_size = 1000
    num_workers = 10
    if check_dataset_exists(dataset_path):
        print(f"[INFO] Found dataset for task {task_idx} in base_dir={ds_base_dir}")
        if args.use_slurm_copy:
            slurm_tmpdir = os.getenv("SLURM_TMPDIR", "/tmp")
            dataset_path = copy_dataset_to_tmpdir(dataset_path, slurm_tmpdir)        

        print(f"[INFO] Loading from disk => {dataset_path}")
        dataset = load_from_disk(dataset_path)

    else:
        # If no dataset found => generate a small dataset
        print(f"[WARN] No dataset found for idx={task_idx}. Generating dataset of size 1M.")        
        key, dataset, stats = generate_single_dataset_hf(
            task_idx=task_idx,
            seed=args.seed,            
            n_examples=num_examples,
            testsize=test_size
        )
        print(f"[INFO] dataset => {stats}")        
        os.makedirs(dataset_path, exist_ok=True)        
        dataset.save_to_disk(dataset_path)
        print(f"[INFO] Saved dataset to {dataset_path}")

    # Format columns
    dataset["train"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "input_type_ids"]
    )
    dataset["validation"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "input_type_ids"]
    )

    train_dl = DataLoader(dataset["train"], shuffle=True, batch_size=args.batch_size, num_workers=num_workers)
    val_dl = DataLoader(dataset["validation"], batch_size=args.batch_size, num_workers=num_workers)
    train_len = len(train_dl)

    # Build T5 config
    # If you want bigger dimension, adjust:
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

    # Wrap in our PL module
    arc_module = ARCTrainerModule(
        config=config,
        train_dataloader_len=train_len,
        lr=5e-4,
        num_train_epochs=args.epochs,
        warmup_steps=1000
    )

    # Set up trainer
    early_stop_callback = EarlyStopping(
        monitor="validation_loss",
        patience=3,
        mode="min"
    )
    lr_monitor = LearningRateMonitor("step")
    
    model_save_dir = os.path.join(dataset_path, f"model")
    chkpoints_save_dir = os.path.join(dataset_path, f"checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=args.epochs,
        log_every_n_steps=10,
        callbacks=[early_stop_callback, lr_monitor],
        default_root_dir=chkpoints_save_dir
    )

    trainer.fit(arc_module, train_dl, val_dl)

    # Save final    
    arc_module.model.save_pretrained(model_save_dir)
    print("[INFO] Model saved to =>", model_save_dir)

    # Optionally evaluate on test set
    if "test" in dataset:
        # e.g. evaluate on a small subset
        #test_data = dataset["test"].select(range(min(100, len(dataset["test"]))))
        test_data = dataset["test"]
        evaluate_model(
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            dataset_split=test_data,
            tokenizer=tokenizer,
            model_name=f"ViTARC",
            model_save_path=model_save_dir,
            eval_model=arc_module.model,
            task_id=task_id,
            ds_type="test",
            batch_size=args.batch_size,
            force_rerun=True,
            rm_ws=True
        )

    print("[INFO] Training & evaluation complete.")

if __name__ == "__main__":    
    main()
