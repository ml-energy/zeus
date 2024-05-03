# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetuning a 🤗 Transformers model for sentiment analysis on Capriccio."""

import os
import argparse
import logging
import random
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)

from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import GlobalPowerLimitOptimizer
from zeus.optimizer.batch_size import BatchSizeOptimizer, JobSpec
from zeus.utils.lr_scaler import SquareRootScaler

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    ########################## Zeus ##########################
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the Capriccio dataset is stored.",
        required=True,
    )
    parser.add_argument(
        "--slice_number",
        type=int,
        help=(
            "Which Capriccio dataset slice to use. "
            "Together with --data_dir, the paths to the train and val files are determined."
        ),
        required=True,
    )
    parser.add_argument(
        "--target_metric",
        default=0.84,
        type=float,
        help="Stop training when the target metric is reached.",
    )
    ##########################################################

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4.00e-7,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="A seed for reproducible training."
    )
    args = parser.parse_args()

    ########################## Zeus ##########################
    # Determine the paths to the Capriccio train and val files.
    train_file = Path(args.data_dir) / f"{args.slice_number}_train.json"
    if not train_file.exists():
        raise ValueError(f"'{train_file}' does not exist")
    args.train_file = str(train_file)

    val_file = Path(args.data_dir) / f"{args.slice_number}_val.json"
    if not val_file.exists():
        raise ValueError(f"'{val_file}' does not exist")
    args.val_file = str(val_file)
    ##########################################################

    return args


def main() -> None:
    """Run the main training routine."""
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    ########################## Zeus ##########################
    monitor = ZeusMonitor(gpu_indices=[0])  # Assumes single-GPU training.
    plo = GlobalPowerLimitOptimizer(monitor)
    bso = BatchSizeOptimizer(
        monitor=monitor,
        server_url=os.environ["ZEUS_SERVER_URL"],
        job=JobSpec(
            job_id=os.environ.get("ZEUS_JOB_ID"),
            job_id_prefix="capriccio",
            default_batch_size=128,
            batch_sizes=[8, 16, 32, 64, 128],
            max_epochs=args.max_epochs,
            target_metric=args.target_metric,
            window_size=10,
        ),
    )
    # Fetch the batch size from the BSO server.
    batch_size = bso.get_batch_size()
    print("Chosen batach size:", batch_size)
    # Scale the learning rate accordingly.
    # Default was batch size 128 and learing rate 4.00e-7, and we use the square root
    # scaling rule since the optimizer is AdamW.
    args.learning_rate = SquareRootScaler(bs=128, lr=4.00e-7).compute_lr(new_bs=batch_size)
    ##########################################################

    # Load the specific slice of the Capriccio dataset.
    data_path = dict(train=args.train_file, validation=args.val_file)
    logger.info("Using dataset slice: %s", data_path)
    raw_datasets = load_dataset("json", data_files=data_path)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, finetuning_task=None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    sentence1_key = "text"
    sentence2_key = None

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info("Sample %s of the training set: %s.", index, train_dataset[index])

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Send model to CUDA.
    model = model.cuda()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Compute the total number of training steps.
    args.max_train_steps = args.max_epochs * len(train_dataloader)

    # Get the metric function
    metric = load_metric("accuracy")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  Max Epochs = %s", args.max_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %s",
        batch_size,
    )
    logger.info("  Total optimization steps = %s", args.max_train_steps)
    if args.target_metric is not None:
        logger.info("  Target metric = %s", args.target_metric)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    ########################## Zeus ##########################
    bso.on_train_begin()

    for epoch in range(args.max_epochs):
        plo.on_epoch_begin()

        model.train()
        for batch in train_dataloader:
            plo.on_step_begin()

            for key, val in batch.items():
                if torch.is_tensor(val):
                    batch[key] = val.cuda()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            plo.on_step_end()

        plo.on_epoch_end()

        model.eval()
        for batch in eval_dataloader:
            for key, val in batch.items():
                if torch.is_tensor(val):
                    batch[key] = val.cuda()
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info("epoch %s: %s", epoch, eval_metric)

        bso.on_evaluate(eval_metric["accuracy"])

        if bso.training_finished:
            break
        ##########################################################

if __name__ == "__main__":
    main()
