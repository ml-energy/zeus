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

# ZEUS
from zeus.run import ZeusDataLoader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    # CAPRICCIO
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where Capriccio is stored.",
        required=True,
    )
    parser.add_argument(
        "--slice_number",
        type=int,
        help=(
            "Which dataset slice to use."
            "Together with --data_dir, the paths to the train and val files are determined."
        ),
        required=True,
    )

    # ZEUS
    runtime_mode = parser.add_mutually_exclusive_group()
    runtime_mode.add_argument(
        "--zeus", action="store_true", help="Whether to run Zeus."
    )

    parser.add_argument(
        "--target_metric",
        default=None,
        type=float,
        help=(
            "Stop training when the target metric is reached. This is ignored when running in Zeus mode because"
            " ZeusDataLoader will receive the target metric via environment variable and stop training by itself."
        ),
    )

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
        "--batch_size",
        type=int,
        help="Batch size  for the training and eval dataloader.",
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    args = parser.parse_args()

    # CAPRICCIO
    if not (
        train_file := Path(args.data_dir) / f"{args.slice_number}_train.json"
    ).exists():
        raise ValueError(f"'{train_file}' does not exist")
    args.train_file = str(train_file)
    if not (val_file := Path(args.data_dir) / f"{args.slice_number}_val.json").exists():
        raise ValueError(f"'{val_file}' does not exist")
    args.val_file = str(val_file)

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

    # Load the dataset.
    # CAPRICCIO
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

    # CAPRICCIO
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

    # CAPRICCIO
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

    # ZEUS
    if args.zeus:
        train_dataloader = ZeusDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            max_epochs=args.num_train_epochs,
            shuffle=True,
            collate_fn=data_collator,
        )
        eval_dataloader = ZeusDataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=data_collator
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.batch_size,
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.batch_size
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
    args.max_train_steps = args.num_train_epochs * len(train_dataloader)

    # Get the metric function
    metric = load_metric("accuracy")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  Num Epochs = %s", args.num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %s",
        args.batch_size,
    )
    logger.info("  Total optimization steps = %s", args.max_train_steps)
    if args.target_metric is not None:
        logger.info("  Target metric = %s", args.target_metric)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    # ZEUS
    if args.zeus:
        assert isinstance(train_dataloader, ZeusDataLoader)
        epoch_iter = train_dataloader.epochs()
    else:
        epoch_iter = range(args.num_train_epochs)

    for epoch in epoch_iter:
        model.train()
        for batch in train_dataloader:
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

        # ZEUS
        if args.zeus:
            assert isinstance(train_dataloader, ZeusDataLoader)
            train_dataloader.report_metric(
                eval_metric["accuracy"], higher_is_better=True
            )
        # If this were Zeus, the train dataloader will stop training by itself.
        elif args.target_metric is not None:
            if eval_metric["accuracy"] >= args.target_metric:
                logger.info(
                    "Reached target metric %s in %s epochs.",
                    args.target_metric,
                    epoch + 1,
                )
                break


if __name__ == "__main__":
    main()
