# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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

# using our distributed trainer
import Merak
from Merak import MerakArguments, MerakTrainer
from utils import create_tokenizer, preprocessing_datasets, load_wikitext
from config import load_config_and_model

from transformers import (
    DataCollatorForLanguageModeling,
    set_seed,
    HfArgumentParser,
)


def parse_option(parser):
    # easy config modification
    # parser.add_argument('--cache-dir', type=str, help='where to save cache')
    # parser.add_argument('--model-name', type=str, help='gpt2 or t5-base')
    parser.add_argument('--validation-split-percentage', type=int, default=5, help='split data for validation')
    parser.add_argument('--pp', type=int, default=4, help='Pipeline parallel degree')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel degree')
    parser.add_argument('--dp', type=int, default=1, help='Data parallel degree')

    return parser

def main():
    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # init dist
    pp = args.pp
    tp = args.tp
    dp = args.dp
    Merak.init(pp=pp, tp=tp, dp=dp)
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # create dataset
    raw_datasets = load_wikitext(training_args.cache_dir, args.validation_split_percentage)

    # Load model config and instantiate model.
    config, model = load_config_and_model(training_args.model_name)

    # Set `shard_count` based on number of layers
    # bert-base-uncased  => 14 layers (embedding, 12 transformer encoders, lm_head)
    # bert-large-uncased => 26 layers (embedding, 24 transformer encoders, lm_head)
    # bert-huge-uncased  => 26 layers (embedding, 24 transformer encoders, lm_head)
    training_args.shard_count = 1 + config.num_hidden_layers + 1
    training_args.num_transformers = config.num_hidden_layers
    training_args.num_initial_embeddings = 1

    # create tokenizer
    orig_model_name = training_args.model_name
    training_args.model_name = "bert-large-uncased"
    tokenizer = create_tokenizer(training_args.cache_dir, training_args.model_name, config)
    training_args.model_name = orig_model_name

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    train_dataset, eval_dataset = preprocessing_datasets(raw_datasets, tokenizer, training_args.model_name)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
    )

    # using our distributed trainer        
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
    )

    # Train or profile
    if training_args.profile:
        trainer.profile()
    else:
        train_result = trainer.train()
        if train_result is not None:
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()
