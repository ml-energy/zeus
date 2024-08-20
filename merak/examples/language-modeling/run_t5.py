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
from utils import create_tokenizer, Prepare_data
from config import load_config_and_model

from transformers import (
    default_data_collator,
    set_seed,
    HfArgumentParser,
)


def parse_option(parser):
    # easy config modification
    # parser.add_argument('--cache-dir', type=str, help='where to save cache')
    # parser.add_argument('--dataset-name', type=str, help='name of dataset from the datasets package')
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
    Merak.init(pp, tp, dp)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load model config and instantiate model.
    config, model = load_config_and_model(training_args.model_name)

    # Set shard_count based on number of layers
    # t5-small => 12 layers (embedding, 6 transformer encoders split into 4 layers, 6 transformer decoders, lm_head)
    # t5-base  => 22 layers (embedding, 12 transformer encoders split into 8 layers, 12 transformer decoders, lm_head)
    # t5-large => 42 layers (embedding, 24 transformer encoders split into 16 layers, 24 transformer decoders, lm_head)
    # training_args.shard_count = 1 + int(config.num_layers / 1.5) + config.num_decoder_layers + 1

    # Above is when we did uniform_transformer. Now we're optimally balancing.
    # 100 is just "a large number." This will put one encoder in one GraphModule and one decoder in two GraphModules,
    # where the self/cross-attention layers will be in one GraphModule and the FFN layer will be in another.
    training_args.shard_count = 100
    training_args.num_transformers = config.num_layers + config.num_decoder_layers
    training_args.num_initial_embeddings = 1

    # create tokenizer
    tokenizer = create_tokenizer(training_args.cache_dir, training_args.model_name, config)

    # create dataset
    dataset = Prepare_data(model, tokenizer, input_length=512, output_length=512)

    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset, 
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
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
