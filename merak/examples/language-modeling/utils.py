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

# Parts of the code here are adapted from https://github.com/joeljang/Pretraining_T5_custom_dataset/blob/master/pretrain.py
# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/t5/modeling_t5.py
# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/language-modeling/run_clm.py
# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/language-modeling/run_mlm.py

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer
)
from torch.utils.data import Dataset
import pandas as pd
import math
import numpy as np
import os
import torch.distributed as dist

from bloom.tokenization_bloom_fast import BloomTokenizerFast


def load_wikitext(cache_dir, validation_split_percentage):
    raw_datasets = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', cache_dir=cache_dir)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'Salesforce/wikitext',
            'wikitext-103-raw-v1',
            split=f'train[:{validation_split_percentage}%]',
            cache_dir=cache_dir
        )
    return raw_datasets

# create dataset
def load_data(data_path, cache_dir, validation_split_percentage):
    data_files = {}
    dataset_args = {}
    if data_path is not None:
        data_files["train"] = data_path

    extension = (
        data_path.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = True
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, **dataset_args)


    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{validation_split_percentage}%]",
            cache_dir=cache_dir,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{validation_split_percentage}%:]",
            cache_dir=cache_dir,
            **dataset_args,
        )
    return raw_datasets

def check_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    files = os.listdir(cache_dir)
    exit_file = False
    for file in files:
        f = str(cache_dir+'/'+file)
        if os.path.isfile(f) and not file.startswith('.'):
            exit_file = True
            break
    return exit_file
            

def create_tokenizer(cache_dir, model_name, config):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": None,
    }

    TokenizerClass = BloomTokenizerFast if "bloom" in model_name else AutoTokenizer
    
    # Only rank 0 to download files
    if dist.get_rank() == 0:
        tokenizer = TokenizerClass.from_pretrained(model_name, 
            # local_files_only=check_cache(cache_dir), 
            config=config,
            **tokenizer_kwargs)
    dist.barrier()
    if dist.get_rank() != 0:
        tokenizer = TokenizerClass.from_pretrained(model_name, 
            # local_files_only=check_cache(cache_dir), 
            config=config,
            **tokenizer_kwargs)
    dist.barrier()
    return tokenizer

def preprocessing_datasets(datasets, tokenizer_func, model_name):
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = tokenizer_func.model_max_length
    if max_seq_length > 1024:
        max_seq_length = 1024

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    # if model_name == 'bert-large-uncased':
    if model_name.startswith("bert"):
        def tokenize_function(examples):
            return tokenizer_func(examples[text_column_name], return_special_tokens_mask=True)
    else:
        def tokenize_function(examples):
            output = tokenizer_func(examples[text_column_name])
            return output

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    # if model_name == 'bert-large-uncased':
    if model_name.startswith("bert"):
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
    # elif model_name == 'gpt2':
    elif model_name.startswith("gpt") or model_name.startswith("bloom"):
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
    )

    return lm_datasets['train'], lm_datasets['validation']


class Prepare_data(Dataset):
    def __init__(self, model, tokenizer, input_length, output_length, print_text=False):
      self.dataset = self.split_into_segment(pd.read_csv("./train_context.csv"),input_length)
      self.input_length = input_length
      self.tokenizer = tokenizer
      self.output_length = output_length
      self.print_text = print_text
      self.model = model

    def split_into_segment(self, ds, input_length):
        new_rows = []
        for index, row in ds.iterrows():
            if len(row['context'].split()) > input_length:
                word_list = row['context'].split()
                seg1 = word_list[:input_length]
                segment1, seg2_a = (' '.join(seg1)).rsplit('.',1)
                segment2 = seg2_a + (' '.join(word_list[input_length:]))
                ds.loc[index, 'context'] = segment1
                while(len(segment2.split()) > input_length):
                    word_list = segment2.split()
                    seg1_ = word_list[:input_length]
                    if '.' in ' '.join(seg1_):
                        segment1_, seg2_a_ = (' '.join(seg1_)).rsplit('.',1)
                        segment2 = seg2_a_ + (' '.join(word_list[input_length:]))
                    else:
                        segment1_ = ' '.join(seg1_)
                        segment2 = (' '.join(word_list[input_length:]))
                    new_rows.append(segment1_)
                new_rows.append(segment2)
        ds2 = pd.DataFrame(new_rows, columns=['context'])
        ds = ds.append(ds2)
        return ds

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text

    def span_corruption_mask(self, text, noise_span_length=3, noise_density=.15):
        max_index = len(text.split())
        mask = max_index * [0]
        span_num = math.ceil(( max_index * noise_density ) / 3 )
        exclude=[max_index-2, max_index-1]
        for i in range(span_num):
            while True:
                rand_num = np.random.randint(low=0, high=max_index) #Getting random number for mask index
                if rand_num not in exclude:
                    span = [rand_num, rand_num+1, rand_num+2]
                    for s in span:
                        mask[s] = 1
                        exclude.append(s)
                    if rand_num==1:
                        exclude.append(rand_num-1)
                    elif rand_num==2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                    elif rand_num>2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                        exclude.append(rand_num-3)
                    if not rand_num==max_index-3:
                        exclude.append(span[-1]+1)
                    break
                else:
                    continue
        return mask

    def noise_span_to_unique_sentinel(self, text, mask,sentinels):
        tokens = text.split()
        text_ = []
        one_count=0
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 1:
                one_count+=1
                if one_count==1:
                    text_.append(sentinels[sentinel_cnt])
                    sentinel_cnt+=1
                else:
                    if one_count==3:
                        one_count=0
            else:
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def nonnoise_span_to_unique_sentinel(self, text, mask,sentinels):
        tokens = text.split()
        text_ = []
        zero_first=True
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 0:
                if zero_first:
                    text_.append(sentinels[sentinel_cnt])
                    zero_first=False
                    sentinel_cnt+=1
            else:
                zero_first=True
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['context']))
        text = self.clean_text(example_batch['context'])
        mask = self.span_corruption_mask(text)
        sentinels=[]
        for i in range(100):
            sentinels.append(f'<extra_id_{i}>')
        input_ = self.noise_span_to_unique_sentinel(text,mask,sentinels)
        target_ = self.nonnoise_span_to_unique_sentinel(text,mask,sentinels)
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset.iloc[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        decoder_ids = self.model.prepare_decoder_input_ids_from_labels(target_ids)

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"input_ids": source_ids, "decoder_attention_mask": src_mask, "decoder_input_ids": decoder_ids, "labels": target_ids, "target_mask": target_mask}
