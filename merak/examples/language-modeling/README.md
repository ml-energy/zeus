<!---
Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.

Maintainer: TXacs (txacs1993@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Language modeling training

Merak is adaptive for `transformers.utils.fx`, so if the model can be traced by `transformers.utils.fx`, it will be run with 3D parallelism in Merak.

## Training language model with 3D parallelism

---

Running according to following bash:

```bash
python -m torch.distributed.launch --nproc_per_node=4  run_t5.py \
                --model-name t5-base \
                --data-files ./train_context.csv \
                --cache-dir ./t5_cache \
                --output_dir ./output \
                --per_device_train_batch_size 4 --gradient_accumulation_steps 4


python -m torch.distributed.launch --nproc_per_node=4  run_gpt.py \
                --model-name gpt2 \
                --data-files ./train_context.csv \
                --cache-dir ./gpt2_cache \
                --output_dir ./output \
                --per_device_train_batch_size 4 --gradient_accumulation_steps 4


python -m torch.distributed.launch --nproc_per_node=4  run_bert.py \
                --model-name bert-large-uncased  \
                --data-files ./train_context.csv \
                --cache-dir ./bert_cache \
                --output_dir ./output \
                --remove_unused_columns false \
                --per_device_train_batch_size 4 --gradient_accumulation_steps 4
```

Code is based on [transformers](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling) repository.

`train_context.csv` is from [Pretraining_T5_custom_dataset](https://github.com/joeljang/Pretraining_T5_custom_dataset), just for training test.