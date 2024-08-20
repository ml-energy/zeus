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

# Running with Swin-transformer and torchvision models

This script shows that the torch model, which is not from `transformers` library, but can be traced by `torch.fx`, how to run with 3D parallelism in Merak.

Running with following bash:

```bash
python -m torch.distributed.launch --nproc_per_node=4 run_swin.py   \
                --cfg ./swin_base_patch4_window7_224.yaml  \
                --output_dir ./output \
                --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
                --num_layers 16 --wall_clock_breakdown True --logging_steps 10 \
                --data_path /path/to/datasets
```

Code is based on [Swin-transformer](https://github.com/microsoft/Swin-Transformer) repository.