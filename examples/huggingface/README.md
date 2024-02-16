# Integrating Zeus with HuggingFace 🤗

This example will demonstrate how to integrate Zeus with `HuggingFace 🤗 Trainer` using `HFGlobalPowerLimitOptimizer`.

[`run_clm.py`](run_clm.py) was adapted from [HuggingFace 🤗's example training code for fine-tuning language models](https://github.com/huggingface/transformers/tree/f3aa7db439a2a3942f76c115197fe953984ac334/examples/pytorch/language-modeling).

## Dependencies

Use the included requirements.txt file to include all extra dependencies:
```sh
    pip install -r requirements.txt
```

## `ZeusMonitor` and `HFGlobalPowerLimitOptimizer`

- [`ZeusMonitor`](http://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor): Measures the GPU time and energy consumption of arbitrary code blocks.
- [`HFGlobalPowerLimitOptimizer`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer): Online-profiles each power limit with `ZeusMonitor` and finds the cost-optimal power limit. Calls GlobalPowerLimitOptimizer under the hood.

## Integration with HuggingFace 🤗 Trainer
For easy use with [HuggingFace 🤗 Transformers](https://huggingface.co/docs/transformers/en/index), [`HFGlobalPowerLimitOptimizer`](zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer) is a drop-in compatible [HuggingFace 🤗 Trainer Callback](https://huggingface.co/docs/transformers/en/main_classes/callback). When initializing a [HuggingFace 🤗 Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), initialize and pass in [`HFGlobalPowerLimitOptimizer`](zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer) as shown below:

```python
    monitor = ZeusMonitor()
    optimizer = HFGlobalPowerLimitOptimizer(monitor)

    # Initialize HuggingFace 🤗 Trainer
    trainer = Trainer(
        ...,
        callbacks=[optimizer], # Add the `HFGlobalPowerLimitOptimizer` callback
    )
```

## Fine-tuning Example
```bash
# Single-GPU example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 
# (no tokens were replaced before the tokenization). 
# The loss here is that of causal language modeling.
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gpu_indices="0" # Specify GPUs to ZeusMonitor. If left out, Zeus Monitor uses all available GPUs. 

# Multi-GPU example fine-tunes GPT-2 on WikiText-2 using 4 GPUs
torchrun \
    --nproc_per_node 4 run_clm.py \
	--model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gpu_indices="0,1,2,3" # Specify GPUs to ZeusMonitor. If left out, Zeus Monitor uses all available GPUs. 
```
