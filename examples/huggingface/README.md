# Integrating Zeus with HuggingFace 🤗

This example will demonstrate how to integrate Zeus's `HFGlobalPowerLimitOptimizer` with HuggingFace Transformers:
- `run_clm.py`: Transformers [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) for **causal langauge modeling** (i.e., pre-training)
- `run_gemma_sft_qlora.py`: TRL [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer) for **Gemma 7b supervised fine-tuning with QLoRA**

## Dependencies

To run the `Trainer` integration script (`run_clm.py`):
```sh
pip install -r requirements.txt
```

To run the `SFTTrainer` integration script (`run_gemma_sft_qlora.py`):
```sh
pip install -r requirements-qlora.txt
```
Note that you may have to tweak `requirements-qlora.txt` depending on your setup. The current requirements file assumes that you are using CUDA 11, and installs `nvidia-cusparse-cu11` for `bitsandbytes`. Basically, you want to get a setup where training runs, and just add `pip install zeus` on top of it.

## `ZeusMonitor` and `HFGlobalPowerLimitOptimizer`

- [`ZeusMonitor`](https://ml.energy/zeus/reference/monitor/energy/#zeus.monitor.energy.ZeusMonitor): Measures the GPU time and energy consumption of arbitrary code blocks.
- [`HFGlobalPowerLimitOptimizer`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer): Online-profiles each power limit with `ZeusMonitor` and finds the cost-optimal power limit. Calls GlobalPowerLimitOptimizer under the hood.

## Integration with HuggingFace Trainer
For easy use with [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/index), [`HFGlobalPowerLimitOptimizer`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer) is a drop-in compatible [HuggingFace Trainer Callback](https://huggingface.co/docs/transformers/en/main_classes/callback). When initializing a [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), initialize and pass in [`HFGlobalPowerLimitOptimizer`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer) as shown below:

```python
    monitor = ZeusMonitor()
    optimizer = HFGlobalPowerLimitOptimizer(monitor)

    # Also works for SFTTrainer.
    trainer = Trainer(
        ...,
        callbacks=[optimizer], # Add the `HFGlobalPowerLimitOptimizer` callback
    )
```

## Running the Example

By default, `Trainer`/`SFTTrainer` will make use of all available GPUs. If you would like to use only a subset of the GPUs, specify the `CUDA_VISIBLE_DEVICES` environment variable, which Zeus will also automatically respect.

```bash
# For Trainer.
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm

# For SFTTrainer.
python run_gemma_sft_qlora.py \
    --dataset_name stingning/ultrachat
```
