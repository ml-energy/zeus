# Integrating Zeus with HuggingFace 🤗

This example will demonstrate how to integrate Zeus's `CarbonEmissionMonitor` with HuggingFace Transformers:

`run_clm.py`: Transformers [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) for **causal langauge modeling** (i.e., pre-training)

## Dependencies

To run the `Trainer` integration script (`run_clm.py`):
```sh
pip install -r requirements.txt

## `CarbonEmissionMonitor`

[`CarbonEmissionMonitor`](https://ml.energy/zeus/reference/monitor/carbon/#zeus.monitor.carbon.CarbonEmissionMonitor): Measures the GPU time, energy consumption, and carbon emission of arbitrary code blocks.

## Running the Example

By default, `Trainer` will make use of all available GPUs. If you would like to use only a subset of the GPUs, specify the `CUDA_VISIBLE_DEVICES` environment variable, which Zeus will also automatically respect.

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
