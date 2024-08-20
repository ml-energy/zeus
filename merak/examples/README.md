## Perseus + Merak examples

### Running training

`run.sh` can run both single node training and multi-node training.

Single-node training:

```bash
bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES]
```

Multi-node training (two nodes in example below):

```bash
# Run on master node
NNODES=2 NODE_RANK=0 MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES]

# Run on non-master node
NNODES=2 NODE_RANK=1 MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES]
```

Using only a subset of a node's GPUs (two GPUs in example below) and setting the 3D parallelism degrees:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES] --pp 2 --dp 1 --tp 1
```

Notable command line arguments that `run.sh` accepts are:
- `--partition_method`: Pipeline stage partitioning policy to use. `uniform_transformer` puts the same number of transformer layers in each stage, and it's only applicable to Transformer-based models. Custom partitioning can be used by passing `custom:0,8,14,20,26` for example, which means *four stages where each stage begins with the layer ID 0, 8, 14, and 20, and the model itself has 26 layers in total*.
- `--export_timing_csv`: Setting this to `true` will make Merak run in profiling mode. At termination, Merak will output `instructions-%d.csv` and `time-energy-%d.csv` (`%d` is the rank ID) in its output directory, where the former records the start and end timestamp of each pipeline instruction and the latter the timestamp and GPU cumulative energy consumption over time. Used together with the `InstructionProfiler` scheduler in the Perseus server, these files are used to derive the time/energy profile of each stage.
- `--num_prof_steps`: How many training iterations to measure for each GPU frequency plan. In other words, Merak will query the Perseus server for the next frequency plan, run training for `num_prof_steps` iterations, and report the aggregated time and energy measurements of the `num_prof_steps` iterations to the Perseus server.

See `Merak/utils/merak_args.py` for all arguments and defaults.

### List of supported models

| `[MODEL_NAME]` | Parameters |
|---|---:|
| `bert-base-uncased` | 109.43M |
| `bert-large-uncased` | 334.95M |
| `bert-huge-uncased` | 1276.40M |
| `gpt3-small` | 124.44M |
| `gpt3-medium` | 354.82M |
| `gpt3-large` | 758.73M |
| `gpt3-xl` | 1313.63M |
| `gpt3-2.7b` | 2648.93M |
| `gpt3-6.7b` | 6654.21M |
| `gpt3-13b` | 12848.14M |
| `gpt3-39b` | 39491.63M |
| `gpt3-80b` | 80600.16M |
| `gpt3-175b` | 174591.68M |
| `t5-small` | 60.51M |
| `t5-base` | 222.90M |
| `t5-large` | 737.67M |
| `t5-3b` | 3B |
| `t5-11b` | 11B |
| `wide-resnet50_2` | 68.88M |
| `wide-resnet50_4` | 223.44M |
| `wide-resnet50_8` | 804.16M |
| `wide-resnet101_2` | 126.89M |
| `wide-resnet101_4` | 419.63M |
| `wide-resnet101_8` | 1517.37M |

### Commands used for the paper

Below are the training commands we ran for each model.
For pipeline stage time/energy profiling, you need to append `--export_timing_csv true` to the end of each command.
Be it profiling or not, the larger the value of `--num_prof_steps`, the more stable the measurements will be (as it will mitigate outlier measurements) and the longer it will take for experiments to run (linearly proportional).

#### A100 4-stage pipeline parallelism

```sh
bash run.sh bert-huge-uncased 8 32 --partition_method custom:0,8,14,20,26 --num_prof_steps 30
bash run.sh bloom-3b 4 128 --partition_method custom:0,10,18,26,32 --num_prof_steps 4
bash run.sh gpt3-xl 8 128 --partition_method custom:0,8,14,21,27 --num_prof_steps 4
bash run.sh t5-3b 4 32 --partition_method custom:0,16,33,53,74 --num_prof_steps 30
bash run.sh wide-resnet101_8 64 24 --partition_method custom:0,7,16,25,34 --num_prof_steps 20
```

Note:

- These commands were run on one node with four A100 GPUs. See the main README for the exact specification of this node.

#### A40 8-stage pipeline parallelism

```sh
NNODES=2 NODE_RANK=[NODE_RANK] MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh bert-huge-uncased 8 32 --partition_method uniform_transformer --num_prof_steps 40 --pp 8 --dp 1 --tp 1
NNODES=2 NODE_RANK=[NODE_RANK] MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh bloom-3b 4 128 --partition_method custom:0,6,10,14,18,22,26,30,32 --num_prof_steps 5 --pp 8  --dp 1 --tp 1
NNODES=2 NODE_RANK=[NODE_RANK] MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh gpt3-2.7b 4 256 --partition_method uniform_transformer --num_prof_steps 4 --pp 8 --dp 1 --tp 1
NNODES=2 NODE_RANK=[NODE_RANK] MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh t5-3b 4 32 --partition_method custom:0,9,16,24,33,43,53,63,74 --num_prof_steps 20 --pp 8 --dp 1 --tp 1
NNODES=2 NODE_RANK=[NODE_RANK] MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh wide-resnet101_8 32 48 --partition_method custom:0,4,9,13,17,21,25,30,34 --num_prof_steps 20 --pp 8 --dp 1 --tp 1
```

Note:

- These commands were run on two nodes, each with four A40 GPUs. See the main README for the exact specification of these nodes.
- `[NODE_RANK]` should be `0` on the master node and `1` on the other node.
- `[MASTER_NODE_ADDR]` should be the address of the master node; it has been masked out in this artifact release in order not to publicly expose the address of our servers.
