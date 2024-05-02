# Running Zeus in a Trace-Driven Fashion

While the existence of recurring jobs in production GPU clusters is clear, it is not always easy to run 50 DNN training jobs in a sequential manner to evaluate energy optimization methods.
Thus, Zeus provides a trace-driven simulator that allows users to plug in their own customized batch size optimizer and power limit optimizers and observe gains.

We provide two types of traces.  

1. Train trace: We trained six different (model, dataset) pairs with many different batch sizes. And we repeated training at least four times for each triplet with different random seeds. Thus, when we would like to know the result of training a model on a dataset with a certain batch size, we can sample a *training path* from this trace.
2. Power trace: We profiled the the duration of one epoch and average power consumption for six (model, dataset) pairs with many different (batch size, power limit) configurations. These results not stochastic, and can be fetched from the trace to construct TTA (time to accuracy) and ETA (energy to accuracy) values.

Refer to the [`trace`](../../trace/) directory for more information about the traces we provide.

## Simulating the recurrence of one job

With [`run_single.py`](run_single.py), you can simulate the optimization trajectory of one recurring job.

### Dependencies

1. Install `zeus` following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/). The power monitor is not needed.

All dependencies are already installed you're using our Docker image (see [Environment setup](https://ml.energy/zeus/getting_started/environment/)).

### Example command

```sh
# All arguments shown below are default values.
python run_single.py \
    --dataset librispeech \
    --model deepspeech2 \
    --optimizer adamw \
    --target_metric 40.0 \
    --max_epochs 16 \
    --b_0 192 \
    --gpu v100 \
    --eta_knob 0.5 \
    --beta_knob 2.0 \
    --seed 1
```

## Simulating jobs based on the Alibaba GPU cluster trace

With [`run_alibaba.py`](run_alibaba.py), you can simulate jobs in the [Alibaba GPU cluster trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020).

Please refer to our paper for details on how jobs in our train/power traces are mapped to tasks in the Alibaba trace.

### Dependencies

1. Install `zeus` following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/). The power monitor is not needed.

All dependencies are already installed you're using our Docker image (see [Environment setup](https://ml.energy/zeus/getting_started/environment/)).

### Example command

```sh
python run_alibaba.py \
    --gpu v100 \
    --eta_knob 0.5 \
    --beta_knob 2.0
```
