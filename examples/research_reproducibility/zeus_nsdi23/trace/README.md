# Zeus traces

While the existence of recurring jobs in production GPU clusters is clear, it is not always easy to run 50 DNN training jobs in a sequential manner to evaluate energy optimization methods.
Thus, Zeus provides a trace-driven simulator that allows users to plug in their own customized batch size optimizer and power limit optimizers and observe gains.

We provide two types of traces (the *train trace* and the *power trace*) for six distinct dataset and network pairs:

| dataset      | network           |
|:-------------|:------------------|
| cifar100     | shufflenetv2      |
| imagenet     | resnet50          |
| librispeech  | deepspeech2       |
| movielens-1m | ncf               |
| squad        | bert_base_uncased |
| sentiment140 | bert_base_uncased |


## Train trace (`summary_train.csv`)

| Columns       | Description                                        | Example value |
|:--------------|:---------------------------------------------------|:--------------|
| dataset       | The name of the dataset                            | imagenet      |
| network       | The name of the DNN model                          | resnet50      |
| batch_size    | The batch size used for training                   | 256           |
| optimizer     | The optimizer used for training                    | Adadelta      |
| learning_rate | The learning rate used for the optimizer           | 2.6e-5        |
| run           | The repitition index of the same config            | 4             |
| target_metric | The target validation metric                       | 0.45          |
| target_epoch  | The epoch the target validation metric was reached | 9             |

We trained six different (model, dataset) pairs with many different batch sizes.
We repeated training for each (model, dataset, batch size) configuration with at least four different random seeds, with run with the same configuration distinguished with the `run` column in the trace.
Thus, when we would like to know the result of training a model on a dataset with a certain batch size, we can sample a *training path* from this trace with the specific triplet.


## Power trace (`summary_power_{gpu}.csv`)

| Columns        | Description                                       | Example value         |
|:---------------|:--------------------------------------------------|:----------------------|
| dataset        | The name of the dataset                           | imagenet              |
| network        | The name of the DNN model                         | resnet50              |
| batch_size     | The batch size used for training                  | 256                   |
| optimizer      | The optimizer used for training                   | adadelta              |
| power_limit    | The power limit of the GPU in Watts               | 225                   |
| time_per_epoch | The duration of training one epoch in seconds     | 11198.95              |
| average_power  | The average power consumption of the GPU in Watts | 116.63087520008567    |

We profiled the the duration of one epoch and average power consumption for six (model, dataset) pairs with many different (batch size, power limit) configurations.
These results not stochastic, and hence only measured once.

---

# Alibaba group trace

We mapped our six workloads to the tasks in the [Alibaba GPU cluster trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020).
Please refer to our paper for details on how jobs are mapped to tasks in the Alibaba trace.

| Columns        | Description                                                     | Example value                    |
|:---------------|:----------------------------------------------------------------|:---------------------------------|
| group          | The group ID of the job (from Alibaba)                          | 0001086788b7de0f13804d22a12a27db |
| start_time     | The start time of the job in seconds (from Alibaba)             | 1427642.0                        |
| end_time       | The end time of the job in seconds (from Alibaba)               | 1446915.0                        |
| dataset        | The dataset name of this job (from our workloads)               | squad                            |
| runtime_ratio  | This job's runtime / mean runtime of all jobs with this dataset | 1.1779405764729405               |

Especially, the `runtime_ratio` column captures the intra-cluster (i.e., intra-dataset) job runtime variation.
Without this, we'd be flattening the runtime of all jobs with the same dataset to a constant value, which is not desirable.

Please also consider citing the Alibaba GPU cluster trace paper if you're utilizing this trace:
```BibTeX
@inproceedings{weng2022mlaas,
  title={{MLaaS} in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous {GPU} Clusters},
  author={Weng, Qizhen and Xiao, Wencong and Yu, Yinghao and Wang, Wei and Wang, Cheng and He, Jian and Li, Yong and Zhang, Liping and Lin, Wei and Ding, Yu},
  booktitle={19th $\{$USENIX$\}$ Symposium on Networked Systems Design and Implementation ($\{$NSDI$\}$ 22)},
  year={2022}
}
```
