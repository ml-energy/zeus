<div align="center">
<h1>Zeus: Understanding and Optimizing<br>GPU Energy Consumption of DNN Training</h1>
</div>

You can read the entire paper here: [https://arxiv.org/abs/2208.06102](https://arxiv.org/abs/2208.06102)

<div class="critic-dark" markdown>
{==

## Abstract

Training deep neural networks (DNNs) is becoming more and more resource- and energy-intensive every year. Unfortunately, existing works primarily focus on optimizing DNN training for faster completion, often without considering the impact on energy efficiency.

In this paper, we observe that common practices to improve training performance can often lead to inefficient energy usage. More importantly, we demonstrate that there is a tradeoff between energy consumption and performance optimization. To this end, we propose an optimization framework, Zeus, to navigate this tradeoff by automatically finding optimal job- and GPU-level configurations for recurring DNN training jobs. Zeus uses an online exploration-exploitation approach in conjunction with just-in-time energy profiling, averting the need for expensive offline measurements, while adapting to data drifts over time. Our evaluation shows that Zeus can improve the energy efficiency of DNN training by 15.3%--75.8% for diverse workloads.

==}
</div>

## Why care about GPU energy?

In recent years, both academia and industry have seen an increasing adoption of DNNs for intelligent applications, and to support such growth, large clusters of GPUs were created.

GPUs are power-hungry hardware; it was reported that GPUs consume more than 70% of the power of the entire server when training DNNs.[^1]
At extreme scales, training the GPT-3 model once consumes 1,287 MWh,[^2] which is enough to supply an average US household for 120 years.[^3]

How can we optimize the GPU energy consumption of DNN training?


## Opportunity for energy savings

Then, how much energy saving is possible in the ideal case?

To see this, we trained the same DNN multiple times using a sweep of possible **batch sizes** and **GPU power limits**.[^5]

<figure>
  <br>
  <img src="img/eta-potential-all-v100-dark.svg" width=600px>
  <figcaption>Potential energy savings on an NVIDIA V100 GPU.</figcaption>
</figure>

The shorter the bar, the less the energy consumed.
You can see that just tuning one of batch size or power limit leads to energy savings, and when we optimize both at the same time, potential savings can be quite large.


## The trade-off between time and energy

The first step of optimizing something must be understanding it.



$$
\mathtt{Cost}
$$


<!-- Abbreviation definitions -->
*[DNN]: Deep Neural Network
*[DNNs]: Deep Neural Networks
*[ETA]: Energy to Accuracy
*[TTA]: Time to Accuracy


[^1]: Dodge, J., Prewitt, T., Combes, R., Odmark, E., Schwartz, R., Strubell, E., Luccioni, A., Smith, N., DeCario, N., & Buchanan, W. (2022). Measuring the Carbon Intensity of AI in Cloud Instances. In 2022 ACM Conference on Fairness, Accountability, and Transparency (pp. 1877–1894). Association for Computing Machinery.
[^2]: Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.M., Rothchild, D., So, D., Texier, M., & Dean, J.. (2021). Carbon Emissions and Large Neural Network Training.
[^3]: How much electricity does an American home use? https://www.eia.gov/tools/faqs/faq.php?id=97&t=3.
[^4]: When we say accuracy, we specifically mean the task validation metric because that's what captures how well the DNN generalizes to never-before-seen data. Please refer to our paper for details on all workloads.
[^5]: It is possible to cap the maximum power draw of a GPU using [NVML](https://developer.nvidia.com/nvidia-management-library-nvml).
