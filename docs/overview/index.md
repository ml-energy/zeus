<div align="center">
<h1>Zeus: Understanding and Optimizing<br>GPU Energy Consumption of DNN Training</h1>
</div>

This page intends to give a high-level overview of *what* Zeus does and *why* you should care.

For details, especially regarding *how*, please refer our [NSDI '23 publication](https://arxiv.org/abs/2208.06102).

<div class="critic-dark" markdown>
{==

## Abstract

Training DNNs is becoming more and more resource- and energy-intensive every year. Unfortunately, existing works primarily focus on optimizing DNN training for faster completion, often without considering the impact on energy efficiency.

In this paper, we observe that common practices to improve training performance can often lead to inefficient energy usage. More importantly, we demonstrate that there is a tradeoff between energy consumption and performance optimization. To this end, we propose an optimization framework, Zeus, to navigate this tradeoff by automatically finding optimal job- and GPU-level configurations for recurring DNN training jobs. Zeus uses an online exploration-exploitation approach in conjunction with just-in-time energy profiling, averting the need for expensive offline measurements, while adapting to data drifts over time. Our evaluation shows that Zeus can improve the energy efficiency of DNN training by 15.3%--75.8% for diverse workloads.

==}
</div>

## Why care about GPU energy?

In recent years, both academia and industry have seen an increasing adoption of DNNs for intelligent applications.
Large clusters of GPUs were created to support such growth, and the surge continues.

GPUs are power-hungry hardware; it was reported that GPUs consume more than 70% of the power of the entire server when training DNNs.[^1]
At extreme scales, training the GPT-3 model just once consumes 1,287 MWh,[^2] which is enough to supply an average US household for 120 years.[^3]

How can we optimize the GPU energy consumption of DNN training?


## Opportunity for energy savings

Our observation is that common practices to minimize the completion time of DNN training can lead to inefficient energy usage.

To see this, we trained[^4] the same DNN multiple times using a sweep of possible **batch sizes** and **GPU power limits**.[^5]

<figure>
  <br>
  <img src="img/eta-potential-all-v100-dark.svg#only-dark" width=600px>
  <img src="img/eta-potential-all-v100-light.svg#only-light" width=600px>
  <figcaption>Potential energy savings on an NVIDIA V100 GPU.</figcaption>
</figure>

The shorter the bar, the less the energy consumed.
You can see that just tuning one of batch size or power limit already leads to energy savings, and when we optimize both at the same time, potential savings can be quite large.


## The trade-off between time and energy

The first step of optimizing something would be understanding it.

The highlight of Zeus is that we discover and characterize the trade-off between DNN training time and energy consumption.

<div class="grid" markdown>

<figure>
  <br>
  <img src="img/pareto-annotated-librispeech-dark.svg#only-dark" width=600px>
  <img src="img/pareto-annotated-librispeech-light.svg#only-light" width=600px>
  <figcaption>All (batch size, power limit) configurations and their time/energy consumption.</figcaption>
</figure>

<figure>
  <br>
  <img src="img/pareto-librispeech-dark.svg#only-dark" width=600px>
  <img src="img/pareto-librispeech-light.svg#only-light" width=600px>
  <figcaption>The energy-time Pareto frontier zoomed in.</figcaption>
</figure>

</div>

These results are from training the DeepSpeech2 model with LibriSpeech on an NVIDIA V100 GPU.
Notice the yellow Pareto front consisted of efficient (time, energy) pairs, resulting from a set of efficient (batch size, power limit) knobs.


## A unified metric

All points on the energy-time Pareto frontier are efficient, but which one is the best?

Different users will have different answers, because they have different preferences of how they would like to trade off time and energy.
For instance, some production training jobs might have tight deadlines; they probably don't want to trade time for energy savings.
On the other hand, exploratory training jobs may have more leeway; it might make sense for them to reduce energy consumption at the cost of longer training time.

To allow users to express their trade-off preference, we define a simple cost metric[^6]

$$
\textrm{Cost} = \eta \cdot \textrm{Energy} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{Time,}
$$

where the user picks the value of $\eta$ between 0 and 1.
Smaller $\eta$ values will reduce more time, while larger ones will prefer to reduce more energy.


## Finding optimal knobs

Given the user's preference via the value of $\eta$, how do we find the best (batch size, power limit) knob on the Pareto frontier?

This is no easy problem because:

- GPU power consumption is a black box function of numerous hardware- and workload-dependent factors, and
- the time the DNN will take to reach its target validation metric is very difficult to predict.

Fortunately, DNN training jobs often **recur** in production GPU clusters,[^7] allowing us to explore, observe, and optimize **across job recurrences**.

This results in two main component in Zeus's design:

- **A JIT energy profiler**. Implemented inside [`ZeusDataLoader`][zeus.run.dataloader.ZeusDataLoader], this module searches for the optimal power limit. It efficiently profiles the time and energy characteristics of the DNN training job in a completely online manner.
- **MAB with Thompson Sampling**. Implemented inside [`ZeusMaster`][zeus.run.master.ZeusMaster], this module searches for the optimal batch size. Thanks to its MAB formulation, it is able to intelligently trade off exploration and exploitation across job recurrences.


<!-- Abbreviation definitions -->
*[DNN]: Deep Neural Network
*[DNNs]: Deep Neural Networks
*[GPU]: Graphics Processing Unit
*[GPUs]: Graphics Processing Units
*[JIT]: Just-in-Time
*[MAB]: Multi-Armed Bandit


[^1]: Jesse Dodge, Taylor Prewitt, Remi Tachet des Combes, Erika Odmark, Roy Schwartz, Emma Strubell, Alexandra Sasha Luccioni, Noah A. Smith, Nicole DeCario, and Will Buchanan. Measuring the carbon intensity of ai in cloud instances. In 2022 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’22, page 1877–1894, New York, NY, USA, 2022. Association for Computing Machinery.
[^2]: David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. arXiv preprint arXiv:2104.10350, 2021.
[^3]: How much electricity does an American home use? https://www.eia.gov/tools/faqs/faq.php?id=97&t=3.
[^4]: In all cases of training, we train until the DNN reaches a specific target validation metric. Thus, when we say time, it's TTA (Time To Accuracy). Likewise for energy, it's ETA (Enerty To Accuracy). Please refer to our paper for the complete workload table.
[^5]: It is possible to cap the maximum power draw of a GPU using [NVML](https://developer.nvidia.com/nvidia-management-library-nvml).
[^6]: $\textrm{MaxPower}$ is the maximum possible power limit of the GPU. It's just a constant number introduced to equalize the units of the left and right terms to Joules.
[^7]: Kim Hazelwood, Sarah Bird, David Brooks, Soumith Chintala, Utku Diril, Dmytro Dzhulgakov, Mohamed Fawzy, Bill Jia, Yangqing Jia, Aditya Kalro, et al. Applied machine learning at facebook: A datacenter infrastructure perspective. In 2018 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 620–629. IEEE, 2018.
