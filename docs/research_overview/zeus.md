---
title: Zeus (NSDI '23)
description: Understanding and Optimizing GPU Energy Consumption of DNN Training
---

<div align="center" markdown>
<h1>Zeus: Understanding and Optimizing<br>GPU Energy Consumption of DNN Training</h1>

NSDI '23

[**Paper**](https://www.usenix.org/conference/nsdi23/presentation/you) | [**Slides**](https://www.usenix.org/system/files/nsdi23_slides_chung.pdf) | [**YouTube**](https://youtu.be/aZoD-jgO3fE)
</div>

<div class="critic-dark" markdown>
{==

## Abstract

Training Deep Neural Networks (DNNs) is becoming more and more resource- and energy-intensive every year. Unfortunately, existing works primarily focus on optimizing DNN training for faster completion, often without considering the impact on energy efficiency.

In this paper, we observe that common practices of DNN training can lead to inefficient energy usage. More importantly, we demonstrate that there is a tradeoff between energy consumption and performance optimization. To this end, we propose an optimization framework, Zeus, to navigate this tradeoff by automatically finding optimal job- and GPU-level configurations for recurring DNN training jobs. Zeus does not require any offline profiling and can adapt to data drifts.

==}
</div>


## Why care about GPU energy?

Recent years have seen an increasing adoption of DNNs for intelligent applications.
Large clusters of GPUs were created to support such growth, and the surge continues.

GPUs are power-hungry hardware; GPUs consume ~ 70% of the power of the entire server when training DNNs.[^1]
At extreme scales, training the GPT-3 model just once consumes 1,287 MWh,[^2] which is enough to supply an average US household for 120 years.[^3]

However, latency and throughput have been the primary targets of existing optimization techniques, devoid of any careful consideration of how such optimizations might impact energy efficiency.
We argue that **energy** should be considered as the **third dimension**.


## Opportunity for energy savings

We observe that common practices of DNN training can often lead to energy inefficiency.

To see this, we trained[^4] the same DNN multiple times using a sweep of possible **batch sizes** and **GPU power limits**.[^5]

<figure>
  <br>
  <img src="../img/eta-potential-all-v100-dark.svg#only-dark" width=600px>
  <img src="../img/eta-potential-all-v100-light.svg#only-light" width=600px>
  <figcaption>Potential energy savings on an NVIDIA V100 GPU.</figcaption>
</figure>

The baseline dotted line uses the default batch size from the model's publication and the default (maximum) GPU power limit.
It can be seen that choosing the best batch size and power limit can lead to large energy savings.


## Tradeoff between time & energy

Is energy reduction free?

We discover that there is a **tradeoff** between DNN training time and energy consumption.

<div class="grid" markdown>

<figure>
  <br>
  <img src="../img/pareto-annotated-librispeech-dark.svg#only-dark" width=600px>
  <img src="../img/pareto-annotated-librispeech-light.svg#only-light" width=600px>
  <figcaption>All (batch size, power limit) configurations and their time/energy consumption.</figcaption>
</figure>

<figure>
  <br>
  <img src="../img/pareto-librispeech-dark.svg#only-dark" width=600px>
  <img src="../img/pareto-librispeech-light.svg#only-light" width=600px>
  <figcaption>The energy-time Pareto frontier zoomed in.</figcaption>
</figure>

</div>

These results are from training DeepSpeech2 on LibriSpeech with an NVIDIA V100 GPU.
Notice the yellow Pareto frontier of efficient (time, energy) pairs, resulting from a set of efficient (batch size, power limit) knobs.


## Navigating the tradeoff

All points on the Pareto frontier are efficient, but which one is the best?

Different users will have different answers, because they have different preferences of how they would like to trade off time and energy.[^6]

To allow users to express their tradeoff preference, we define a simple cost metric[^7]

$$
\textrm{Cost} = \eta \cdot \textrm{Energy} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{Time,}
$$

where the user picks the value of $\eta$ between 0 and 1.
Smaller $\eta$ values will reduce more time, while larger ones will prefer to reduce more energy.


## Finding the optimal knob

Given the user's preference via the value of $\eta$, how do we find the best (batch size, power limit) knob on the Pareto frontier?

This is no easy problem. We only have the Pareto frontier in the previous plot because we trained all possible combinations of batch size and power limit until completion to characterize the tradeoff.[^8]

Fortunately, DNN training jobs often **recur** in production GPU clusters,[^9] allowing us to explore, observe, and optimize **across job recurrences**.

This results in two main components in Zeus:

- **Just-In-Time energy profiler**: Finds the optimal power limit via online profiling.
- **Multi-Armed Bandit + Thompson Sampling**: Finds the optimal batch size across recurring training runs.


[^1]: Jesse Dodge, Taylor Prewitt, Remi Tachet des Combes, Erika Odmark, Roy Schwartz, Emma Strubell, Alexandra Sasha Luccioni, Noah A. Smith, Nicole DeCario, and Will Buchanan. Measuring the carbon intensity of ai in cloud instances. In 2022 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’22, page 1877–1894, New York, NY, USA, 2022. Association for Computing Machinery.
[^2]: David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. arXiv preprint arXiv:2104.10350, 2021.
[^3]: How much electricity does an American home use? https://www.eia.gov/tools/faqs/faq.php?id=97&t=3.
[^4]: In all cases of training, we train until the DNN reaches a specific target validation metric. Thus, when we say time, it's TTA (Time To Accuracy). Likewise for energy, it's ETA (Enerty To Accuracy). Please refer to our paper for the complete workload table.
[^5]: It is possible to cap the maximum power draw of a GPU using [NVML](https://developer.nvidia.com/nvidia-management-library-nvml).
[^6]: For instance, some production training jobs might have tight deadlines; they probably don't want to trade time for energy savings. On the other hand, exploratory training jobs may have more leeway; it might make sense for them to reduce energy consumption at the cost of longer training time.
[^7]: $\textrm{MaxPower}$ is the maximum possible power limit of the GPU. It's just a constant number introduced to equalize the units of the left and right terms to Joules.
[^8]: Since doing this will consume so much time and energy, it may even offset or exceed the energy savings from choosing the optimal knobs if we decide to do it for every future incoming job!
[^9]: Kim Hazelwood, Sarah Bird, David Brooks, Soumith Chintala, Utku Diril, Dmytro Dzhulgakov, Mohamed Fawzy, Bill Jia, Yangqing Jia, Aditya Kalro, et al. Applied machine learning at facebook: A datacenter infrastructure perspective. In 2018 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 620–629. IEEE, 2018.

---

## Research reproducibility

We have our trace-driven simulator open-sourced [here](https://github.com/ml-energy/zeus/tree/master/examples/research_reproducibility/zeus_nsdi23){.external} with instructions.

### Extending the Zeus simulator

Users can implement custom policies that optimize batch size and power limit, and plug it into the Zeus simulator.
We have training and energy traces for 6 different DNNs and 4 different NVIDIA GPU microarchitectures [here](https://github.com/ml-energy/zeus/tree/master/examples/research_reproducibility/zeus_nsdi23/trace){.external}, which the simulator runs with.

Zeus defines two abstract classes [`BatchSizeOptimizer`][zeus._legacy.policy.BatchSizeOptimizer] and [`PowerLimitOptimizer`][zeus._legacy.policy.PowerLimitOptimizer] in [`zeus._legacy.policy.interface`][zeus._legacy.policy.interface].
Each class optimizes the batch size and power limit of a recurring training job respectively.
As in our paper, the batch size optimizer is first invoked to decide which batch size to use, and then the power limit optimizer is invoked with both the job and the batch size chosen to decide which power limit to use.
You can find examples of policy implementations in [`zeus._legacy.policy.optimizer`][zeus._legacy.policy.optimizer].

The Zeus simulator ([`Simulator`][zeus._legacy.simulate.Simulator]) accepts one [`BatchSizeOptimizer`][zeus._legacy.policy.BatchSizeOptimizer] and [`PowerLimitOptimizer`][zeus._legacy.policy.PowerLimitOptimizer] in its constructor.
A full-example can be found [here](https://github.com/ml-energy/zeus/tree/master/examples/research_reproducibility/zeus_nsdi23/){.external}.
