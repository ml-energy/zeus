<div align="center">
<h1>Zeus: Understanding and Optimizing<br>GPU Energy Consumption of DNN Training</h1>
</div>

You can read the entire paper here: [https://arxiv.org/abs/2208.xxxx](https://arxiv.org/abs/2208.xxxx)

{==

## Abstract

Training deep neural networks (DNNs) is becoming more and more resource- and energy-intensive every year. Unfortunately, existing works primarily focus on optimizing DNN training for faster completion, often without considering the impact on energy efficiency.

In this paper, we observe that common practices to improve training performance can often lead to inefficient energy usage. More importantly, we demonstrate that there is a tradeoff between energy consumption and performance optimization. To this end, we propose an optimization framework, Zeus, to navigate this tradeoff by automatically finding optimal job- and GPU-level configurations for recurring DNN training jobs. Zeus uses an online exploration-exploitation approach in conjunction with just-in-time energy profiling, averting the need for expensive offline measurements, while adapting to data drifts over time. Our evaluation shows that Zeus can improve the energy efficiency of DNN training by 15.3%--75.8% for diverse workloads.

==}


