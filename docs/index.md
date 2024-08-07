---
description: "An open-source framework for Deep Learning energy measurement and optimization"
hide:
  - toc
  - navigation
  - footer
---
<div align="center">
<img src="assets/img/logo_dark.svg#only-dark" width="45%" alt="Zeus logo" style="margin-bottom: 1em">
<img src="assets/img/logo_light.svg#only-light" width="45%" alt="Zeus logo" style="margin-bottom: 1em">
<h1>Deep Learning Energy Measurement and Optimization</h1>
</div>

---
**Project News** ⚡ 

- \[2024/08\] Perseus, our optimizer for large model training, was accepted to SOSP'24! [Preprint](https://arxiv.org/abs/2312.06902){.external} | [Blog](research_overview/perseus.md) | [Optimizer](optimize/pipeline_frequency_optimizer.md)
- \[2024/07\] Added AMD GPU, CPU, and DRAM energy measurement support, and preliminary JAX support!
- \[2024/05\] Zeus is now a PyTorch ecosystem project. Read the PyTorch blog post [here](https://pytorch.org/blog/zeus/){.external}!
- \[2024/02\] Zeus was selected as a [2024 Mozilla Technology Fund awardee](https://foundation.mozilla.org/en/blog/open-source-AI-for-environmental-justice/){.external}!
- \[2023/07\] We used the [`ZeusMonitor`][zeus.monitor.ZeusMonitor] to profile GPU time and energy consumption for the [ML.ENERGY leaderboard & Colosseum](https://ml.energy/leaderboard){.external}.
---

Zeus is a library for (1) [**measuring**](measure/index.md) the energy consumption of Deep Learning workloads and (2) [**optimizing**](optimize/index.md) their energy consumption.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).

## Documentation Organization

- [Getting Started](getting_started/index.md): Instructions on installation and setup.
- [Measuring Energy](measure/index.md): How to measure time and energy programmatically and on the command line.
- [Optimizing Energy](optimize/index.md): How to optimize energy.
- [Research Overview](research_overview/index.md): Overview of the research papers Zeus is rooted on.
- [Source Code Reference](reference/index.md): Auto-generated source code reference for the entire codebase.

We also provide usage [examples](https://github.com/ml-energy/zeus/tree/master/examples) in our GitHub repository.

---

If you find Zeus relevant to your research, please consider citing:

```bibtex
@inproceedings{zeus-nsdi23,
    title     = {Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training},
    author    = {Jie You and Jae-Won Chung and Mosharaf Chowdhury},
    booktitle = {USENIX NSDI},
    year      = {2023}
}
```

## Contact

Jae-Won Chung (jwnchung@umich.edu)
