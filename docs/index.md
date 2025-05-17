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

- \[2025/05\] We shared our experience and design philosophy for the [ML.ENERGY leaderboard](https://ml.energy/leaderboard){.external} in [this paper](https://arxiv.org/abs/2505.06371){.external}.
- \[2025/05\] Zeus now supports CPU, DRAM, AMD GPU, Apple Silicon, and NVIDIA Jetson platform energy measurement!
- \[2024/11\] Perseus, an optimizer for large model training, appeared at SOSP'24! [Paper](https://dl.acm.org/doi/10.1145/3694715.3695970) | [Blog](https://ml.energy/zeus/research_overview/perseus) | [Optimizer](https://ml.energy/zeus/optimize/pipeline_frequency_optimizer)
- \[2024/05\] Zeus is now a PyTorch ecosystem project. Read the PyTorch blog post [here](https://pytorch.org/blog/zeus/){.external}!
- \[2024/02\] Zeus was selected as a [2024 Mozilla Technology Fund awardee](https://foundation.mozilla.org/en/blog/open-source-AI-for-environmental-justice/){.external}!
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
