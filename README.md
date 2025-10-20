<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_light.svg">
  <img alt="Zeus logo" width="55%" src="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_light.svg">
</picture>
<h1>Deep Learning Energy Measurement and Optimization</h1>

[![Slack workspace](https://badgen.net/badge/icon/Join%20workspace/b31b1b?icon=slack&label=Slack)](https://join.slack.com/t/zeus-ml/shared_invite/zt-36fl1m7qa-Ihky6FbfxLtobx40hMj3VA)
[![Docker Hub](https://badgen.net/docker/pulls/symbioticlab/zeus?icon=docker&label=Docker%20pulls)](https://hub.docker.com/r/symbioticlab/zeus)
[![Homepage](https://custom-icon-badges.demolab.com/badge/Homepage-ml.energy-23d175.svg?logo=home&logoColor=white&logoSource=feather)](https://ml.energy/zeus)
[![Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/ml-energy/zeus?logo=law)](/LICENSE)
</div>

---
**Project News** ⚡ 

- \[2025/05\] We shared our experience and design philosophy for the [ML.ENERGY leaderboard](https://ml.energy/leaderboard) in [this paper](https://arxiv.org/abs/2505.06371).
- \[2025/05\] Zeus now supports CPU, DRAM, AMD GPU, Apple Silicon, and NVIDIA Jetson platform energy measurement!
- \[2024/11\] Perseus, an optimizer for large model training, appeared at SOSP'24! [Paper](https://dl.acm.org/doi/10.1145/3694715.3695970) | [Blog](https://ml.energy/zeus/research_overview/perseus) | [Optimizer](https://ml.energy/zeus/optimize/pipeline_frequency_optimizer)
- \[2024/05\] Zeus is now a PyTorch ecosystem project. Read the PyTorch blog post [here](https://pytorch.org/blog/zeus/)!
- \[2024/02\] Zeus was selected as a [2024 Mozilla Technology Fund awardee](https://foundation.mozilla.org/en/blog/open-source-AI-for-environmental-justice/)!
---

Zeus is a library for (1) [**measuring**](https://ml.energy/zeus/measure) the energy consumption of Deep Learning workloads and (2) [**optimizing**](https://ml.energy/zeus/optimize) their energy consumption.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).

## Repository Organization

```
zeus/
├── zeus/             # ⚡ Zeus Python package
│  ├── monitor/       #    - Energy and power measurement (programmatic & CLI)
│  ├── optimizer/     #    - Collection of time and energy optimizers
│  ├── device/        #    - Abstraction layer over CPU and GPU devices
│  ├── utils/         #    - Utility functions and classes
│  ├── _legacy/       #    - Legacy code to keep our research papers reproducible
│  ├── metric.py      #    - Prometheus metric export support
│  ├── show_env.py    #    - Installation & device detection verification script
│  └── callback.py    #    - Base class for callbacks during training
│
├── zeusd             # 🌩️ Zeus daemon
│
├── docker/           # 🐳 Dockerfiles and Docker Compose files
│
└── examples/         # 🛠️ Zeus usage examples
```

## Getting Started

Please refer to our [Getting Started](https://ml.energy/zeus/getting_started) page.
After that, you might look at

- [Measuring Energy](https://ml.energy/zeus/measure)
- [Optimizing Energy](https://ml.energy/zeus/optimize)

### Docker image

We provide a Docker image fully equipped with all dependencies and environments.
Refer to our [Docker Hub repository](https://hub.docker.com/r/mlenergy/zeus) and [`Dockerfile`](docker/Dockerfile).

### Examples

We provide working examples for integrating and running Zeus in the [`examples/`](/examples) directory.

## Research

Zeus is rooted on multiple research papers.
Even more research is ongoing, and Zeus will continue to expand and get better at what it's doing.

1. Zeus (NSDI 23): [Paper](https://www.usenix.org/conference/nsdi23/presentation/you) | [Blog](https://ml.energy/zeus/research_overview/zeus) | [Slides](https://www.usenix.org/system/files/nsdi23_slides_chung.pdf)
1. Chase (ICLR Workshop 23): [Paper](https://arxiv.org/abs/2303.02508)
1. Perseus (SOSP 24): [Paper](https://arxiv.org/abs/2312.06902) | [Blog](https://ml.energy/zeus/research_overview/perseus) | [Slides](https://jaewonchung.me/pdf.js/web/viewer.html?file=/assets/attachments/pubs/Perseus_slides.pdf#pagemode=none)
1. The ML.ENERGY Benchmark (NeurIPS 25 D&B Spotlight): [Paper](https://arxiv.org/abs/2505.06371) | [Repository](https://github.com/ml-energy/benchmark)

If you find Zeus relevant to your research, please consider citing:

```bibtex
@inproceedings{zeus-nsdi23,
    title     = {Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training},
    author    = {Jie You and Jae-Won Chung and Mosharaf Chowdhury},
    booktitle = {USENIX NSDI},
    year      = {2023}
}
```

## Other Resources

1. Energy-Efficient Deep Learning with PyTorch and Zeus (PyTorch conference 2023): [Recording](https://youtu.be/veM3x9Lhw2A) | [Slides](https://ml.energy/assets/attachments/pytorch_conf_2023_slides.pdf)

## Contact

Jae-Won Chung (jwnchung@umich.edu)
