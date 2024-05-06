<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_light.svg">
  <img alt="Zeus logo" width="55%" src="https://raw.githubusercontent.com/ml-energy/zeus/master/docs/assets/img/logo_light.svg">
</picture>
<h1>Deep Learning Energy Measurement and Optimization</h1>

[![Slack workspace](https://badgen.net/badge/icon/Join%20workspace/b31b1b?icon=slack&label=Slack)](https://join.slack.com/t/zeus-ml/shared_invite/zt-1najba5mb-WExy7zoNTyaZZfTlUWoLLg)
[![Docker Hub](https://badgen.net/docker/pulls/symbioticlab/zeus?icon=docker&label=Docker%20pulls)](https://hub.docker.com/r/symbioticlab/zeus)
[![Homepage](https://custom-icon-badges.demolab.com/badge/Homepage-ml.energy-23d175.svg?logo=home&logoColor=white&logoSource=feather)](https://ml.energy/zeus)
[![Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/ml-energy/zeus?logo=law)](/LICENSE)
</div>

---
**Project News** ⚡ 

- \[2024/02\] Zeus was selected as a [2024 Mozilla Technology Fund awardee](https://foundation.mozilla.org/en/blog/open-source-AI-for-environmental-justice/)!
- \[2023/12\] We released Perseus, an energy optimizer for large model training: [Preprint](https://arxiv.org/abs/2312.06902) | [Blog](https://ml.energy/zeus/research_overview/perseus) | [Optimizer](https://ml.energy/zeus/optimize/pipeline_frequency_optimizer)
- \[2023/07\] We used the [`ZeusMonitor`](https://ml.energy/zeus/reference/monitor/energy/#zeus.monitor.energy.ZeusMonitor) to profile GPU time and energy consumption for the [ML.ENERGY leaderboard & Colosseum](https://ml.energy/leaderboard).
---

Zeus is a library for (1) [**measuring**](https://ml.energy/zeus/measure) the energy consumption of Deep Learning workloads and (2) [**optimizing**](https://ml.energy/zeus/optimize) their energy consumption.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).

## Repository Organization

```
.
├── zeus/                # ⚡ Zeus Python package
│   ├── optimizer/       #    - A collection of optimizers for time and energy
│   ├── monitor/         #    - Programmatic power and energy measurement tools
│   ├── utils/           #    - Utility functions and classes
│   ├── _legacy/         #    - Legacy code mostly to keep our papers reproducible
│   ├── device.py        #    - Abstraction layer over compute devices
│   └── callback.py      #    - Base class for callbacks during training
│
├── docker/              # 🐳 Dockerfiles and Docker Compose files
│
├── examples/            # 🛠️ Zeus usage examples
│
├── capriccio/           # 🌊 A drifting sentiment analysis dataset
│
└── trace/               # 🗃️ Training and energy traces for various GPUs and DNNs
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

1. Zeus (2023): [Paper](https://www.usenix.org/conference/nsdi23/presentation/you) | [Blog](https://ml.energy/zeus/research_overview/zeus) | [Slides](https://www.usenix.org/system/files/nsdi23_slides_chung.pdf)
1. Chase (2023): [Paper](https://arxiv.org/abs/2303.02508)
1. Perseus (2023): [Paper](https://arxiv.org/abs/2312.06902) | [Blog](https://ml.energy/zeus/research_overview/perseus)

## Other Resources

1. Energy-Efficient Deep Learning with PyTorch and Zeus (PyTorch conference 2023): [Recording](https://youtu.be/veM3x9Lhw2A) | [Slides](https://ml.energy/assets/attachments/pytorch_conf_2023_slides.pdf)

## Contact

Jae-Won Chung (jwnchung@umich.edu)
