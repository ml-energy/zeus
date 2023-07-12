---
hide:
  - toc
  - navigation
  - footer
---
<div align="center">
<img src="assets/img/logo_dark.svg#only-dark" width="60%" alt="Zeus logo" style="margin-bottom: 1em">
<img src="assets/img/logo_light.svg#only-light" width="60%" alt="Zeus logo" style="margin-bottom: 1em">
<h1>Deep Learning Energy Measurement and Optimization</h1>
</div>

!!! Success "Join the Zeus Slack workspace!"
    Zeus exceeded 100k+ pulls on [Docker Hub](https://hub.docker.com/r/symbioticlab/zeus)!
    The Zeus team is always happy to chat with Zeus users and help out.
    Reach out to us in the [Zeus Slack workspace](https://join.slack.com/t/zeus-ml/shared_invite/zt-1najba5mb-WExy7zoNTyaZZfTlUWoLLg).

---
**Project News** ⚡ 

- \[2023/07\] [`ZeusMonitor`][zeus.monitor.ZeusMonitor] was used to profile GPU time and energy consumption for the [ML.ENERGY leaderboard](https://ml.energy/leaderboard).
- \[2023/03\] [Chase](https://symbioticlab.org/publications/files/chase:ccai23/chase-ccai23.pdf), an automatic carbon optimization framework for DNN training, will appear at ICLR'23 workshop.
- \[2022/11\] [Carbon-Aware Zeus](https://taikai.network/gsf/hackathons/carbonhack22/projects/cl95qxjpa70555701uhg96r0ek6/idea) won the **second overall best solution award** at Carbon Hack 22.
---

Zeus is a framework for (1) measuring GPU energy consumption and (2) optimizing energy and time for DNN training.

### Measuring GPU energy

```python
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0,1,2,3])

monitor.begin_window("heavy computation")
# Four GPUs consuming energy like crazy!
measurement = monitor.end_window("heavy computation")

print(f"Energy: {measurement.total_energy} J")
print(f"Time  : {measurement.time} s")
```

### Finding the optimal GPU power limit

Zeus silently profiles different power limits during training and converges to the optimal one.

```python
from zeus.monitor import ZeusMonitor
from zeus.optimizer import GlobalPowerLimitOptimizer

# Data parallel training with four GPUs
monitor = ZeusMonitor(gpu_indices=[0,1,2,3])
plo = GlobalPowerLimitOptimizer(monitor)

plo.on_epoch_begin()

for x, y in train_dataloader:
    plo.on_step_begin()
    # Learn from x and y!
    plo.on_step_end()

plo.on_epoch_end()
```

Please refer to our NSDI’23 [paper](https://www.usenix.org/conference/nsdi23/presentation/you) and [slides](https://www.usenix.org/system/files/nsdi23_slides_chung.pdf) for details.
Checkout [Overview](overview/index.md) for a summary.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).


## Getting Started

Refer to [Getting Started](getting_started/index.md) for instructions on environment setup, installation, and integration.
We also provide integration [examples](https://github.com/SymbioticLab/Zeus/tree/master/examples) in our GitHub repository.

## Extending Zeus

You can easily implement custom policies for batch size and power limit optimization and plug it into Zeus.

Refer to [Extending Zeus](extend.md) for details.
