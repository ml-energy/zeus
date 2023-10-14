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
    Zeus exceeded 100k+ pulls on [Docker Hub](https://hub.docker.com/r/mlenergy/zeus)!
    The Zeus team is always happy to chat with Zeus users and help out.
    Reach out to us in the [Zeus Slack workspace](https://join.slack.com/t/zeus-ml/shared_invite/zt-1najba5mb-WExy7zoNTyaZZfTlUWoLLg).

---
**Project News** ⚡ 

- \[2023/10\] We released Perseus, an energy optimizer for large model training. Get started [here](https://ml.energy/zeus/perseus/)!
- \[2023/09\] We moved to under [`ml-energy`](https://github.com/ml-energy)! Please stay tuned for new exciting projects!
- \[2023/07\] [`ZeusMonitor`][zeus.monitor.ZeusMonitor] was used to profile GPU time and energy consumption for the [ML.ENERGY leaderboard & Colosseum](https://ml.energy/leaderboard).
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

### CLI power and energy monitor

```console
$ python -m zeus.monitor power
[2023-08-22 22:39:59,787] [PowerMonitor](power.py:134) Monitoring power usage of GPUs [0, 1, 2, 3]
2023-08-22 22:40:00.800576
{'GPU0': 66.176, 'GPU1': 68.792, 'GPU2': 66.898, 'GPU3': 67.53}
2023-08-22 22:40:01.842590
{'GPU0': 66.078, 'GPU1': 68.595, 'GPU2': 66.996, 'GPU3': 67.138}
2023-08-22 22:40:02.845734
{'GPU0': 66.078, 'GPU1': 68.693, 'GPU2': 66.898, 'GPU3': 67.236}
2023-08-22 22:40:03.848818
{'GPU0': 66.177, 'GPU1': 68.675, 'GPU2': 67.094, 'GPU3': 66.926}
^C
Total time (s): 4.421529293060303
Total energy (J):
{'GPU0': 198.52566362297537, 'GPU1': 206.22215216255188, 'GPU2': 201.08565518283845, 'GPU3': 201.79834523367884}
```

```console
$ python -m zeus.monitor energy
[2023-08-22 22:44:45,106] [ZeusMonitor](energy.py:157) Monitoring GPU [0, 1, 2, 3].
[2023-08-22 22:44:46,210] [zeus.util.framework](framework.py:38) PyTorch with CUDA support is available.
[2023-08-22 22:44:46,760] [ZeusMonitor](energy.py:329) Measurement window 'zeus.monitor.energy' started.
^C[2023-08-22 22:44:50,205] [ZeusMonitor](energy.py:329) Measurement window 'zeus.monitor.energy' ended.
Total energy (J):
Measurement(time=3.4480526447296143, energy={0: 224.2969999909401, 1: 232.83799999952316, 2: 233.3100000023842, 3: 234.53700000047684})
```

Please refer to our NSDI’23 [paper](https://www.usenix.org/conference/nsdi23/presentation/you) and [slides](https://www.usenix.org/system/files/nsdi23_slides_chung.pdf) for details.
Checkout [Overview](overview/index.md) for a summary.

Zeus is part of [The ML.ENERGY Initiative](https://ml.energy).


## Getting Started

Refer to [Getting Started](getting_started/index.md) for instructions on environment setup, installation, and integration.
We also provide integration [examples](https://github.com/ml-energy/zeus/tree/master/examples) in our GitHub repository.

## Extending Zeus

You can easily implement custom policies for batch size and power limit optimization and plug it into Zeus.

Refer to [Extending Zeus](extend.md) for details.
