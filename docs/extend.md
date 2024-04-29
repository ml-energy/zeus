# Extending Zeus

!!! Warning
    Content in this page pertains to Zeus when it was a research artifact of our NSDI paper.
    We will soon refactor the simulator and replace this page with something along the lines of "How to reproduce our research results."
    Track this issue [here](https://github.com/ml-energy/zeus/issues/38).

Users can implement custom policies to optimize batch size and power limits, and plug it into Zeus.

## Interfaces

Zeus defines two abstract classes [`BatchSizeOptimizer`][zeus._legacy.policy.BatchSizeOptimizer] and [`PowerLimitOptimizer`][zeus._legacy.policy.PowerLimitOptimizer] in [`zeus._legacy.policy.interface`][zeus._legacy.policy.interface].
Each class optimizes the batch size and power limit of a recurring training job respectively.
As in our paper, the batch size optimizer is first invoked to decide which batch size to use, and then the power limit optimizer is invoked with both the job and the batch size chosen to decide which power limit to use.

You can find examples of policy implementations in [`zeus._legacy.policy.optimizer`][zeus._legacy.policy.optimizer].

## Plugging it into Zeus

The Zeus simulator ([`Simulator`][zeus._legacy.simulate.Simulator]) accepts one [`BatchSizeOptimizer`][zeus._legacy.policy.BatchSizeOptimizer] and [`PowerLimitOptimizer`][zeus._legacy.policy.PowerLimitOptimizer] in its constructor.
A full-example can be found in [`examples/trace_driven`](https://github.com/ml-energy/zeus/tree/master/examples/trace_driven/).
