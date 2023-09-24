# Extending Zeus

Users can implement custom policies to optimize batch size and power limits, and plug it into Zeus.

## Interfaces

Zeus defines two abstract classes [`BatchSizeOptimizer`][zeus.policy.BatchSizeOptimizer] and [`PowerLimitOptimizer`][zeus.policy.PowerLimitOptimizer] in [`zeus.policy.interface`][zeus.policy.interface].
Each class optimizes the batch size and power limit of a recurring training job respectively.
As in our paper, the batch size optimizer is first invoked to decide which batch size to use, and then the power limit optimizer is invoked with both the job and the batch size chosen to decide which power limit to use.

You can find examples of policy implementations in [`zeus.policy.optimizer`][zeus.policy.optimizer].


## Plugging it into Zeus

There are two ways to run Zeus: trace-driven and end-to-end.

### Trace-driven Zeus

The Zeus simulator ([`Simulator`][zeus.simulate.Simulator]) accepts one [`BatchSizeOptimizer`][zeus.policy.BatchSizeOptimizer] and [`PowerLimitOptimizer`][zeus.policy.PowerLimitOptimizer] in its constructor.
A full-example can be found in [`examples/trace_driven`](https://github.com/ml-energy/zeus/tree/master/examples/trace_driven/).

### End-to-end Zeus

There are two central components in end-to-end Zeus: [`ZeusMaster`][zeus.run.ZeusMaster] and [`ZeusDataLoader`][zeus.run.ZeusDataLoader].
The former takes charge of driving the entire optimization over recurring jobs, and accepts an instance of [`BatchSizeOptimizer`][zeus.policy.BatchSizeOptimizer] in its constructor.
The latter takes charge of JIT-profiling power in the background, determining the optimal power limit, and setting it.
Hence, the functionality of [`JITPowerLimitOptimizer`][zeus.policy.optimizer.JITPowerLimitOptimizer] is already tightly integrated into `ZeusDataLoader`.
Users will have to implement their own [`ZeusDataLoader`][zeus.run.ZeusDataLoader] in order to test another [`PowerLimitOptimizer`][zeus.policy.PowerLimitOptimizer] policy.
