<h1 align="center">Lowtime: A Time-Cost Tradeoff Problem Solver</h1>

Lowtime is a library for solving the [time-cost tradeoff problem](https://link.springer.com/chapter/10.1007/978-3-030-61423-2_5).

## What do I use `lowtime` for?

Say you want to execute a **DAG of operations or tasks**, and each operation has multiple execution options each with **different time and cost**.

Given the definition of the DAG, `lowtime` will find the **complete time-cost Pareto frontier** of the entire DAG.

You define *cost*. Any positive floating point number that is at odds with time!

> [!NOTE]
> The actual open-source version of Lowtime is [here](https://github.com/ml-energy/lowtime).
> This repository was created when we were actually doing research.
> It contains all the debug outputs that are useful for visualization, so we're using this version for artifact evaluation.
