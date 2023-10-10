# Perseus: Energy Scheduling in Large Model Training

!!! Warning
    Perseus is under active development, and breaking changes may happen, both in Perseus and its integration with Zeus.

## Overview

Perseus finds the iteration time--energy Pareto frontier of large model training.
Users can pick any point on the frontier -- be it minimum time, minimum energy, or something in the middle.

Large model training requires the distribution of work to multiple GPUs.
The core observation of Perseus is that work cannot be perfectly split and balanced across every GPUs; some GPUs have more work to do and some less.
GPUs with smaller amounts of work finish before GPUs with more amounts of work, but ultimately training throughput is bound my GPUs with the most amount of work.
In other words, GPUs with lighter load are running unnecessarily fast and wasting energy (i.e., there is **energy bloat**).

Each pipeline instruction, e.g., forward and backward, can be executed with its own GPU frequency, and each frequency consumes different time and energy.
We call the assignment of each pipeline instruction with GPU frequencies *frequency plan*, and Perseus basically gives you **every Pareto-optimal frequency plan** that you can choose any point on the iteration time--energy Pareto frontier.

## How it's done

Largely there are three steps -- profile, optimize, and execute.

### Profile

In order to run our algorithm, we need 
