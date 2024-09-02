Artifact release for the paper "Perseus: Reducing Energy Bloat in Large Model Training."

## Videos

I created an [introduction video](https://youtu.be/mIoBT9Bg08w).
It covers the paper, system components, and miscellaneous details regarding evaluation, at a high-level.

I also created a [screencast](https://umich.zoom.us/rec/share/PjgAepU0oAHymX7UrlKA7rQNMDOSkfx8DD8E8uioSWRPHhIyR8wuQrJNjFFBVVE-.YyJcX9VtaOM9Q5gB) of me performing all the steps in this document myself: On a node with four A40 GPUs, I go from scratch to getting energy savings numbers for GPT-3 Large.
The Zoom recording link above has synchronized transcripts which would help you easily skip over long experiment runs during which I don't say anything, but if it somehow doesn't work, please try [this YouTube link](https://youtu.be/dVc1TdryPwA).
Reading this README and the paper before the screencast would be helpful in understanding what's happening, although I still explain as I run things.

## Artifact organization

The code artifact has three pieces: Perseus (control plane), Merak (training system integrated with Perseus), and Lowtime (GPU frequency planner).

```
 ./
├──  perseus/               # Perseus server and client
├──  merak/                 # Training framework integrated with Perseus
├──  lowtime/               # Optimizer that produces GPU frequency plans
├──  Dockerfile             # Unified Dockerfile for all three components above
│
├──  sosp24_data/           # Experiment data for paper reproduction
├──  evaluation.ipynb       # Jupyter notebook to reproduce evaluation figures and tables
├──  plot.py                # Plotting utility used by the notebook
└──  requirements.txt       # Python dependencies for the notebook
```

> [!IMPORTANT]
> Perseus and Lowtime are our creation and are part of the paper. Merak is *not* our creation.
>
> As such, when evaluating the quality of code, only `perseus/` and `lowtime/` should be considered.

## Reproducing evaluation figures and tables

All the experiment data necessary to reproduce the figures and tables in the paper's evaluation section (Section 6) is included in `sosp24_data`.
Results were generated in three different kinds of environments:

1. **A40 nodes**: 1x AMD EPYC 7513 CPU, 512 GB DRAM, 4x NVIDIA A40-48G GPUs, RHEL 8.7
1. **A100 nodes**: 2x Intel Xeon Platinum 8380 CPUs, 512 GB DRAM, 4x NVIDIA A100-80G PCIe GPUs, Ubuntu 22.04
1. **A100 SXM nodes**: 2x AMD EPYC 7763 CPUs, 512 GB DRAM, 4x NVIDIA A100-80G SXM GPUs, Ubuntu 22.04

A100 SXM nodes were only used for the *Emulation* part of our evaluation. In *Experiment* evaluations, all results are explicitly marked as A40 or A100.

> [!NOTE]
> While the artifact was primarily tested on the three environments above, we expect it to run well on other Linux-based platforms with datacenter NVIDIA GPUs.

```sh
# Clone the repository (Assuming you have set up SSH auth)
git clone git@github.com:ml-energy/zeus.git -b kronos
cd zeus

# Create a new virtual environment. Example is with conda.
# https://docs.anaconda.com/miniconda/miniconda-install/
conda create -n perseus python=3.9 -y
conda activate perseus

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

After this, you can step through `evaluation.ipynb`, which will reproduce the paper's evaluation (Section 6) figures and tables.
Especially, our *Emulation* evaluations are all done in this notebook.

## Running Perseus

This section describes how to run Perseus on real GPUs.

### System components

- **Perseus** is the control plane that tells the training system which GPU frequencies to use for each forward and backward computation.
- **Merak** is the actual training framework that runs training on GPUs. It communicates with the Perseus server via the Perseus client integrated into the training engine. At startup, all ranks register themselves to the Perseus server. Then all ranks repeat: (1) Fetch the next GPU frequency schedule to run, (2) run the schedule for `num_prof_steps` iterations, and (3) report time and energy consumption (of each rank) to the server.
- **Lowtime** is the planner that determines GPU frequencies based on the graph cut-based algorithm described in Section 4. It takes a *pipeline stage time/energy profile* as input and generates *a set of GPU frequency plans* that form the iteration time-energy Pareto frontier.

### Overall experiment workflow

We'll first give an overview of how the experiment workflow looks like. Following sections will provide detailed instructions including commands.

1. **Obtain the pipeline stage time/energy profile**
   - **What**: We profile the time and energy consumption of each stage's forward and backward instruction *during* real training.
   - **How**: Start up the Perseus server in *instruction profiler* mode. Start training in *profiling* mode. Wait until the server finishes profiling and signals the training framework to terminate.
   - **Final output**: A single CSV file that contains all profiling data.
   - **Resource estimation**: Profiling requires all GPUs, since it's running real training. Estimated time varies widely depending on training iteration time, how many iterations you profile for each frequency, and the number of GPU frequencies supported by the GPU. For instance, on A40 GPUs running 4-stage pipeline parallelism for GPT-3 Large, one iteration roughly takes a minute and the GPU supports 103 different frequencies. Even if we ignore very low frequencies and profile only 60 high frequencies, profiling each frequency for 5 training iterations will take 5 hours. In general, on both A40 and A100 GPUs, the initial profiling stage typically takes multiple hours if we want to collect enough profiling samples per (stage, instruction, frequency) triplet. In practice, this initial profiling phase is anyway part of real training and may also be shared with future training runs of the same model.
1. **Obtain optimized GPU frequency plans**
   - **What**: With the profiling data, we run our graph cut-based optimization algorithm (Lowtime) to generate a set of optimized GPU frequency plans on the iteration time-energy Pareto frontier.
   - **How**: Run the Lowtime script (`lowtime/scripts/perseus.py`) with the profile CSV file.
   - **Final output**: Inside the script's output directory, Lowtime will generate many `freqs_pipeline_%d.py` files, each of which is a GPU frequency plan on the iteration time-energy Pareto frontier.
   - **Resource estimation**: Lowtime is written in pure Python and will require just one CPU and moderate DRAM. The optimization problem we're solving is NP-hard, so even though our approximate algorithm has polynomial runtime, it is not something that will run instantly. Runtime also varies wildly depending on the model, number of GPU frequencies supported, and the number of pipeline stages and microbatches. For instance, for our A100 4-stage pipeline cases, it took 6.5 minutes on average (min 20 second, max 16 minutes). In the worst case, it can take over an hour.
1. **Run the optimized GPU frequency plans**
   - **What**: We run actual training with the optimized GPU frequency plans and measure their time and energy consumption.
   - **How**: Start up the Perseus server in *point solution* mode, which runs all `freqs_pipeline_%d.py` plans in the specified directory in order. Start training in normal mode. Wait until the server runs all plans and signals the training framework to terminate.
   - **Final output**: In its output directory (`perseus/dump` by default), the Perseus server will produce one `%d.prof.json` file for each `freqs_pipeline_%d.py` it ran. The JSON file contains the time and energy consumption measurements of that specific GPU frequency plan.
   - **Resource estimation**: This step will also require all GPUs. Again, estimated time varies widely depending on training iteration time, how many iterations you profile for each plan, and the number of plans you want to run. Lowtime produces hundreds to sometimes thousands of fine-grained plans, so we sample 20 ~ 40 plans in equal time distance and only run those. For instance, for A40 GPUs running 4-stage pipeline parallelism for GPT-3 Large, we may sample 30 plans from the frontier and run 15 iterations for each plan to obtain stable measurements, which will take 7.5 hours. Note that we cannot sample too few plans, as we need a reasonably dense iteration time-energy frontier measurement to be able to pick out the right point given a straggler slowdown factor.

### 0. Environment setup

We provide a unified `Dockerfile` in which all three components can run.

```sh
# In the root of this repository
docker build -t perseus:latest .
```

Start the container in daemon mode.

```sh
# In the root of this repository
docker run -dit \
    --gpus all \
    --name perseus \
    -v /dev/infiniband:/dev/infiniband \
    -v $(pwd):/workspace \
    --cap-add SYS_ADMIN \
    --cap-add IPC_LOCK \
    --net host \
    --ipc host \
    perseus:latest bash
```

> [!NOTE]
> 1. You will want to mount the exact number of GPUs you will use. Below our examples are for 4 GPUs, so if your node has more than 4 GPUs, pass in `--gpus '"device=0,1,2,3"'` instead of `--gpus all`.
> 1. `-v /dev/infiniband:/dev/infiniband` was added in order to utilize Infiniband for internode communication. You may not have this depending on your environment.
> 1. `-v $(pwd):/workspace` assumes you are running the command in the root of the repository. It mounts the entire repository into the container and will make it easier when you need to run the Jupyter notebook outside the container with data that was produced inside the container.

Finally, we recommend that you use `tmux` to open two windows -- one for the Perseus server and another for Merak.

```sh
# Run in each window to open a shell inside the container.
docker exec -it perseus bash
```

### 1. Obtain the pipeline stage time/energy profile

Everything is to be run inside the container.

Start the Perseus server in *instruction profiler mode*.

```sh
cd /workspace/perseus
bash scripts/run_server.sh InstructionProfiler '{}' frequency
```

> [!NOTE]
> The three arguments to `scripts/run_server.sh` are:  
> 1. The name of the Perseus scheduler (i.e., classes defined in `perseus.server.scheduler`)
> 2. A literal JSON string specifying any extra arguments to pass to the scheduler's `__init__` function
> 3. Which GPU power-related knob to change. This is either `'frequency'` or `'power limit'`. The latter is only used by the Zeus baselines.

> [!TIP]
> You can pass in `'{"minimum_power_state": 850}'` instead of `'{}'`, for example, to force frequency profiling to skip all GPU frequencies lower than 850 MHz. This is useful because typically after some mid-level GPU frequency, lower ones consume more time and energy, making them strictly suboptimal.

Then, run training in *profiling* mode. Below is an example for GPT-3 Large. We listed the commands for each model we used in the evaluation [here](merak/examples). This step will take hours, as mentioned above.

```sh
cd /workspace/merak/examples
bash run.sh gpt3-large 4 128 --partition_method uniform_transformer --num_prof_steps 5 --export_timing_csv true
```

The `--export_timing_csv true` part makes the training framework also run time and energy profiling in the background as it runs training.

After the Perseus server runs through all the frequencies, it will signal Merak to automatically terminate. (This will look like a 500 Internal Error from the Perseus server, but this is intentional.)
Just before terminating, Merak will print out its output directory:

```console
Output dir is /workspace/merak/examples/language-modeling/output/runs/DATETIME_HOSTNAME
```

You will find `instructions-%d.csv` files (the start and end timestamps of each forward and backward instructions) in rank `%d` and `time-energy-%d.csv` (a time series of current timestamp and the GPU's accumulated energy consumption counter value) of rank `%d`.

Finally, we can generate the final pipeline stage time/energy profile with this. This step will take only a couple seconds. The example below corresponds to the GPT-3 Large example above.

```sh
cd /workspace/merak
python scripts/generate_profile_csv.py \
    --profile_dir /workspace/merak/examples/language-modeling/output/runs/DATETIME_HOSTNAME \
    --num_microbatches 128 \
    --num_prof_steps 5 \
    --gpu_type A40
```

This will generate `profile.csv` inside the output directory. This is the final stage pipeline time/energy profile.

### 2. Obtain optimized GPU frequency plans

Now, you will run Lowtime to generate GPU frequency plans on the iteration time-energy Pareto frontier. You would typically expect this step to take a couple to tens of minutes, as mentioned above.

```sh
cd /workspace/lowtime
python scripts/perseus.py \
    --inst-profile /workspace/merak/examples/language-modeling/output/runs/DATETIME_HOSTNAME/profile.csv \
    --gpu-type A40 \
    --output-dir results/perseus/a40-gpt3-large \
    --num-mbs 128 \
    --num-stages 4
```

Inside the output directory (`results/perseus/a40-gpt3-large`), Lowtime will generate two notable types of files:

- `freqs_pipeline_%d.py`: Each of these Python files is one GPU frequency plan. The larger the number, the shorter its expected iteration time and the larger its expected iteration energy consumption.
- `pipeline_%d.png`: This visualizes the state of the training pipeline for GPU frequency plan `freqs_pipeline_%d.py`. For large pipelines, generating this file also takes quite a bit of time. You can control how frequently the pipeline is visualized by setting `--plot-interval`.

The number of GPU frequency plans that Lowtime generates is typically in the order of hundreds, or even thousands.
Given that one training iteration takes at least tens of seconds, and we would want to run the same GPU frequency plan multiple times for stable measurements, it is infeasible to run all of them.
Thus, the final step is to sample GPU frequency plans in equal time distance. This step will be instant. The example below corresponds to the GPT-3 Large example above.

```sh
cd /workspace/lowtime
python scripts/sample.py results/perseus/a40-gpt3-large sampled/a40-gpt3-large -n 300
```

`-n` is the distance between each sample.
Since our unit time is 1 ms, `-n 300` amounts to each sampled GPU frequency plan being apart by 0.3 s.
The resulting GPU frequency plan files in `sampled/a40-gpt3-large` are the final output of this step.

> [!NOTE]
> The distance between sampled plans can be adjusted, but please note that we do not want to run too few samples as we need a reasonably dense measurement of the iteration time-energy frontier to be able to derive energy savings with stragglers.

### 3. Run the optimized GPU frequency plans

Now, we measure the time and energy consumption of each sampled GPU frequency plan.
Perseus provides a convenient scheduler that runs every plan inside a specific directory.

```sh
cd /workspace/perseus
bash scripts/run_server.sh PointSolution '{"solution_path": "/workspace/lowtime/sampled/a40-gpt3-large"}' frequency
```

> [!TIP]
> The `solution_path` argument for the `PointSolution` scheduler supports both a single `.py` file and a single directory.
> In the latter case, all `.py` files inside the directory will be run in alphabetically sorted order.

Then, start training. Below is an example for GPT-3 Large. We listed the commands for each model we used in the evaluation [here](merak/examples). This step will take the longest among all steps; hours at the minimum, as mentioned above.

```sh
cd /workspace/merak/examples
bash run.sh gpt3-large 4 128 --partition_method uniform_transformer --num_prof_steps 5
```

After the Perseus server runs all the plans, it will signal Merak to automatically terminate.
Final measurements of each frequency plan can be found under the server's state dump directory.

```console
$ cd /workspace/perseus/dump
$ ls
YYYY-MM-DD-HH-MM-SS+merak+gpt3-large+uniform_transformer+dp1+pp4+tp1+mbs4+nmb128+PointSolution
```

This is also the directory we have put under `sosp24_data/perseus`, while just removing the date & time stamp in the beginning.
As the next step, please refer to [Inspecting results](#inspecting-results).

## Running baselines

We have two baselines: Zeus and EnvPipe.

### ZeusGlobal and ZeusPerStage

Both Zeus baselines are implemented as a scheduler in the Perseus server.
No pipeline stage time/energy profiling is needed for the Zeus baselines.

**ZeusGlobal**:

```sh
cd /workspace/perseus
bash scripts/run_server.sh ZeusGlobalPowerLimit '{}' 'power limit'
```

**ZeusPerStage**:

```sh
cd /workspace/perseus
bash scripts/run_server.sh ZeusLocalPowerState '{}' 'power limit'
```

> [!NOTE]
> `ZeusLocalPowerState` is the dev-name of the ZeusPerStage baseline. We decided to leave the name as is, because this name is also shown in the baseline data we provide in `sosp24_data` and thought it might be confusing if the names are different.

After starting up the Perseus server, you can run training Merak as usual.
Results will be saved in the Perseus server state dump directory (`/workspace/perseus/dump`).

### EnvPipe

First, run the same pipeline stage time/energy profiling process to obtain `profile.csv`.
Then, run the EnvPipe frequency assignment algorithm.

```sh
cd /workspace/lowtime
python scripts/envpipe.py \
    --inst-profile /workspace/merak/examples/language-modeling/output/runs/DATETIME_HOSTNAME/profile.csv \
    --gpu-type A40 \
    --output-dir results/envpipe/a40-gpt3-large \
    --num-mbs 128 \
    --num-stages 4
```

Due to a limitation of EnvPipe's algorithm, which assumes that the last pipeline stage is always the heaviest and terminates only when the critical path reaches the outer envelope, it may run an infinite loop.
As such, when you see stdout repeating that is nothing more to speed up, you can interrupt the script with CTRL-C.

When `envpipe.py` terminates, the final GPU frequency plan (`freqs_pipeline_%d.py` with the largest number) is the GPU frequency schedule of EnvPipe.

Just like for Perseus, you can start the Perseus server with the `PointSolution` scheduler and point the server to that `freqs_pipeline_%d.py` file, and run training with Merak normally to obtain measurements.

## Inspecting results

### Reusing the Jupyter notebook

When you run experiments and produce your own data, you can create a new data directory that replicates the structure of `sosp24_data` and set the `plot.DATA` variable in `evaluation.ipynb` to that directory to plot and inspect the results.
Especially, Perseus results are saved in directory path:

```
sosp24_data/perseus/A40/dp1+pp8+tp1/merak+gpt3-2.7b+uniform_transformer+dp1+pp8+tp1+mbs4+nmb256+PointSolution
^^^^^^^^^^^         ^^^ ^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  │                  │   │                 │
  │                  │   │                 └─► Experiment name
  │                  │   └─► Parallelization
  │                  └─► GPU type
  └─► Base directory
```

Especially, the experiment name is a `"+".join` of: the name of the training framework (fixed to `merak`), the name of the model, pipeline stage partitioning policy, data parallel degree (`dp`), pipeline parallel degree (`pp`), tensor parallel degree (`tp`), microbatch size (`mbs`), number of microbatches (`nmb`), and the Perseus server's scheduler name (fixed to `PointSolution`).
This is basically how the Perseus server names its output directory for a specific job.
The only thing you need to do is to remove the date & time stamp and move it into the right place under the base data directory.

### Directly inspecting numbers

The `perseus` Python package provides two useful utility functions:

- `perseus.utils.state.load_prof` reads in a `%d.prof.json` file and parses it into a `list[perseus.models.ProfilingResult]` object.
- `perseus.utils.analysis.total_time_and_energy` takes a `list[perseus.models.ProfilingResult]` object and returns its average iteration time and iteration energy.

For instance, you may want to inspect the iteration time slowdown and energy reduction of one of Perseus's plans.
You can run that specific plan by starting the Perseus server with the `PointSolution` scheduler and running training.

```sh
cd perseus
bash scripts/run_server.sh PointSolution '{"solution_path": "./freqs_pipeline_00333.py"}' frequency
```

After two warm-ups with all-max frequencies, the designated GPU frequency plan will run, producing `0.prof.json`, `1.prof.json`, and `2.prof.json` inside `/workspace/perseus/dump`.
You can inspect the first two JSON files to obtain the baseline number of running everything with the maximum frequency, and then the third JSON file for the results of the GPU frequency plan that you have designated with `solution_path`.
