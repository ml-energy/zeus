"""
Unit tests for the zeus.optimizer.pipeline_frequency.server.scheduler module
"""

import os
import shutil
import numpy as np
import pytest

from zeus.optimizer.pipeline_frequency.common import (
    JobInfo,
    RankInfo,
    ProfilingResult,
    PipeInstruction,
    PFOServerSettings,
)
from zeus.optimizer.pipeline_frequency.server.scheduler import (
    get_scheduler,
    FrequencySchedule,
)


#  Dummy JobInfo using minimal fields required.
@pytest.fixture
def dummy_job_info(tmp_path):
    # Set the dump directory to the temporary path
    return JobInfo(
        job_id="",
        pp_degree=2,
        dp_degree=1,
        tp_degree=1,
        world_size=2,
        job_metadata="pytest_run",
        framework="TestFramework",
        model_name="TestModel",
        partition_method="TestPartition",
        microbatch_size=1,
        num_microbatches=1,
    )


#  Dummy RankInfo for two ranks.
@pytest.fixture
def dummy_rank_infos():
    dummy_available_freqs = [1000, 900, 800, 700]
    dummy_pipe_schedule = [PipeInstruction.FORWARD, PipeInstruction.BACKWARD]
    rank_infos = []
    for i in range(2):
        rank_infos.append(
            RankInfo(
                rank=i,
                dp_rank=0,
                pp_rank=i,  # assume each rank is its own pipeline stage
                tp_rank=0,
                available_frequencies=dummy_available_freqs,
                pipe_schedule=dummy_pipe_schedule,
                power_state_range=dummy_available_freqs.copy(),
            )
        )
    return rank_infos


#  PFOServerSettings with dump_data enabled and dump_dir set to a temporary directory.
@pytest.fixture
def dummy_pfosettings(tmp_path):
    return PFOServerSettings(
        scheduler="InstructionProfiler",  # use InstructionProfiler for this test
        dump_data=True,
        dump_dir=str(tmp_path / "dump"),
        # In case scheduler_args are required by the default scheduler,
        # you can add: scheduler_args={"solution_path": "frequencies.py"}
    )


#  Dummy ProfilingResult to simulate profiling output.
@pytest.fixture
def dummy_profiling_result(dummy_rank_infos):
    dummy_results = []
    for fs in dummy_rank_infos:
        # Create dummy breakdown dictionaries: one list per instruction.
        time_breakdown = {
            PipeInstruction.FORWARD.value: [
                [np.random.uniform(0.1, 0.2) for _ in range(5)]
            ],
            PipeInstruction.BACKWARD.value: [
                [np.random.uniform(0.2, 0.3) for _ in range(5)]
            ],
        }
        energy_breakdown = {
            PipeInstruction.FORWARD.value: [
                [np.random.uniform(10, 20) for _ in range(5)]
            ],
            PipeInstruction.BACKWARD.value: [
                [np.random.uniform(20, 30) for _ in range(5)]
            ],
        }
        dummy_results.append(
            ProfilingResult(
                rank=fs.rank,
                iter_time=[np.random.uniform(0.5, 1.0)],
                iter_energy=[np.random.uniform(100, 200)],
                time_breakdown=time_breakdown,
                energy_breakdown=energy_breakdown,
            )
        )
    return dummy_results


def test_instruction_profiler_scheduler(
    dummy_job_info, dummy_rank_infos, dummy_pfosettings, dummy_profiling_result
):
    # Instantiate scheduler using the factory function.
    scheduler = get_scheduler(dummy_job_info, dummy_rank_infos, dummy_pfosettings)

    # Run a few iterations of the scheduler.
    iteration = 0
    profiling_done = False
    while True:
        try:
            schedule = scheduler.next_schedule()
            # Basic assertions: schedule must be a list with FrequencySchedule objects.
            assert isinstance(schedule, list)
            for fs in schedule:
                # We expect each schedule to have a frequencies attribute.
                assert hasattr(fs, "frequencies")
            # Feed the dummy profiling results into the scheduler.
            scheduler.observe(dummy_profiling_result)
            iteration += 1
            # Limit iterations in case the scheduler doesn't finish
            if iteration > 5:
                break
        except RuntimeError as err:
            # When profiling is complete, InstructionProfiler raises a RuntimeError with a tag.
            assert "[profiling-done]" in str(err)
            profiling_done = True
            break

    assert profiling_done, "Scheduler did not terminate profiling as expected."

    # Verify that a CSV file was written to the dump directory.
    dump_dir = dummy_pfosettings.dump_dir
    assert os.path.exists(dump_dir), f"Dump directory {dump_dir} does not exist."
    csv_files = [f for f in os.listdir(dump_dir) if f.endswith(".csv")]
    assert len(csv_files) > 0, "No CSV file was written to the dump directory."

    # Clean up the temporary dump directory.
    shutil.rmtree(dump_dir)
