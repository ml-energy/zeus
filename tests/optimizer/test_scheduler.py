"""Unit tests for the zeus.optimizer.pipeline_frequency.server.scheduler module."""

from __future__ import annotations

import os
import shutil
import numpy as np
import pytest

from zeus.optimizer.pipeline_frequency.common import (
    JobInfo,
    RankInfo,
    ProfilingResult,
    PFOServerSettings,
)
from zeus.optimizer.pipeline_frequency.server.scheduler import FrequencySchedule


@pytest.fixture
def dummy_job_info():
    """Dummy JobInfo for testing using minimal fields required."""
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


@pytest.fixture
def dummy_rank_infos():
    """Dummy RankInfo for two ranks; testing using minimal fields required."""
    dummy_available_freqs = [1000, 900, 800, 700]
    # Use plain strings for the pipeline schedule.
    dummy_pipe_schedule = ["forward", "backward"]
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
            )
        )
    return rank_infos


@pytest.fixture
def dummy_pfosettings(tmp_path):
    """PFOServerSettings with dump_data enabled and dump_dir set to a temporary directory."""
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir(exist_ok=True)  # Create the dump directory
    return PFOServerSettings(
        scheduler="InstructionProfiler",  # This will be automatically imported.
        dump_data=True,
        dump_dir=str(dump_dir.resolve()),
    )


@pytest.fixture
def dummy_profiling_result(dummy_rank_infos):
    """
    Dummy ProfilingResult to simulate profiling output.
    Creates dummy breakdown dictionaries: one list per instruction.
    """
    dummy_results = []
    for fs in dummy_rank_infos:
        time_breakdown = {
            "forward": [[np.random.uniform(0.1, 0.2) for _ in range(10)]],
            "backward": [[np.random.uniform(0.2, 0.3) for _ in range(10)]],
        }
        energy_breakdown = {
            "forward": [[np.random.uniform(10, 20) for _ in range(10)]],
            "backward": [[np.random.uniform(20, 30) for _ in range(10)]],
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
    """
    Test the InstructionProfiler scheduler with meaningful assertions:
    - Verifies that warm-up iterations return the maximum frequency.
    - Checks that subsequent iterations yield descending frequency values.
    - Asserts that a CSV file is written.
    """
    # Instantiate the scheduler using the class stored in pfo_settings.
    # Pydantic automatically imports the scheduler class specified by the
    # ZEUS_PFO_SCHEDULER environment variable (or the default in PFOServerSettings).
    scheduler_cls = dummy_pfosettings.scheduler
    scheduler = scheduler_cls(
        dummy_job_info,
        dummy_rank_infos,
        dummy_pfosettings,
        **dummy_pfosettings.scheduler_args,
    )

    # Expected maximum frequency from available_frequencies.
    max_freq = max(dummy_rank_infos[0].available_frequencies)  # 1000 in this case

    # --- Warm-up Iterations ---
    warmup_schedule1 = scheduler.next_schedule()
    for fs in warmup_schedule1:
        for inst, freq in fs.frequencies:
            assert (
                freq == max_freq
            ), f"Expected warm-up frequency {max_freq}, got {freq}"
    scheduler.observe(dummy_profiling_result)

    warmup_schedule2 = scheduler.next_schedule()
    for fs in warmup_schedule2:
        for inst, freq in fs.frequencies:
            assert (
                freq == max_freq
            ), f"Expected warm-up frequency {max_freq}, got {freq}"
    scheduler.observe(dummy_profiling_result)

    # --- Scheduling Iterations ---
    profiling_done = False
    for _ in range(10):
        try:
            schedule = scheduler.next_schedule()
            # Assert that frequencies are either 1000 or 900
            for fs in schedule:
                for inst, freq in fs.frequencies:
                    assert freq in (1000, 900), f"Unexpected frequency {freq}"
            scheduler.observe(dummy_profiling_result)
        except RuntimeError as err:
            if "[profiling-done]" in str(err):
                profiling_done = True
                break
            else:
                raise

    assert profiling_done, "Scheduler did not terminate profiling as expected."

    # --- Verify CSV Output ---
    dump_dir = dummy_pfosettings.dump_dir
    assert os.path.exists(dump_dir), f"Dump directory {dump_dir} does not exist."
    csv_files = [f for f in os.listdir(dump_dir) if f.endswith(".csv")]
    assert len(csv_files) > 0, "No CSV file was written to the dump directory."

    # Clean up the temporary dump directory.
    shutil.rmtree(dump_dir)
