"""Monitor the temperature of GPUs."""

from __future__ import annotations

import bisect
import collections
import logging
import multiprocessing as mp
import weakref
from time import time, sleep
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING

from zeus.device.gpu.common import ZeusGPUNotSupportedError
from zeus.device import get_gpus

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from multiprocessing.context import SpawnProcess

logger = logging.getLogger(__name__)


def _cleanup_temperature_process(
    stop_event: EventClass,
    process: SpawnProcess,
) -> None:
    """Idempotent cleanup function for temperature monitoring process."""
    # Signal the process to stop
    stop_event.set()

    # Wait for the process to complete
    if process.is_alive():
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)


@dataclass
class TemperatureSample:
    """A single temperature measurement sample."""

    timestamp: float
    gpu_index: int
    temperature_c: int


class TemperatureMonitor:
    """Monitor GPU temperature over time.

    This class provides:
    1. Continuous temperature monitoring in a background process
    2. Timeline export with deduplication
    3. Point-in-time temperature queries

    !!! Note
        The current implementation only supports cases where all GPUs are homogeneous
        (i.e., the same model).

    !!! Warning
        Since the monitor spawns child processes, **it should not be instantiated as a global variable**.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        update_period: float = 1.0,
        max_samples_per_gpu: int | None = None,
    ) -> None:
        """Initialize the temperature monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the temperature monitor in seconds.
                Defaults to 1.0 second. Temperature typically doesn't change as
                rapidly as power, so a longer update period is reasonable.
            max_samples_per_gpu: Maximum number of temperature samples to keep per GPU
                in memory. If None (default), unlimited samples are kept.
        """
        if gpu_indices is not None and not gpu_indices:
            raise ValueError("`gpu_indices` must be either `None` or non-empty")

        # Get GPUs
        gpus = get_gpus(ensure_homogeneous=True)

        # Configure GPU indices
        self.gpu_indices = gpu_indices if gpu_indices is not None else list(range(len(gpus)))
        if not self.gpu_indices:
            raise ValueError("At least one GPU index must be specified")
        logger.info("Monitoring temperature of GPUs %s", self.gpu_indices)

        self.update_period = update_period

        # Temperature samples are collected for each device index.
        self.temperature_samples: dict[int, collections.deque[TemperatureSample]] = {}
        for gpu_idx in self.gpu_indices:
            self.temperature_samples[gpu_idx] = collections.deque(maxlen=max_samples_per_gpu)

        # Spawn temperature collector process
        ctx = mp.get_context("spawn")
        self.temperature_queue = ctx.Queue()
        self.temperature_ready_event = ctx.Event()
        self.temperature_stop_event = ctx.Event()
        self.temperature_process = ctx.Process(
            target=_temperature_polling_process,
            kwargs=dict(
                gpu_indices=self.gpu_indices,
                data_queue=self.temperature_queue,
                ready_event=self.temperature_ready_event,
                stop_event=self.temperature_stop_event,
                update_period=update_period,
            ),
            daemon=True,
            name="zeus-temperature-monitor",
        )
        self.temperature_process.start()

        # Cleanup function
        self._finalizer = weakref.finalize(
            self,
            _cleanup_temperature_process,
            self.temperature_stop_event,
            self.temperature_process,
        )

        # Wait for subprocess to signal it's ready
        logger.info("Waiting for temperature monitoring subprocess to be ready...")
        if not self.temperature_ready_event.wait(timeout=10.0):
            logger.warning("Temperature monitor subprocess did not signal ready within timeout")
        logger.info("Temperature monitoring subprocess is ready")

    def stop(self) -> None:
        """Stop the monitoring process."""
        if self._finalizer.alive:
            self._finalizer()

    def _process_temperature_queue_data(self) -> None:
        """Process all pending temperature samples from the queue."""
        if not hasattr(self, "temperature_queue"):
            return

        while True:
            try:
                sample = self.temperature_queue.get_nowait()
                if sample == "STOP":
                    break
                assert isinstance(sample, TemperatureSample)
                self.temperature_samples[sample.gpu_index].append(sample)
            except Empty:
                break

    def get_temperature_timeline(
        self,
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[int, list[tuple[float, int]]]:
        """Get temperature timeline for specific GPU(s).

        Args:
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp)
            end_time: End time filter (unix timestamp)

        Returns:
            Dictionary mapping GPU indices to timeline data.
            Timeline data is list of (timestamp, temperature_celsius) tuples.
        """
        # Process any pending queue data
        self._process_temperature_queue_data()

        # Determine which GPUs to query
        target_gpus = [gpu_index] if gpu_index is not None else self.gpu_indices

        result = {}
        for gpu_idx in target_gpus:
            if gpu_idx not in self.temperature_samples:
                continue

            # Extract timeline from samples
            timeline = []
            for sample in self.temperature_samples[gpu_idx]:
                # Apply time filters
                if start_time is not None and sample.timestamp < start_time:
                    continue
                if end_time is not None and sample.timestamp > end_time:
                    continue

                timeline.append((sample.timestamp, sample.temperature_c))

            # Sort by timestamp
            timeline.sort(key=lambda x: x[0])
            result[gpu_idx] = timeline

        return result

    def get_temperature(self, time: float | None = None) -> dict[int, int] | None:
        """Get the GPU temperature at a specific time point.

        Args:
            time: Time point to get the temperature at. If None, get the temperature
                at the last recorded time point.

        Returns:
            A dictionary mapping GPU indices to the temperature of the GPU at the
            specified time point. If there are no temperature readings, return None.
        """
        # Process any pending queue data
        self._process_temperature_queue_data()

        result = {}
        for gpu_idx in self.gpu_indices:
            samples = self.temperature_samples[gpu_idx]
            if not samples:
                return None

            if time is None:
                # Get the most recent sample
                latest_sample = samples[-1]
                result[gpu_idx] = latest_sample.temperature_c
            else:
                # Find the closest sample to the requested time using bisect
                timestamps = [sample.timestamp for sample in samples]
                pos = bisect.bisect_left(timestamps, time)

                if pos == 0:
                    closest_sample = samples[0]
                elif pos == len(samples):
                    closest_sample = samples[-1]
                else:
                    # Check the closest sample before and after the requested time
                    before = samples[pos - 1]
                    after = samples[pos]
                    closest_sample = before if time - before.timestamp <= after.timestamp - time else after
                result[gpu_idx] = closest_sample.temperature_c

        return result


def _temperature_polling_process(
    gpu_indices: list[int],
    data_queue: mp.Queue,
    ready_event: EventClass,
    stop_event: EventClass,
    update_period: float,
) -> None:
    """Polling process for GPU temperature with deduplication."""
    try:
        # Get GPUs
        gpus = get_gpus()

        # Track previous temperature values for deduplication
        prev_temperature: dict[int, int] = {}

        # Signal that this process is ready to start monitoring
        ready_event.set()

        # Start polling loop
        while not stop_event.is_set():
            timestamp = time()

            for gpu_index in gpu_indices:
                try:
                    temperature_c = gpus.get_gpu_temperature(gpu_index)

                    # Deduplication: only send if temperature changed
                    if gpu_index in prev_temperature and prev_temperature[gpu_index] == temperature_c:
                        continue

                    prev_temperature[gpu_index] = temperature_c

                    # Create and send temperature sample
                    sample = TemperatureSample(
                        timestamp=timestamp,
                        gpu_index=gpu_index,
                        temperature_c=temperature_c,
                    )

                    data_queue.put(sample)
                except ZeusGPUNotSupportedError as e:
                    logger.warning(
                        "GPU %d temperature reading not supported: %s",
                        gpu_index,
                        e,
                    )
                    # Don't keep trying if it's not supported
                    break
                except Exception as e:
                    logger.exception(
                        "Error polling temperature for GPU %d: %s",
                        gpu_index,
                        e,
                    )
                    raise

            # Sleep for the remaining time
            elapsed = time() - timestamp
            sleep_time = update_period - elapsed
            if sleep_time > 0:
                sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Exiting temperature polling process due to error: %s",
            e,
        )
        raise e
    finally:
        # Send stop signal
        data_queue.put("STOP")
