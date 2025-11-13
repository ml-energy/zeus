"""Monitor the power usage of GPUs."""

from __future__ import annotations

import bisect
import collections
import logging
import multiprocessing as mp
import weakref
from enum import Enum
from time import time, sleep
from dataclasses import dataclass
from queue import Empty
from typing import Literal, Callable, TYPE_CHECKING

from sklearn.metrics import auc

from zeus.device.gpu.common import ZeusGPUNotSupportedError
from zeus.device import get_gpus

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from multiprocessing.context import SpawnProcess

logger = logging.getLogger(__name__)


def infer_counter_update_period(gpu_indicies: list[int]) -> float:
    """Infer the update period of the GPU power counter.

    GPU power counters can update as slow as 10 Hz depending on the GPU model, so
    there's no need to poll them too faster than that. This function infers the
    update period for each unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.
    """
    gpus = get_gpus()

    # For each unique GPU model, infer the update period.
    update_period = 0.0
    gpu_models_covered = set()
    for index in gpu_indicies:
        if (model := gpus.get_name(index)) not in gpu_models_covered:
            logger.info("Detected %s, inferring GPU power counter update period.", model)
            gpu_models_covered.add(model)
            detected_period = _infer_counter_update_period_single(index)
            logger.info(
                "Counter update period for %s is %.2f s",
                model,
                detected_period,
            )
            update_period = min(update_period, detected_period)

    # Target half the update period to ensure that the counter is enough.
    update_period /= 2.0

    # Anything less than ten times a second is probably too slow.
    if update_period > 0.1:
        logger.warning(
            "Inferred update period (%.2f s) is too long. Using 0.1 s instead.",
            update_period,
        )
        update_period = 0.1
    return update_period


def _infer_counter_update_period_single(gpu_index: int) -> float:
    """Infer the update period of the GPU power counter for a single GPU."""
    gpus = get_gpus()

    # Determine which power measurement method to use
    # Try instant power first, fall back to average power
    power_method: Callable[[int], int] | None = None
    try:
        # Test if instant power is available
        _ = gpus.get_instant_power_usage(gpu_index)
        power_method = gpus.get_instant_power_usage
    except ZeusGPUNotSupportedError:
        try:
            # Fall back to average power
            _ = gpus.get_average_power_usage(gpu_index)
            power_method = gpus.get_average_power_usage
            logger.info(
                "Instant power not available for GPU %d, using average power for counter period inference",
                gpu_index,
            )
        except ZeusGPUNotSupportedError:
            # Neither method available, return conservative default
            logger.warning(
                "Neither instant nor average power available for GPU %d. "
                "Using conservative default update period of 0.1s",
                gpu_index,
            )
            return 0.2  # Will be halved later to 0.1s

    # Collect 1000 samples of the power counter with timestamps.
    time_power_samples: list[tuple[float, int]] = [(0.0, 0) for _ in range(1000)]
    for i in range(len(time_power_samples)):
        time_power_samples[i] = (
            time(),
            power_method(gpu_index),
        )

    # Find the timestamps when the power readings changed.
    time_power_samples = time_power_samples[10:]
    changed_times = []
    prev_power = time_power_samples[0][1]
    for t, p in time_power_samples:
        if p != prev_power:
            changed_times.append(t)
            prev_power = p

    # Compute the minimum time difference between power change timestamps.
    intervals = [time2 - time1 for time1, time2 in zip(changed_times, changed_times[1:])]
    if len(intervals) == 0:
        return 0.1
    return min(intervals)


class PowerDomain(Enum):
    """Power measurement domains with different update characteristics."""

    DEVICE_INSTANT = "device_instant"
    DEVICE_AVERAGE = "device_average"
    MEMORY_AVERAGE = "memory_average"


@dataclass
class PowerSample:
    """A single power measurement sample."""

    timestamp: float
    gpu_index: int
    power_mw: float


def _cleanup_processes(
    stop_events: dict[PowerDomain, EventClass],
    processes: dict[PowerDomain, SpawnProcess],
) -> None:
    """Idempotent cleanup function for power monitoring processes."""
    # Signal all processes to stop
    for event in stop_events.values():
        event.set()

    # Wait for each process to complete
    for process in processes.values():
        if process.is_alive():
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)

    # Clean up dictionaries
    stop_events.clear()
    processes.clear()


class PowerMonitor:
    """Enhanced PowerMonitor with multiple power domains and timeline export.

    This class provides:
    1. Multiple power domains: device instant, device average, and memory average
    2. Timeline export with independent deduplication per domain
    3. Separate processes for each power domain (2-3 processes depending on GPU support)
    4. Backward compatibility with existing PowerMonitor interface

    !!! Note
        The current implementation only supports cases where all GPUs are homegeneous
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
        update_period: float | None = None,
        max_samples_per_gpu: int | None = None,
    ) -> None:
        """Initialize the enhanced power monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the power monitor in seconds. If None,
                infer the update period by max speed polling the power counter for
                each GPU model.
            max_samples_per_gpu: Maximum number of power samples to keep per GPU per domain
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
        logger.info("Monitoring power usage of GPUs %s", self.gpu_indices)

        # Infer update period from GPU instant power, if necessary
        if update_period is None:
            update_period = infer_counter_update_period(self.gpu_indices)
        elif update_period < 0.05:
            logger.warning(
                "An update period of %g might be too fast, which may lead to unexpected "
                "errors (e.g., NotSupported) and/or zero values being returned. "
                "If you see these, consider increasing to >= 0.05.",
                update_period,
            )
        self.update_period = update_period

        # Inter-process communication - separate unbounded queue per domain
        self.data_queues: dict[PowerDomain, mp.Queue] = {}
        self.ready_events: dict[PowerDomain, EventClass] = {}
        self.stop_events: dict[PowerDomain, EventClass] = {}
        self.processes: dict[PowerDomain, SpawnProcess] = {}

        # Determine which domains are supported
        self.supported_domains = self._determine_supported_domains()
        logger.info("Supported power domains: %s", [d.value for d in self.supported_domains])

        # Power samples are collected for each power domain and device index.
        self.samples: dict[PowerDomain, dict[int, collections.deque[PowerSample]]] = {}
        for domain in self.supported_domains:
            self.samples[domain] = {}
            for gpu_idx in self.gpu_indices:
                self.samples[domain][gpu_idx] = collections.deque(maxlen=max_samples_per_gpu)

        # Spawn collector processes for each supported domain
        ctx = mp.get_context("spawn")
        for domain in self.supported_domains:
            self.data_queues[domain] = ctx.Queue()
            self.ready_events[domain] = ctx.Event()
            self.stop_events[domain] = ctx.Event()
            self.processes[domain] = ctx.Process(
                target=_domain_polling_process,
                kwargs=dict(
                    power_domain=domain,
                    gpu_indices=self.gpu_indices,
                    data_queue=self.data_queues[domain],
                    ready_event=self.ready_events[domain],
                    stop_event=self.stop_events[domain],
                    update_period=update_period,
                ),
                daemon=True,
                name=f"zeus-power-monitor-{domain.value}",
            )
        for process in self.processes.values():
            process.start()

        # Cleanup functions
        self._finalizer = weakref.finalize(self, _cleanup_processes, self.stop_events, self.processes)

        # Wait for all subprocesses to signal they're ready
        logger.info("Waiting for all power monitoring subprocesses to be ready...")
        for domain in self.supported_domains:
            if not self.ready_events[domain].wait(timeout=10.0):
                logger.warning(
                    "Power monitor subprocess for %s did not signal ready within timeout",
                    domain.value,
                )
        logger.info("All power monitoring subprocesses are ready")

    def _determine_supported_domains(self) -> list[PowerDomain]:
        """Determine which power domains are supported by the current GPUs."""
        supported = []
        gpus = get_gpus(ensure_homogeneous=True)
        methods = {
            PowerDomain.DEVICE_INSTANT: gpus.get_instant_power_usage,
            PowerDomain.DEVICE_AVERAGE: gpus.get_average_power_usage,
            PowerDomain.MEMORY_AVERAGE: gpus.get_average_memory_power_usage,
        }

        # Just check the first GPU for support, since all GPUs are homogeneous.
        for domain, method in methods.items():
            try:
                _ = method(0)
                supported.append(domain)
                logger.info("Power domain %s is supported", domain.value)
            except ZeusGPUNotSupportedError:
                logger.info("Power domain %s is not supported", domain.value)
            except Exception as e:
                logger.warning(
                    "Unexpected error while checking for %s support on GPU %d: %s",
                    domain.value,
                    self.gpu_indices[0],
                    e,
                )

        return supported

    def stop(self) -> None:
        """Stop all monitoring processes."""
        if self._finalizer.alive:
            self._finalizer()

    def _process_queue_data(self, domain: PowerDomain) -> None:
        """Process all pending samples from a specific domain's queue."""
        if domain not in self.data_queues:
            return

        while True:
            try:
                sample = self.data_queues[domain].get_nowait()
                if sample == "STOP":
                    break
                assert isinstance(sample, PowerSample)
                self.samples[domain][sample.gpu_index].append(sample)
            except Empty:
                break

    def _process_all_queue_data(self) -> None:
        """Process all pending samples from all domain queues."""
        for domain in self.supported_domains:
            self._process_queue_data(domain)

    def get_power_timeline(
        self,
        power_domain: PowerDomain | Literal["device_instant", "device_average", "memory_average"],
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[int, list[tuple[float, float]]]:
        """Get power timeline for specific power domain and GPU(s).

        Args:
            power_domain: Power domain to query
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp from time.time() or similar)
            end_time: End time filter (unix timestamp from time.time() or similar)

        Returns:
            Dictionary mapping GPU indices to timeline data with deduplication.
            Timeline data is list of (timestamp, power_watts) tuples.
        """
        if isinstance(power_domain, str):
            power_domain = PowerDomain(power_domain)

        if power_domain not in self.supported_domains:
            raise ValueError(f"Power domain {power_domain.value} is not supported by the current GPUs.")

        # Process any pending queue data for this domain
        self._process_queue_data(power_domain)

        # Determine which GPUs to query
        target_gpus = [gpu_index] if gpu_index is not None else self.gpu_indices

        result = {}
        for gpu_idx in target_gpus:
            if gpu_idx not in self.samples[power_domain]:
                continue

            # Extract timeline from samples
            timeline = []
            for sample in self.samples[power_domain][gpu_idx]:
                # Apply time filters
                if start_time is not None and sample.timestamp < start_time:
                    continue
                if end_time is not None and sample.timestamp > end_time:
                    continue

                timeline.append((sample.timestamp, sample.power_mw / 1000.0))  # Convert to watts

            # Sort by timestamp
            timeline.sort(key=lambda x: x[0])
            result[gpu_idx] = timeline

        return result

    def get_all_power_timelines(
        self,
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, dict[int, list[tuple[float, float]]]]:
        """Get all power timelines organized by power domain.

        Args:
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp from time.time() or similar)
            end_time: End time filter (unix timestamp from time.time() or similar)

        Returns:
            Dictionary with power domain names as keys and each value is a dict
            mapping GPU indices to timeline data.
        """
        result = {}
        for domain in self.supported_domains:
            result[domain.value] = self.get_power_timeline(domain, gpu_index, start_time, end_time)
        return result

    def get_energy(self, start_time: float, end_time: float) -> dict[int, float] | None:
        """Get the energy used by the GPUs between two times (backward compatibility).

        Uses device instant power for energy calculation.

        Args:
            start_time: Start time of the interval, from time.time().
            end_time: End time of the interval, from time.time().

        Returns:
            A dictionary mapping GPU indices to the energy used by the GPU between the
            two times. If there are no power readings, return None.
        """
        timelines = self.get_power_timeline(PowerDomain.DEVICE_INSTANT, start_time=start_time, end_time=end_time)

        if not timelines:
            return None

        energy_result = {}
        for gpu_idx, timeline in timelines.items():
            if not timeline or len(timeline) < 2:
                energy_result[gpu_idx] = 0.0
                continue

            timestamps = [t[0] for t in timeline]
            powers = [t[1] for t in timeline]

            try:
                energy_result[gpu_idx] = float(auc(timestamps, powers))
            except ValueError:
                energy_result[gpu_idx] = 0.0

        return energy_result

    def get_power(self, time: float | None = None) -> dict[int, float] | None:
        """Get the instant power usage of the GPUs at a specific time point.

        Uses device instant power for compatibility.

        Args:
            time: Time point to get the power usage at. If None, get the power usage
                at the last recorded time point.

        Returns:
            A dictionary mapping GPU indices to the power usage of the GPU at the
            specified time point. If there are no power readings, return None.
        """
        if PowerDomain.DEVICE_INSTANT not in self.supported_domains:
            raise ValueError("PowerDomain.DEVICE_INSTANT is not supported by the current GPUs.")

        # Process any pending queue data
        self._process_all_queue_data()

        result = {}
        for gpu_idx in self.gpu_indices:
            samples = self.samples[PowerDomain.DEVICE_INSTANT][gpu_idx]
            if not samples:
                return None

            if time is None:
                # Get the most recent sample
                latest_sample = samples[-1]
                result[gpu_idx] = latest_sample.power_mw / 1000.0  # Convert to watts
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
                result[gpu_idx] = closest_sample.power_mw / 1000.0  # To Watts

        return result


def _domain_polling_process(
    power_domain: PowerDomain,
    gpu_indices: list[int],
    data_queue: mp.Queue,
    ready_event: EventClass,
    stop_event: EventClass,
    update_period: float,
) -> None:
    """Polling process for a specific power domain with deduplication."""
    try:
        # Get GPUs
        gpus = get_gpus(ensure_homogeneous=True)

        # Determine the GPU method to call based on domain
        power_methods = {
            PowerDomain.DEVICE_INSTANT: gpus.get_instant_power_usage,
            PowerDomain.DEVICE_AVERAGE: gpus.get_average_power_usage,
            PowerDomain.MEMORY_AVERAGE: gpus.get_average_memory_power_usage,
        }
        try:
            power_method = power_methods[power_domain]
        except KeyError:
            raise ValueError(f"Unknown power domain: {power_domain}") from None

        # Track previous power values for deduplication
        prev_power: dict[int, float] = {}

        # Signal that this process is ready to start monitoring
        ready_event.set()

        # Start polling loop
        num_not_supported_encounter = 0
        while not stop_event.is_set():
            timestamp = time()

            for gpu_index in gpu_indices:
                try:
                    power_mw = power_method(gpu_index)

                    # Sometimes, if we poll too fast, power can return 0. Skip.
                    if power_mw <= 0:
                        logger.warning(
                            "GPU %d power domain %s encountered %g mW measurement. "
                            "Skipping. Polling frequency may be too high.",
                            gpu_index,
                            power_domain.value,
                            power_mw,
                        )
                        continue

                    # Deduplication: only send if power changed
                    if gpu_index in prev_power and prev_power[gpu_index] == power_mw:
                        continue

                    prev_power[gpu_index] = power_mw

                    # Create and send power sample
                    sample = PowerSample(
                        timestamp=timestamp,
                        gpu_index=gpu_index,
                        power_mw=power_mw,
                    )

                    data_queue.put(sample)
                except ZeusGPUNotSupportedError as e:
                    # When polling at a high frequency, NVML sometimes raises
                    # a NotSupported error.
                    num_not_supported_encounter += 1
                    if num_not_supported_encounter > 10:
                        num_not_supported_encounter = 0
                        logger.warning(
                            "GPU %d domain %s encountered 10 NotSupported errors. "
                            "This may indicate a polling frequency that is too high. "
                            "Consider increasing the update period. "
                            "Exception: '%s'",
                            gpu_index,
                            power_domain.value,
                            e,
                        )
                except Exception as e:
                    logger.exception(
                        "Error polling power for GPU %d in domain %s: %s",
                        gpu_index,
                        power_domain.value,
                        e,
                    )
                    raise e

            # Sleep for the remaining time
            elapsed = time() - timestamp
            sleep_time = update_period - elapsed
            if sleep_time > 0:
                sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Exiting polling process for domain %s due to error: %s",
            power_domain.value,
            e,
        )
        raise e
    finally:
        # Send stop signal
        data_queue.put("STOP")
