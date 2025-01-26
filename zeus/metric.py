"""Track and export energy and power metrics via Prometheus."""

from __future__ import annotations

import abc
import time
import warnings
from typing import Sequence
import multiprocessing as mp
from multiprocessing.context import SpawnProcess
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Histogram,
    Counter,
    Gauge,
    push_to_gateway,
)

from zeus.monitor.power import PowerMonitor
from zeus.monitor.energy import ZeusMonitor
from zeus.utils.framework import sync_execution as sync_execution_fn
from zeus.device.cpu import get_cpus


@dataclass
class MonitoringProcessState:
    """Represents the state of a monitoring window."""

    queue: mp.Queue
    proc: SpawnProcess


class Metric(abc.ABC):
    """Abstract base class for all metric types in Zeus.

    Defines a common interface for metrics, ensuring consistent behavior
    for `begin_window` and `end_window` operations.
    """

    @abc.abstractmethod
    def begin_window(self, name: str, sync_execution: bool = True) -> None:
        """Start a new measurement window.

        Args:
            name (str): Name of the measurement window.
            sync_execution (bool): Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window.
        """
        pass

    @abc.abstractmethod
    def end_window(self, name: str, sync_execution: bool = True) -> None:
        """End the current measurement window and report metrics.

        Args:
            name (str): Name of the measurement window.
            sync_execution (bool): Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window.
        """
        pass


class EnergyHistogram(Metric):
    """Measures the energy consumption a code range and exports a histogram metrics.

    Tracks energy consumption for GPUs, CPUs, and DRAM as Prometheus Histogram metrics.
    """

    def __init__(
        self,
        cpu_indices: list,
        gpu_indices: list,
        pushgateway_url: str,
        job: str,
        gpu_bucket_range: Sequence[float] = [50.0, 100.0, 200.0, 500.0, 1000.0],
        cpu_bucket_range: Sequence[float] = [10.0, 50.0, 100.0, 500.0, 1000.0],
        dram_bucket_range: Sequence[float] = [5.0, 10.0, 20.0, 50.0, 150.0],
    ) -> None:
        """Initialize the EnergyHistogram class.

        Sets up the Prometheus Histogram metrics to track energy consumption for GPUs, CPUs, and DRAMs.
        The data will be collected and pushed to the Prometheus Push Gateway at regular intervals.

        Args:
            cpu_indices (list): List of CPU indices to monitor.
            gpu_indices (list): List of GPU indices to monitor.
            pushgateway_url (str): URL of the Prometheus Push Gateway where metrics will be pushed.
            job (str): Name of the Prometheus job to associate with the energy metrics.
            gpu_bucket_range (list[float], optional): Bucket ranges for GPU energy histograms.
                Defaults to [50.0, 100.0, 200.0, 500.0, 1000.0].
            cpu_bucket_range (list[float], optional): Bucket ranges for CPU energy histograms.
                Defaults to [10.0, 20.0, 50.0, 100.0, 200.0].
            dram_bucket_range (list[float], optional): Bucket ranges for DRAM energy histograms.
                Defaults to [5.0, 10.0, 20.0, 50.0, 150.0].

        Raises:
            ValueError: If any of the bucket ranges (GPU, CPU, DRAM) is an empty list.
        """
        self.gpu_bucket_range = gpu_bucket_range
        self.cpu_bucket_range = cpu_bucket_range
        self.dram_bucket_range = dram_bucket_range
        self.cpu_indices = cpu_indices
        self.gpu_indices = gpu_indices
        self.pushgateway_url = pushgateway_url
        self.job = job
        self.registry = CollectorRegistry()

        if not gpu_bucket_range:
            raise ValueError(
                "GPU bucket range cannot be empty. Please provide a valid range or omit the argument to use defaults."
            )
        if not cpu_bucket_range:
            raise ValueError(
                "CPU bucket range cannot be empty. Please provide a valid range or omit the argument to use defaults."
            )
        if not dram_bucket_range:
            raise ValueError(
                "DRAM bucket range cannot be empty. Please provide a valid range or omit the argument to use defaults."
            )

        # Initialize GPU histograms
        if self.gpu_indices:
            self.gpu_histograms = Histogram(
                "energy_monitor_gpu_energy_joules",
                "GPU energy consumption",
                ["window", "index"],
                buckets=self.gpu_bucket_range,
                registry=self.registry,
            )
        # Initialize CPU histograms
        if self.cpu_indices:
            self.cpu_histograms = Histogram(
                "energy_monitor_cpu_energy_joules",
                "CPU energy consumption",
                ["window", "index"],
                buckets=self.cpu_bucket_range,
                registry=self.registry,
            )
            # Initialize CPU and DRAM histograms
            if any(cpu.supportsGetDramEnergyConsumption() for cpu in get_cpus().cpus):
                self.dram_histograms = Histogram(
                    "energy_monitor_dram_energy_joules",
                    "DRAM energy consumption",
                    ["window", "index"],
                    buckets=self.dram_bucket_range,
                    registry=self.registry,
                )

        self.max_gpu_bucket = max(self.gpu_bucket_range)
        self.max_cpu_bucket = max(self.cpu_bucket_range)
        self.max_dram_bucket = max(self.dram_bucket_range)

        self.min_gpu_bucket = min(self.gpu_bucket_range)
        self.min_cpu_bucket = min(self.cpu_bucket_range)
        self.min_dram_bucket = min(self.dram_bucket_range)

        self.energy_monitor = ZeusMonitor(
            cpu_indices=cpu_indices, gpu_indices=gpu_indices
        )

    def begin_window(self, name: str, sync_execution: bool = True) -> None:
        """Begin the energy monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool): Whether to execute synchronously. Defaults to True. If assigned True, calls sync_execution_fn with the defined gpu
        """
        if sync_execution:
            sync_execution_fn(self.gpu_indices)

        self.energy_monitor.begin_window(
            f"__EnergyHistogram_{name}", sync_execution=sync_execution
        )

    def end_window(self, name: str, sync_execution: bool = True) -> None:
        """End the current energy monitoring window and record the energy data.

        Retrieves the energy consumption data (for GPUs, CPUs, and DRAMs) for the monitoring window
        and updates the corresponding Histogram metrics. The data is then pushed to the Prometheus Push Gateway.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool): Whether to execute synchronously. Defaults to True.
        """
        if sync_execution:
            sync_execution_fn(self.gpu_indices)

        measurement = self.energy_monitor.end_window(
            f"__EnergyHistogram_{name}", sync_execution=sync_execution
        )

        if measurement.gpu_energy:
            for gpu_index, gpu_energy in measurement.gpu_energy.items():
                self.gpu_histograms.labels(window=name, index=gpu_index).observe(
                    gpu_energy
                )
                if gpu_energy > self.max_gpu_bucket:
                    warnings.warn(
                        f"GPU {gpu_index} energy {gpu_energy} exceeds the maximum bucket value of {self.max_gpu_bucket}",
                        stacklevel=1,
                    )
                if gpu_energy < self.min_gpu_bucket:
                    warnings.warn(
                        f"GPU {gpu_index} energy {gpu_energy} exceeds the minimum bucket value of {self.min_gpu_bucket}",
                        stacklevel=1,
                    )

        if measurement.cpu_energy:
            for cpu_index, cpu_energy in measurement.cpu_energy.items():
                self.cpu_histograms.labels(window=name, index=cpu_index).observe(
                    cpu_energy
                )
                if cpu_energy > self.max_cpu_bucket:
                    warnings.warn(
                        f"CPU {cpu_index} energy {cpu_energy} exceeds the maximum bucket value of {self.max_cpu_bucket}",
                        stacklevel=1,
                    )
                if cpu_energy < self.min_cpu_bucket:
                    warnings.warn(
                        f"CPU {cpu_index} energy {cpu_energy} exceeds the minimum bucket value of {self.min_cpu_bucket}",
                        stacklevel=1,
                    )

        if measurement.dram_energy:
            for dram_index, dram_energy in measurement.dram_energy.items():
                self.dram_histograms.labels(window=name, index=dram_index).observe(
                    dram_energy
                )
                if dram_energy > self.max_dram_bucket:
                    warnings.warn(
                        f"DRAM {dram_index} energy {dram_energy} exceeds the maximum bucket value of {self.max_dram_bucket}",
                        stacklevel=1,
                    )
                if dram_energy < self.min_dram_bucket:
                    warnings.warn(
                        f"DRAM {dram_index} energy {dram_energy} exceeds the minimum bucket value of {self.min_dram_bucket}",
                        stacklevel=1,
                    )

        push_to_gateway(self.pushgateway_url, job=self.job, registry=self.registry)


class EnergyCumulativeCounter(Metric):
    """EnergyCumulativeCounter class to monitor and record cumulative energy consumption.

    This class tracks GPU, CPU, and DRAM energy usage over time, and records the data as Prometheus Counter metrics.
    The energy consumption metrics are periodically updated and pushed to a Prometheus Push Gateway for monitoring and analysis.

    The cumulative nature of the Counter ensures that energy values are always incremented over time, never reset,
    which is ideal for tracking continuously increasing values like energy usage.
    """

    def __init__(
        self,
        cpu_indices: list,
        gpu_indices: list,
        update_period: int,
        pushgateway_url: str,
        job: str,
    ) -> None:
        """Initialize the EnergyCumulativeCounter.

        Args:
            cpu_indices (list): List of CPU indices to monitor.
            gpu_indices (list): List of GPU indices to monitor.
            update_period: The time interval (in seconds) at which energy measurements are updated.
            pushgateway_url: The URL for the Prometheus Push Gateway where the metrics will be pushed.
            job: The name of the job to be associated with the Prometheus metrics.
        """
        self.cpu_indices = cpu_indices
        self.gpu_indices = gpu_indices
        self.update_period = update_period
        self.pushgateway_url = pushgateway_url
        self.job = job
        self.window_state: dict[str, MonitoringProcessState] = {}

    def begin_window(self, name: str, sync_execution: bool = False) -> None:
        """Begin the energy monitoring window.

        Starts a new multiprocessing process that monitors energy usage periodically
        and pushes the results to the Prometheus Push Gateway.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        if sync_execution:
            sync_execution_fn(self.gpu_indices)

        context = mp.get_context("spawn")
        queue = context.Queue()
        proc = context.Process(
            target=energy_monitoring_loop,
            args=(
                name,
                queue,
                self.cpu_indices,
                self.gpu_indices,
                self.update_period,
                self.pushgateway_url,
                self.job,
            ),
        )
        proc.start()
        if not proc.is_alive():
            raise RuntimeError(f"Failed to start monitoring process for {name}.")

        self.window_state[name] = MonitoringProcessState(queue=queue, proc=proc)

    def end_window(self, name: str, sync_execution: bool = False) -> None:
        """End the energy monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        if name not in self.window_state:
            raise ValueError(f"No active monitoring process found for '{name}'.")

        if sync_execution:
            sync_execution_fn(self.gpu_indices)

        state = self.window_state.pop(name)
        state.queue.put("stop")
        state.proc.join(timeout=20)

        if state.proc.is_alive():
            state.proc.terminate()


def energy_monitoring_loop(
    name: str,
    pipe: mp.Queue,
    cpu_indices: list,
    gpu_indices: list,
    update_period: int,
    pushgateway_url: str,
    job: str,
) -> None:
    """Runs in a separate process to collect and update energy consumption metrics (for GPUs, CPUs, and DRAM).

    Args:
        name (str): The user-defined name of the monitoring window (used as a label for Prometheus metrics).
        pipe (mp.Queue): A multiprocessing queue for inter-process communication, used to signal when to stop the process.
        cpu_indices (list): List of CPU indices to monitor.
        gpu_indices (list): List of GPU indices to monitor.
        update_period (int): The interval (in seconds) between consecutive energy data updates.
        pushgateway_url (str): The URL of the Prometheus Push Gateway where the metrics will be pushed.
        job (str): The name of the Prometheus job associated with these metrics.
    """
    registry = CollectorRegistry()
    energy_monitor = ZeusMonitor(cpu_indices=cpu_indices, gpu_indices=gpu_indices)
    gpu_counters = None
    cpu_counters = None
    dram_counters = None

    if energy_monitor.gpu_indices:
        gpu_counters = Counter(
            "energy_monitor_gpu_energy_joules",
            "GPU energy consumption",
            ["window", "index"],
            registry=registry,
        )

    if energy_monitor.cpu_indices:
        cpu_counters = Counter(
            "energy_monitor_cpu_energy_joules",
            "CPU energy consumption",
            ["window", "index"],
            registry=registry,
        )
        if any(cpu.supportsGetDramEnergyConsumption() for cpu in get_cpus().cpus):
            dram_counters = Counter(
                "energy_monitor_dram_energy_joules",
                "DRAM energy consumption",
                ["window", "index"],
                registry=registry,
            )

    while True:
        if not pipe.empty():
            break
        # Begin and end monitoring window using sync_execution
        energy_monitor.begin_window(
            f"__EnergyCumulativeCounter_{name}", sync_execution=False
        )
        time.sleep(update_period)
        measurement = energy_monitor.end_window(
            f"__EnergyCumulativeCounter_{name}", sync_execution=False
        )

        if measurement.gpu_energy:
            for gpu_index, energy in measurement.gpu_energy.items():
                if gpu_counters:
                    gpu_counters.labels(window=name, index=gpu_index).inc(energy)

        if measurement.cpu_energy:
            for cpu_index, energy in measurement.cpu_energy.items():
                if cpu_counters:
                    cpu_counters.labels(window=name, index=cpu_index).inc(energy)

        if measurement.dram_energy:
            for dram_index, energy in measurement.dram_energy.items():
                if dram_counters:
                    dram_counters.labels(window=name, index=dram_index).inc(energy)
        # Push metrics to Prometheus
        push_to_gateway(pushgateway_url, job=job, registry=registry)


class PowerGauge(Metric):
    """PowerGauge class to monitor and record power consumption.

    This class tracks GPU power usage in real time and records it as **Prometheus Gauge** metrics.
    The Gauge metric type is suitable for tracking values that can go up and down over time, like power consumption.

    Power usage data is collected at regular intervals and pushed to a Prometheus Push Gateway for monitoring.
    """

    def __init__(
        self,
        gpu_indices: list,
        update_period: int,
        pushgateway_url: str,
        job: str,
    ) -> None:
        """Initialize the PowerGauge metric.

        Args:
            gpu_indices (list[int]): List of GPU indices to monitor for power consumption.
            update_period (int): Interval (in seconds) between consecutive power measurements.
            pushgateway_url (str): URL of the Prometheus Push Gateway where Gauge metrics are pushed.
            job (str): Name of the Prometheus job to associate with the power metrics.
        """
        self.gpu_indices = gpu_indices
        self.update_period = update_period
        self.pushgateway_url = pushgateway_url
        self.job = job
        self.window_state: dict[str, MonitoringProcessState] = {}

    def begin_window(self, name: str, sync_execution: bool = False) -> None:
        """Begin the power monitoring window.

        Starts a new multiprocessing process that runs the power monitoring loop.
        The process collects real-time power consumption data and updates the corresponding
        Gauge metrics in Prometheus.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        if name in self.window_state:
            raise ValueError(f"PowerGauge metric '{name}' already exists.")

        if sync_execution:
            sync_execution_fn(self.gpu_indices)

        context = mp.get_context("spawn")
        queue = context.Queue()
        proc = context.Process(
            target=power_monitoring_loop,
            args=(
                name,
                queue,
                self.gpu_indices,
                self.update_period,
                self.pushgateway_url,
                self.job,
            ),
        )
        proc.start()
        if not proc.is_alive():
            raise RuntimeError(
                f"Failed to start power monitoring process for '{name}'."
            )

        self.window_state[name] = MonitoringProcessState(queue=queue, proc=proc)

    def end_window(self, name: str, sync_execution: bool = False) -> None:
        """End the power monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
            sync_execution (bool, optional): Whether to execute monitoring synchronously. Defaults to False.
        """
        if sync_execution:
            sync_execution_fn(self.gpu_indices)

        state = self.window_state.pop(name)
        state.queue.put("stop")
        state.proc.join(timeout=20)

        if state.proc.is_alive():
            state.proc.terminate()


def power_monitoring_loop(
    name: str,
    pipe: mp.Queue,
    gpu_indices: list[int],
    update_period: int,
    pushgateway_url: str,
    job: str,
) -> None:
    """Runs in a separate process and periodically collects power consumption data for each GPU and pushes the results to the Prometheus Push Gateway.

    Args:
        name (str): Unique name for the monitoring window (used as a label in Prometheus metrics).
        pipe (multiprocessing.Queue): Queue to receive control signals (e.g., "stop").
        gpu_indices (list[int]): List of GPU indices to monitor for power consumption.
        update_period (int): Interval (in seconds) between consecutive power data polls.
        pushgateway_url (str): URL of the Prometheus Push Gateway where metrics are pushed.
        job (str): Name of the Prometheus job to associate with the metrics.
    """
    power_monitor = PowerMonitor(gpu_indices=gpu_indices)
    registry = CollectorRegistry()

    gpu_gauges = Gauge(
        "power_monitor_gpu_power_watts",
        "Records power consumption for GPU over time",
        ["window", "index"],
        registry=registry,
    )

    while True:
        if not pipe.empty():
            break

        power_measurement = power_monitor.get_power()

        try:
            if power_measurement:
                for gpu_index, power_value in power_measurement.items():
                    gpu_gauges.labels(window=name, index=gpu_index).set(power_value)
        except Exception as e:
            print(f"Error during processing power measurement: {e}")

        try:
            push_to_gateway(pushgateway_url, job=job, registry=registry)
        except Exception as e:
            print(f"Error pushing metrics: {e}")

        time.sleep(update_period)
