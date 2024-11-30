from __future__ import annotations

import abc
import time
import warnings
import multiprocessing as mp

from prometheus_client import (
    CollectorRegistry,
    Histogram,
    Counter,
    Gauge,
    push_to_gateway,
)

from zeus.monitor.power import PowerMonitor
from zeus.monitor.energy import ZeusMonitor

from zeus.device.cpu import get_cpus


class Metric(abc.ABC):
    """
    Abstract base class for all metric types in Zeus.

    Defines a common interface for metrics, ensuring consistent behavior
    for `begin_window` and `end_window` operations.
    """

    @abc.abstractmethod
    def begin_window(self, name: str):
        """Start a new measurement window."""
        pass

    @abc.abstractmethod
    def end_window(self, name: str):
        """End the current measurement window and report metrics."""
        pass


class EnergyHistogram(Metric):
    """Measures the energy consumption a code range and exports a histogram metrics.

    Tracks energy consumption for GPUs, CPUs, and DRAM as Prometheus Histogram metrics.

    Attributes:
        cpu_indices: List of CPU indices to monitor.
        gpu_indices: List of GPU indices to monitor.
        prometheus_url: Prometheus Push Gateway URL.
        job: Prometheus job name.
        gpu_bucket_range: Histogram buckets for GPU energy.
        cpu_bucket_range: Histogram buckets for CPU energy.
        dram_bucket_range: Histogram buckets for DRAM energy.
    """

    def __init__(
        self,
        cpu_indices: list,
        gpu_indices: list,
        prometheus_url: str,
        job: str,
        gpu_bucket_range: list[float] = [50.0, 100.0, 200.0, 500.0, 1000.0],
        cpu_bucket_range: list[float] = [10.0, 20.0, 50.0, 100.0, 200.0],
        dram_bucket_range: list[float] = [5.0, 10.0, 20.0, 50.0, 150.0],
    ) -> None:
        """
        Initialize the EnergyHistogram class.

        Sets up the Prometheus Histogram metrics to track energy consumption for GPUs, CPUs, and DRAMs.
        The data will be collected and pushed to the Prometheus Push Gateway at regular intervals.

        Args:
            cpu_indices (list): List of CPU indices to monitor.
            gpu_indices (list): List of GPU indices to monitor.
            prometheus_url (str): URL of the Prometheus Push Gateway where metrics will be pushed.
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

        self.gpu_bucket_range = gpu_bucket_range
        self.cpu_bucket_range = cpu_bucket_range
        self.dram_bucket_range = dram_bucket_range
        self.cpu_indices = cpu_indices
        self.gpu_indices = gpu_indices
        self.prometheus_url = prometheus_url
        self.job = job

        self.registry = CollectorRegistry()

        # Initialize GPU histograms
        self.gpu_histograms = {}
        if self.gpu_indices:
            for gpu_index in gpu_indices:
                self.gpu_histograms[gpu_index] = Histogram(
                    f"energy_monitor_gpu_{gpu_index}_energy_joules",
                    f"GPU {gpu_index} energy consumption",
                    ["window", "index"],
                    buckets=self.gpu_bucket_range,
                    registry=self.registry,
                )
        # Initialize CPU histograms
        self.cpu_histograms = {}
        self.dram_histograms = {}
        if self.cpu_indices:
            for cpu_index in self.cpu_indices:
                self.cpu_histograms[cpu_index] = Histogram(
                    f"energy_monitor_cpu_{cpu_index}_energy_joules",
                    f"CPU {cpu_index} energy consumption",
                    ["window", "index"],
                    buckets=self.cpu_bucket_range,
                    registry=self.registry,
                )
            # Initialize CPU and DRAM histograms
            # Only when CPUs are available, we check if DRAM is available.
            for i, cpu in enumerate(get_cpus().cpus):
                if cpu.supportsGetDramEnergyConsumption():
                    self.dram_histograms[i] = Histogram(
                        f"energy_monitor_dram_{i}_energy_joules",
                        f"DRAM {i} energy consumption",
                        ["window", "index"],
                        buckets=self.dram_bucket_range,
                        registry=self.registry,
                    )

        self.max_gpu_bucket = max(self.gpu_bucket_range)
        self.max_cpu_bucket = max(self.cpu_bucket_range)
        self.max_dram_bucket = max(self.dram_bucket_range)

        self.energy_monitor = ZeusMonitor(
            cpu_indices=cpu_indices, gpu_indices=gpu_indices
        )

    def begin_window(self, name: str) -> None:
        """
        Begin the energy monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
        """
        self.energy_monitor.begin_window(
            f"__EnergyHistogram_{name}", sync_execution=True
        )

    def end_window(self, name: str) -> None:
        """
        End the current energy monitoring window and record the energy data.

        Retrieves the energy consumption data (for GPUs, CPUs, and DRAMs) for the monitoring window
        and updates the corresponding Histogram metrics. The data is then pushed to the Prometheus Push Gateway.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.

        Pushes:
            - GPU energy data to the Prometheus Push Gateway via the associated Histogram metric.
            - CPU energy data to the Prometheus Push Gateway via the associated Histogram metric.
            - DRAM energy data to the Prometheus Push Gateway via the associated Histogram metric.
        """
        measurement = self.energy_monitor.end_window(
            f"__EnergyHistogram_{name}", sync_execution=True
        )

        if measurement.gpu_energy:
            for gpu_index, gpu_energy in measurement.gpu_energy.items():
                if gpu_index in self.gpu_histograms:
                    self.gpu_histograms[gpu_index].labels(
                        window=name, index=gpu_index
                    ).observe(gpu_energy)
                if gpu_energy > self.max_gpu_bucket:
                    warnings.warn(
                        f"GPU {gpu_index} energy {gpu_energy} exceeds the maximum bucket value of {self.max_gpu_bucket}"
                    )

        if measurement.cpu_energy:
            for cpu_index, cpu_energy in measurement.cpu_energy.items():
                if cpu_index in self.cpu_histograms:
                    self.cpu_histograms[cpu_index].labels(
                        window=name, index=cpu_index
                    ).observe(cpu_energy)
                if cpu_energy > self.max_cpu_bucket:
                    warnings.warn(
                        f"CPU {cpu_index} energy {cpu_energy} exceeds the maximum bucket value of {self.max_cpu_bucket}"
                    )

        if measurement.dram_energy:
            for dram_index, dram_energy in measurement.dram_energy.items():
                if dram_index in self.dram_histograms:
                    self.dram_histograms[dram_index].labels(
                        window=name, index=dram_index
                    ).observe(dram_energy)
                if dram_energy > self.max_dram_bucket:
                    warnings.warn(
                        f"DRAM {dram_index} energy {dram_energy} exceeds the maximum bucket value of {self.max_dram_bucket}"
                    )

        push_to_gateway(self.prometheus_url, job=self.job, registry=self.registry)


class EnergyCumulativeCounter(Metric):
    """
    EnergyCumulativeCounter class to monitor and record cumulative energy consumption.

    This class tracks GPU, CPU, and DRAM energy usage over time, and records the data as Prometheus Counter metrics.
    The energy consumption metrics are periodically updated and pushed to a Prometheus Push Gateway for monitoring and analysis.

    The cumulative nature of the Counter ensures that energy values are always incremented over time, never reset,
    which is ideal for tracking continuously increasing values like energy usage.

    Attributes:
        energy_monitor: The ZeusMonitor instance that collects energy consumption data for the system.
        update_period: The interval (in seconds) between consecutive energy data updates.
        prometheus_url: The URL of the Prometheus Push Gateway where the Counter metrics will be pushed.
        job: The name of the job associated with the energy monitoring in Prometheus.
        gpu_counters: A dictionary storing the Prometheus Counter metrics for each GPU.
        cpu_counters: A dictionary storing the Prometheus Counter metrics for each CPU.
        dram_counters: A dictionary storing the Prometheus Counter metrics for DRAM.
        queue: A multiprocessing queue used to send signals to start/stop energy monitoring.
        proc: A multiprocessing process that runs the energy monitoring loop.
    """

    def __init__(
        self,
        cpu_indices: list,
        gpu_indices: list,
        update_period: int,
        prometheus_url: str,
        job: str,
    ) -> None:
        """
        Initialize the EnergyCumulativeCounter.

        Args:
            cpu_indices (list): List of CPU indices to monitor.
            gpu_indices (list): List of GPU indices to monitor.
            update_period: The time interval (in seconds) at which energy measurements are updated.
            prometheus_url: The URL for the Prometheus Push Gateway where the metrics will be pushed.
            job: The name of the job to be associated with the Prometheus metrics.
        """
        self.cpu_indices = cpu_indices
        self.gpu_indices = gpu_indices
        self.update_period = update_period
        self.prometheus_url = prometheus_url
        self.job = job
        self.gpu_counters = {}
        self.cpu_counters = {}
        self.dram_counters = {}
        self.queue = None
        self.proc = None

    def begin_window(self, name: str) -> None:
        """
        Begin the energy monitoring window.

        Starts a new multiprocessing process that monitors energy usage periodically
        and pushes the results to the Prometheus Push Gateway.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
        """
        context = mp.get_context("spawn")
        self.queue = context.Queue()
        self.proc = context.Process(
            target=energy_monitoring_loop,
            args=(
                name,
                self.queue,
                self.cpu_indices,
                self.gpu_indices,
                self.update_period,
                self.prometheus_url,
                self.job,
            ),
        )
        self.proc.start()
        if not self.proc.is_alive():
            raise RuntimeError(f"Failed to start monitoring process for {name}.")

    def end_window(self, name: str) -> None:
        """
        End the energy monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
        """

        if not hasattr(self, "queue") or self.queue is None:
            raise RuntimeError(
                "EnergyCumulativeCounter's 'queue' is not initialized. "
                "Make sure 'begin_window' is called before 'end_window'."
            )
        self.queue.put("stop")
        self.proc.join(timeout=20)
        if self.proc.is_alive():
            warnings.warn(f"Forcefully terminating monitoring process for {name}.")
            self.proc.terminate()


def energy_monitoring_loop(
    name: str,
    pipe: mp.Queue,
    cpu_indices: list,
    gpu_indices: list,
    update_period: int,
    prometheus_url: str,
    job: str,
) -> None:
    """
    This function runs in a separate process to collect and update energy consumption metrics
    (for GPUs, CPUs, and DRAM) at regular intervals. It utilizes the Zeus energy monitoring
    framework and pushes the collected data to the Prometheus Push Gateway for real-time tracking.

    Args:
        name (str): The user-defined name of the monitoring window (used as a label for Prometheus metrics).
        pipe (mp.Queue): A multiprocessing queue for inter-process communication, used to signal when to stop the process.
        cpu_indices (list): List of CPU indices to monitor.
        gpu_indices (list): List of GPU indices to monitor.
        update_period (int): The interval (in seconds) between consecutive energy data updates.
        prometheus_url (str): The URL of the Prometheus Push Gateway where the metrics will be pushed.
        job (str): The name of the Prometheus job associated with these metrics.
    """
    registry = CollectorRegistry()
    energy_monitor = ZeusMonitor(cpu_indices=cpu_indices, gpu_indices=gpu_indices)

    if energy_monitor.gpu_indices:
        gpu_counters = {}
        for gpu_index in energy_monitor.gpu_indices:
            gpu_counters[gpu_index] = Counter(
                f"energy_monitor_gpu_{gpu_index}_energy_joules",
                f"GPU {gpu_index} energy consumption",
                ["window", "index"],
                registry=registry,
            )

    if energy_monitor.cpu_indices:
        cpu_counters = {}
        for cpu_index in energy_monitor.cpu_indices:
            cpu_counters[cpu_index] = Counter(
                f"energy_monitor_cpu_{cpu_index}_energy_joules",
                f"CPU {cpu_index} energy consumption",
                ["window", "index"],
                registry=registry,
            )
        dram_counters = {}
        for i, cpu in enumerate(get_cpus().cpus):
            if cpu.supportsGetDramEnergyConsumption():
                dram_counters[i] = Counter(
                    f"energy_monitor_dram_{i}_energy_joules",
                    f"DRAM {i} energy consumption",
                    ["window", "index"],
                    registry=registry,
                )

    while True:
        if not pipe.empty():
            break

        energy_monitor.begin_window(
            f"__EnergyCumulativeCounter_{name}", sync_execution=False
        )
        time.sleep(update_period)
        measurement = energy_monitor.end_window(
            f"__EnergyCumulativeCounter_{name}", sync_execution=False
        )

        if measurement.gpu_energy:
            for gpu_index, energy in measurement.gpu_energy.items():
                if gpu_index in gpu_counters:
                    gpu_counters[gpu_index].labels(window=name, index=gpu_index).inc(
                        energy
                    )

        if measurement.cpu_energy:
            for cpu_index, energy in measurement.cpu_energy.items():
                if cpu_index in cpu_counters:
                    cpu_counters[cpu_index].labels(window=name, index=cpu_index).inc(
                        energy
                    )

        if measurement.dram_energy:
            for dram_index, energy in measurement.dram_energy.items():
                if dram_index in dram_counters:
                    dram_counters[dram_index].labels(window=name, index=dram_index).inc(
                        energy
                    )

        push_to_gateway(prometheus_url, job=job, registry=registry)


class PowerGauge(Metric):
    """
    PowerGauge class to monitor and record power consumption.

    This class tracks GPU power usage in real time and records it as **Prometheus Gauge** metrics.
    The Gauge metric type is suitable for tracking values that can go up and down over time, like power consumption.

    Power usage data is collected at regular intervals and pushed to a Prometheus Push Gateway for monitoring.

    Attributes:
        gpu_indices: List of GPU indices to monitor for power consumption.
        update_period: Time interval (in seconds) between consecutive power measurements.
        prometheus_url: URL of the Prometheus Push Gateway where Gauge metrics are pushed.
        job: Name of the Prometheus job associated with the power metrics.
        gpu_gauges (dict[int, Gauge]): Dictionary mapping GPU indices to Prometheus Gauge metrics for real-time power consumption tracking.
        queue: Queue for controlling the monitoring process.
        proc: Process running the power monitoring loop.
    """

    def __init__(
        self,
        gpu_indices: list,
        update_period: int,
        prometheus_url: str,
        job: str,
    ) -> None:
        """
        Initialize the PowerGauge metric.

        Args:
            gpu_indices (list[int]): List of GPU indices to monitor for power consumption.
            update_period (int): Interval (in seconds) between consecutive power measurements.
            prometheus_url (str): URL of the Prometheus Push Gateway where Gauge metrics are pushed.
            job (str): Name of the Prometheus job to associate with the power metrics.
        """
        self.gpu_indices = gpu_indices
        self.update_period = update_period
        self.prometheus_url = prometheus_url
        self.job = job
        self.gpu_gauges = {}

    def begin_window(self, name: str) -> None:
        """
        Begin the power monitoring window.

        Starts a new multiprocessing process that runs the power monitoring loop.
        The process collects real-time power consumption data and updates the corresponding
        Gauge metrics in Prometheus.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
        """
        context = mp.get_context("spawn")
        self.queue = context.Queue()
        self.proc = context.Process(
            target=power_monitoring_loop,
            args=(
                name,
                self.queue,
                self.gpu_indices,
                self.update_period,
                self.prometheus_url,
                self.job,
            ),
        )
        self.proc.start()
        time.sleep(5)

    def end_window(self, name: str) -> None:
        """
        End the power monitoring window.

        Args:
            name (str): The unique name of the measurement window. Must match between calls to 'begin_window' and 'end_window'.
        """

        self.queue.put("stop")
        self.proc.join(timeout=20)
        if self.proc.is_alive():
            warnings.warn(f"Forcefully terminating monitoring process for {name}.")
            self.proc.terminate()


def power_monitoring_loop(
    name: str,
    pipe: mp.Queue,
    gpu_indices: list[int],
    update_period: int,
    prometheus_url: str,
    job: str,
) -> None:
    """
    The polling function for power monitoring that runs in a separate process.

    It periodically collects power consumption data for each GPU and pushes the results
    to the Prometheus Push Gateway.

    Args:
        name (str): Unique name for the monitoring window (used as a label in Prometheus metrics).
        pipe (multiprocessing.Queue): Queue to receive control signals (e.g., "stop").
        gpu_indices (list[int]): List of GPU indices to monitor for power consumption.
        update_period (int): Interval (in seconds) between consecutive power data polls.
        prometheus_url (str): URL of the Prometheus Push Gateway where metrics are pushed.
        job (str): Name of the Prometheus job to associate with the metrics.
    """
    gpu_gauges = {}
    power_monitor = PowerMonitor(gpu_indices=gpu_indices)
    registry = CollectorRegistry()

    for gpu_index in gpu_indices:
        gpu_gauges[gpu_index] = Gauge(
            f"power_monitor_gpu_{gpu_index}_power_watts",
            f"Records power consumption for GPU {gpu_index} over time",
            ["gpu_index"],
            registry=registry,
        )

    while True:
        if not pipe.empty():
            break

        power_measurement = power_monitor.get_power()

        try:
            for gpu_index, power_value in power_measurement.items():
                gpu_gauges[gpu_index].labels(gpu_index=f"{name}_gpu{gpu_index}").set(
                    power_value
                )
        except Exception as e:
            print(f"Error during processing power measurement: {e}")

        try:
            push_to_gateway(prometheus_url, job=job, registry=registry)
        except Exception as e:
            print(f"Error pushing metrics: {e}")

        time.sleep(update_period)
