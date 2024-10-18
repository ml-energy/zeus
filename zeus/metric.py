from __future__ import annotations
import abc
import time
import warnings
import multiprocessing as mp
from zeus.monitor.power import PowerMonitor
from zeus.monitor.energy import ZeusMonitor
from zeus.device.cpu.common import CPU
from prometheus_client import CollectorRegistry, Histogram, Counter, Gauge, push_to_gateway

class Metric(abc.ABC):  
    @abc.abstractmethod
    def begin_window(self, name: str):
        pass

    @abc.abstractmethod
    def end_window(self, name: str):
        pass

class EnergyHistogram(Metric):
    """EnergyHistogram class to monitor and record energy consumption metrics.

    This class tracks GPU, CPU, and DRAM energy usage, and records the data as Prometheus Histogram metrics.
    The energy data is pushed to a Prometheus Push Gateway for monitoring and analysis.

    Attributes:
        energy_monitor: The ZeusMonitor instance that collects energy consumption data.
        prometheus_url: The URL of the Prometheus Push Gateway where the metrics will be pushed.
        job: The name of the job to associate with the Prometheus metrics.
        registry: The Prometheus CollectorRegistry that manages all the Histogram metrics for this class.
        bucket_ranges: Optional custom bucket ranges for the Histogram metrics (GPU, CPU, and DRAM).
        gpu_histograms: A dictionary mapping GPU indices to their respective Histogram metrics.
        cpu_histograms: A dictionary mapping CPU indices to their respective Histogram metrics.
        dram_histograms: A dictionary mapping DRAM indices to their respective Histogram metrics.
    """

    def __init__(
            self, 
            energy_monitor: ZeusMonitor, 
            prometheus_url: str, 
            job: str, 
            bucket_ranges=None
    ) -> None:
        """
        Initialize the EnergyHistogram class.

        Sets up the Prometheus Histogram metrics to track energy consumption for GPUs, CPUs, and DRAMs.
        The data will be collected and pushed to the Prometheus Push Gateway at regular intervals.

        Args:
            energy_monitor: The ZeusMonitor instance used to retrieve energy data for the system.
            prometheus_url: The URL for the Prometheus Push Gateway where the metrics will be sent.
            job: The name of the Prometheus job associated with the energy metrics.
            bucket_ranges: Optional custom bucket ranges for the Histogram metrics (GPU, CPU, and DRAM). 
                        If not provided, default bucket ranges will be used for each component.
        """
        self.energy_monitor = energy_monitor
        self.prometheus_url = prometheus_url
        self.job = job

        self.registry = CollectorRegistry()

        default_gpu_buckets = [50.0, 100.0, 200.0, 500.0, 1000.0]
        default_cpu_buckets = [10.0, 20.0, 50.0, 100.0, 200.0]
        default_dram_buckets = [5.0, 10.0, 20.0, 50.0, 150.0]

        self.bucket_ranges = {
            'gpu': default_gpu_buckets,
            'cpu': default_cpu_buckets,
            'dram': default_dram_buckets,
        }

        self.bucket_ranges['gpu'] = (
            bucket_ranges.get('gpu') if bucket_ranges and 'gpu' in bucket_ranges
            else default_gpu_buckets
        )

        self.bucket_ranges['cpu'] = (
            bucket_ranges.get('cpu') if bucket_ranges and 'cpu' in bucket_ranges
            else default_cpu_buckets
        )

        self.bucket_ranges['dram'] = (
            bucket_ranges.get('dram') if bucket_ranges and 'dram' in bucket_ranges
            else default_dram_buckets
        )
        # If GPU availble, for each gpu_indices, create a Histogram metric with the label window, and index.
        if energy_monitor.gpu_indices:
            self.gpu_histograms = {}
            for gpu_index in self.energy_monitor.gpu_indices:
                self.gpu_histograms[gpu_index] = Histogram(
                    f'energy_monitor_gpu_{gpu_index}_energy_joules',
                    f'GPU {gpu_index} energy consumption',
                    ['window', 'index'],  
                    buckets=self.bucket_ranges.get('gpu', []),
                    registry=self.registry
                )
        else:
            self.gpu_histogram = None
        # If CPU available, for each cpu_indices, create a Histogram metric with the label window, and index.
        if energy_monitor.cpu_indices:
            self.cpu_histograms = {}
            for cpu_index in self.energy_monitor.cpu_indices:
                self.cpu_histograms[cpu_index] = Histogram(
                    f'energy_monitor_cpu_{cpu_index}_energy_joules',
                    f'CPU {cpu_index} energy consumption',
                    ['window', 'index'],  
                    buckets=self.bucket_ranges.get('cpu', []),
                    registry=self.registry
                )
            # Only when CPUs are available, we check if DRAM is available using supportsGetDramEnergyConsumption in CPU class
            # If DRAM available, we create histogram for each DRAM indices for each CPU indices
            if CPU.supportsGetDramEnergyConsumption:
                self.dram_histograms = {}
                for dram_index in self.energy_monitor.cpu_indices:
                    self.dram_histograms[dram_index] = Histogram(
                        f'energy_monitor_dram_{dram_index}_energy_joules',
                        f'DRAM {dram_index} energy consumption',
                        ['window', 'index'],  
                        buckets=self.bucket_ranges.get('dram', []),
                        registry=self.registry
                    )
            else:
                self.dram_histogram = None
        else:
            self.cpu_histogram = None

        self.max_gpu_bucket = max(self.bucket_ranges.get('gpu'))
        self.max_cpu_bucket = max(self.bucket_ranges.get('cpu'))
        self.max_dram_bucket = max(self.bucket_ranges.get('dram'))
        
    def begin_window(self, name: str) -> None:
        """Begin a new energy monitoring window."""
        self.energy_monitor.begin_window(f"__EnergyHistogram_{name}")
        
    def end_window(self, name: str) -> None:
        """
        End the current energy monitoring window and record the energy data.

        Retrieves the energy consumption data (for GPUs, CPUs, and DRAMs) for the monitoring window
        and updates the corresponding Histogram metrics. The data is then pushed to the Prometheus Push Gateway.

        Args:
            name: The name of the monitoring window (used as a label for the Prometheus Histogram metrics).

        Pushes:
            - GPU energy data to the Prometheus Push Gateway via the associated Histogram metric.
            - CPU energy data to the Prometheus Push Gateway via the associated Histogram metric.
            - DRAM energy data to the Prometheus Push Gateway via the associated Histogram metric.
        """
        measurement = self.energy_monitor.end_window(f"__EnergyHistogram_{name}")

        if measurement.gpu_energy:
            for gpu_index, gpu_energy in measurement.gpu_energy.items():
                if gpu_index in self.gpu_histograms:
                    self.gpu_histograms[gpu_index].labels(window=f"__EnergyHistogram_{name}", index=gpu_index).observe(gpu_energy)
                if gpu_energy > self.max_gpu_bucket:
                    warnings.warn(f"GPU {gpu_index} energy {gpu_energy} exceeds the maximum bucket value of {self.max_gpu_bucket}")
        
        if measurement.cpu_energy:
            for cpu_index, cpu_energy in measurement.cpu_energy.items():
                if cpu_index in self.cpu_histograms:
                    self.cpu_histograms[cpu_index].labels(window=f"__EnergyHistogram_{name}", index=cpu_index).observe(cpu_energy)
                if cpu_energy > self.max_cpu_bucket:
                    warnings.warn(f"CPU {cpu_index} energy {cpu_energy} exceeds the maximum bucket value of {self.max_cpu_bucket}")

        if measurement.dram_energy:
            for dram_index, dram_energy in measurement.dram_energy.items():
                if dram_index in self.dram_histograms:
                    self.dram_histograms[dram_index].labels(window=f"__EnergyHistogram_{name}", index=dram_index).observe(dram_energy)
                if dram_energy > self.max_dram_bucket:
                    warnings.warn(f"DRAM {dram_index} energy {dram_energy} exceeds the maximum bucket value of {self.max_dram_bucket}")
        
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
        queue: A multiprocessing queue used to send signals to start/stop energy monitoring.
        proc: A multiprocessing process that runs the energy monitoring loop.
    """

    def __init__(
            self, 
            energy_monitor: ZeusMonitor, 
            update_period: int, 
            prometheus_url: str, 
            job: str
    )-> None:
        """
        Initialize the EnergyCumulativeCounter.

        Args:
            energy_monitor: The ZeusMonitor instance used to monitor energy consumption.
            update_period: The time interval (in seconds) at which energy measurements are updated.
            prometheus_url: The URL for the Prometheus Push Gateway where the metrics will be pushed.
            job: The name of the job to be associated with the Prometheus metrics.
        """
        self.energy_monitor = energy_monitor  
        self.update_period = update_period  
        self.prometheus_url = prometheus_url  
        self.job = job  

    def begin_window(self, name: str) -> None:
        """
        Begin the energy monitoring window.

        Starts a new multiprocessing process that monitors energy usage periodically
        and pushes the results to the Prometheus Push Gateway.

        Args:
            name: A unique name for the monitoring window (used as a label in Prometheus metrics).
        """
        self.queue = mp.Queue()
        self.proc = mp.Process(
            target=energy_monitoring_loop,
            args=(name, self.queue, self.energy_monitor, self.update_period, self.prometheus_url, self.job)
        )
        self.proc.start()

    def end_window(self, name: str)-> None:
        """End the energy monitoring window."""
        self.queue.put("stop")
        self.proc.join()

def energy_monitoring_loop(
        name: str, 
        pipe: mp.Queue, 
        energy_monitor: ZeusMonitor, 
        update_period: int, 
        prometheus_url: str, 
        job: str
) -> None:
    """
    The polling function that runs in a separate process to monitor energy usage.

    It periodically collects energy consumption metrics from the energy monitor and
    pushes the results to the Prometheus Push Gateway.

    Args:
        name: The unique name of the monitoring window.
        pipe: A multiprocessing queue used to receive signals (e.g., to stop the process).
        energy_monitor: The ZeusMonitor instance used to retrieve energy data.
        update_period: The interval (in seconds) between energy data polls.
        prometheus_url: The URL of the Prometheus Push Gateway.
        job: The job name used in Prometheus for Counter metrics.
    """
    registry = CollectorRegistry()

    if energy_monitor.gpu_indices:
        gpu_counters = {}
        for gpu_index in energy_monitor.gpu_indices:
            gpu_counters[gpu_index] = Counter(
                f'energy_monitor_gpu_{gpu_index}_energy_joules',
                f'GPU {gpu_index} energy consumption',
                ['window', 'index'],
                registry=registry
            )

    if energy_monitor.cpu_indices:
        cpu_counters = {}
        for cpu_index in energy_monitor.cpu_indices:
            cpu_counters[cpu_index] = Counter(
                f'energy_monitor_cpu_{cpu_index}_energy_joules',
                f'CPU {cpu_index} energy consumption',
                ['window', 'index'],
                registry=registry
            )

        if CPU.supportsGetDramEnergyConsumption:
            dram_counters = {}
            for dram_index in energy_monitor.cpu_indices:
                dram_counters[dram_index] = Counter(
                    f'energy_monitor_dram_{dram_index}_energy_joules',
                    f'DRAM {dram_index} energy consumption',
                    ['window', 'index'],
                    registry=registry
                )

    while True:
        if not pipe.empty():
            signal = pipe.get()
            if signal == "stop":
                break  

        energy_monitor.begin_window(f"__EnergyCumulativeCounter_{name}")
        time.sleep(update_period)  
        measurement = energy_monitor.end_window(f"__EnergyCumulativeCounter_{name}")

        if measurement.gpu_energy:
            for gpu_index, energy in measurement.gpu_energy.items():
                if gpu_index in gpu_counters:
                    gpu_counters[gpu_index].labels(window=f"__EnergyCumulativeCounter_{name}", index=gpu_index).inc(energy)

        if measurement.cpu_energy:
            for cpu_index, energy in measurement.cpu_energy.items():
                if cpu_index in cpu_counters:
                    cpu_counters[cpu_index].labels(window=f"__EnergyCumulativeCounter_{name}", index=cpu_index).inc(energy)

        if measurement.dram_energy:
            for dram_index, energy in measurement.dram_energy.items():
                if dram_index in dram_counters:
                    dram_counters[dram_index].labels(window=f"__EnergyCumulativeCounter_{name}", index=dram_index).inc(energy)

        push_to_gateway(prometheus_url, job=job, registry=registry)

class PowerGauge(Metric):
    """
    PowerGauge class to monitor and record power consumption.

    This class tracks GPU power usage in real time and records it as **Prometheus Gauge** metrics. 
    The Gauge metric type is suitable for tracking values that can go up and down over time, like power consumption.

    Power usage data is collected at regular intervals and pushed to a Prometheus Push Gateway for monitoring.

    Attributes:
        power_monitor: The PowerMonitor instance that retrieves power consumption data for the GPUs.
        update_period: The time interval (in seconds) between consecutive power measurements.
        prometheus_url: The URL of the Prometheus Push Gateway where the Gauge metrics will be pushed.
        job: The name of the job associated with the power metrics in Prometheus.
        queue: A multiprocessing queue used to send signals to start/stop power monitoring.
        proc: A multiprocessing process that runs the power monitoring loop.
    """

    def __init__(
            self, 
            power_monitor: PowerMonitor, 
            update_period: int, 
            prometheus_url: str, 
            job: str
    ) -> None:
        """
        Initialize the PowerGauge metric.

        Args:
            power_monitor: The PowerMonitor instance used to monitor power consumption.
            update_period: The interval (in seconds) between power measurement updates.
            prometheus_url: The URL for the Prometheus Push Gateway where the metrics will be pushed.
            job: The name of the job to be associated with the Prometheus metrics.
        """
        self.power_monitor = power_monitor
        self.update_period = update_period  
        self.prometheus_url = prometheus_url
        self.job = job  
    
    def begin_window(self, name: str) -> None:
        """
        Begin the power monitoring window.

        Starts a new multiprocessing process that runs the power monitoring loop. 
        The process collects real-time power consumption data and updates the corresponding 
        Gauge metrics in Prometheus.

        Args:
            name: A unique name for the monitoring window, used as a label for the Prometheus Gauge metrics.
        """
        self.queue = mp.Queue()
        self.proc = mp.Process(
            target=power_monitoring_loop,
            args=(name, self.queue, self.power_monitor, self.update_period, self.prometheus_url, self.job)
        )
        self.proc.start() 

    def end_window(self, name: str) -> None:
        """End the power monitoring window."""
        self.queue.put("stop")
        self.proc.join()

# For each GPU, it creates a Prometheus Gauge to record power consumption over time. 
# Each gauge is associated with a specific GPU index, and Prometheus uses these to track power consumption.
def power_monitoring_loop(
        name: str, 
        pipe: mp.Queue, 
        power_monitor: PowerMonitor, 
        update_period: int, 
        prometheus_url: str, 
        job: str
) -> None:
    """
    The polling function for power monitoring that runs in a separate process.

    It periodically collects power consumption data for each GPU and pushes the results
    to the Prometheus Push Gateway.

    Args:
        name: The unique name for the monitoring window.
        pipe: A multiprocessing queue to receive control signals (e.g., "stop").
        power_monitor: The PowerMonitor instance used to retrieve power usage data.
        update_period: The interval (in seconds) between power data polls.
        prometheus_url: The URL of the Prometheus Push Gateway where metrics are pushed.
        job: The job name used in Prometheus for Gauge metrics.
    """
    gpu_gauges = {}
    registry = CollectorRegistry()

    for gpu_index in power_monitor.gpu_indices:
        gpu_gauges[gpu_index] = Gauge(
            f'power_monitor_gpu_{gpu_index}_power_watts',  
            f'Records power consumption for GPU {gpu_index} over time',
            ['gpu_index'],  # Label to indicate GPU index
            registry=registry
        )

    while True:
        if not pipe.empty():
            signal = pipe.get()
            if signal == "stop":
                break  

        power_measurement = power_monitor.get_power()
        if power_measurement is not None:
            for gpu_index, power_value in power_measurement:
                gpu_gauges[gpu_index].labels(gpu_index=f"{name}_gpu{gpu_index}").set(power_value)

        push_to_gateway(prometheus_url, job=job, registry=registry)
        time.sleep(update_period)

