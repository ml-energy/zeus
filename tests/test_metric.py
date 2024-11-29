from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from zeus.metric import EnergyHistogram, EnergyCumulativeCounter, PowerGauge, energy_monitoring_loop

import multiprocessing as mp

class MockMeasurement:
    """
    Mock object representing energy measurements for testing.
    Contains energy values for GPU, CPU, and DRAM.
    """
    def __init__(self, gpu_energy=None, cpu_energy=None, dram_energy=None):
        self.gpu_energy = gpu_energy or {}
        self.cpu_energy = cpu_energy or {}
        self.dram_energy = dram_energy or {}

@pytest.fixture
def mock_gpu_indices():
    """Mock GPU indices for testing."""
    return [0, 1, 2, 3]  # 4 GPUs

@pytest.fixture
def mock_cpu_indices():
    """Mock CPU indices for testing."""
    return [0, 1]  # 2 CPUs

class MockZeusMonitor:
    """
    Mock object to simulate an ZeusMonitor, which provides energy measurements
    for GPU, CPU, and DRAM for use in unit testing. The measurement values are fixed 
    to aid in validating the functionality of histogram metrics.
    """
    def __init__(self):
        self.gpu_indices = mock_gpu_indices
        self.cpu_indices = mock_cpu_indices
        self.dram_indices = mock_cpu_indices

    def begin_window(self, name):
        """
        Simulates the start of a measurement window.
        """

    def end_window(self, name: str) -> MockMeasurement:
        """
        Simulates the end of a measurement window, returning fixed energy measurements 
        for testing purposes.
        """
        return MockMeasurement(
            gpu_energy={index: 30.0 for index in self.gpu_indices},  
            cpu_energy={index: 15.0 for index in self.cpu_indices},  
            dram_energy={index: 7.5 for index in self.dram_indices}  
        )
        
class MockPowerMonitor:
    """
    Mock object to simulate a PowerMonitor, which provides power measurements for GPUs.
    The power values are randomized for testing purposes.
    """
    def __init__(self):
        self.gpu_indices = mock_gpu_indices

    def begin_window(self, name):
        """
        Simulates the start of a power measurement window.
        """
    def get_power(self):
        """
        Returns simulated power measurements for each GPU.
        """
        return [(index, 300.0) for index in self.gpu_indices]

    def end_window(self, name):
        """
        Simulates the start of a power measurement window.
        """
        print(f"MockPowerMonitor: end window {name}")

# @pytest.fixture
# def mock_energy_monitor():
#     """
#     Returns a mocked energy monitor instance for testing.
#     """
#     return MockZeusMonitor()

# @pytest.fixture
# def mock_power_monitor():
#     """
#     Returns a mocked power monitor instance for testing.
#     """
#     return MockPowerMonitor()

# Test Cases
@patch("zeus.metric.ZeusMonitor", autospec=True)
def test_energy_histogram(MockZeusMonitor, mock_gpu_indices, mock_cpu_indices):
    """
    Unit test for the EnergyHistogram class. This test validates that the `observe()` 
    method on the Prometheus Histogram is correctly called with the fixed GPU, CPU, and 
    DRAM energy values (30.0, 15.0, and 7.5, respectively).
    """
    # Define custom bucket ranges for GPU, CPU, and DRAM energy histograms
    custom_bucket_ranges = {
        "gpu": [10.0, 25.0, 50.0],
        "cpu": [5.0, 10.0, 25.0],
        "dram": [1.0, 2.5, 10.0]
    }

    # Instantiate the EnergyHistogram class with the mock energy monitor and custom bucket ranges
    # with patch("zeus.metric.ZeusMonitor", autospec=True) as MockZeusMonitor:
    mock_monitor_instance = MockZeusMonitor.return_value
    mock_monitor_instance.gpu_indices = mock_gpu_indices
    mock_monitor_instance.cpu_indices = mock_cpu_indices
    mock_monitor_instance.end_window.return_value = MockMeasurement(
        gpu_energy={index: 30.0 for index in mock_gpu_indices},
        cpu_energy={index: 15.0 for index in mock_cpu_indices},
        dram_energy={index: 7.5 for index in mock_cpu_indices},
    )

    histogram_metric = EnergyHistogram(
        cpu_indices=mock_cpu_indices,
        gpu_indices=mock_gpu_indices,
        prometheus_url="http://localhost:9091",
        job="test_energy_histogram",
        bucket_ranges=custom_bucket_ranges
    )

    # Test GPU energy observations
    for gpu_index in histogram_metric.gpu_histograms.keys():
        with patch.object(histogram_metric.gpu_histograms[gpu_index], 'observe') as mock_observe_gpu:

            histogram_metric.begin_window("test_window")
            histogram_metric.end_window("test_window")

            for call_args in mock_observe_gpu.call_args_list:
                observed_value = call_args[0][0]  
                assert observed_value == 30.0
    # Test CPU energy observations
    for cpu_index in histogram_metric.cpu_histograms.keys():
        with patch.object(histogram_metric.cpu_histograms[cpu_index], 'observe') as mock_observe_cpu:
            # Check that `observe()` was called with the correct CPU energy value
            histogram_metric.begin_window("test_window")
            histogram_metric.end_window("test_window")

            for call_args in mock_observe_cpu.call_args_list:
                observed_value = call_args[0][0]  
                assert observed_value == 15.0
    # Test DRAM energy observations
    for dram_index in histogram_metric.dram_histograms.keys():
        with patch.object(histogram_metric.dram_histograms[dram_index], 'observe') as mock_observe_dram:
            # Check that `observe()` was called with the correct DRAM energy value
            histogram_metric.begin_window("test_window")
            histogram_metric.end_window("test_window")

            for call_args in mock_observe_dram.call_args_list:
                observed_value = call_args[0][0]  
                assert observed_value == 7.5
from unittest.mock import patch, MagicMock
import multiprocessing as mp
from zeus.metric import EnergyCumulativeCounter, energy_monitoring_loop


@patch("prometheus_client.Counter")  # Mock Prometheus Counter
def test_energy_cumulative_counter_with_multiprocessing(MockCounter):
    # Mock indices
    mock_gpu_indices = [0, 1, 2, 3]
    mock_cpu_indices = [0, 1]

    # Mock Prometheus Counters
    mock_gpu_counters = {gpu: MagicMock() for gpu in mock_gpu_indices}
    mock_cpu_counters = {cpu: MagicMock() for cpu in mock_cpu_indices}
    mock_dram_counters = {dram: MagicMock() for dram in mock_cpu_indices}

    # Mock Counter side effect
    def mock_counter_side_effect(name, description, labels, registry):
        index = int(name.split("_")[-2])  # Extract index from name
        if "gpu" in name:
            return mock_gpu_counters[index]
        elif "cpu" in name:
            return mock_cpu_counters[index]
        elif "dram" in name:
            return mock_dram_counters[index]

    MockCounter.side_effect = mock_counter_side_effect

    # Mock ZeusMonitor behavior
    def start_energy_monitoring_loop(queue, cpu_indices, gpu_indices):
        with patch("zeus.metric.ZeusMonitor") as MockZeusMonitor:
            mock_monitor_instance = MockZeusMonitor.return_value
            mock_monitor_instance.begin_window = MagicMock()
            mock_monitor_instance.end_window.return_value = MagicMock(
                gpu_energy={index: 30.0 for index in gpu_indices},
                cpu_energy={index: 15.0 for index in cpu_indices},
                dram_energy={index: 7.5 for index in cpu_indices},
            )

            # Call the energy monitoring loop
            energy_monitoring_loop(
                name="counter_test",
                pipe=queue,
                cpu_indices=cpu_indices,
                gpu_indices=gpu_indices,
                update_period=1,
                prometheus_url="http://localhost:9091",
                job="test_energy_counter",
            )

    # Initialize EnergyCumulativeCounter
    cumulative_counter_metric = EnergyCumulativeCounter(
        cpu_indices=mock_cpu_indices,
        gpu_indices=mock_gpu_indices,
        update_period=1,  # Shorter period for faster test
        prometheus_url="http://localhost:9091",
        job="test_energy_counter",
    )

    # Use a real multiprocessing Queue
    queue = mp.Queue()

    # Start the subprocess
    process = mp.Process(
        target=start_energy_monitoring_loop, args=(queue, mock_cpu_indices, mock_gpu_indices)
    )
    process.start()

    # Allow the loop to run for a few iterations
    import time
    time.sleep(3)  # Wait for some time to let the loop run

    # Send the stop signal to end the loop
    queue.put("stop")
    process.join()  # Wait for the process to finish

    # Validate GPU counters
    for gpu_index, mock_counter in mock_gpu_counters.items():
        try:
            mock_counter.labels.assert_called_with(
                window="__EnergyCumulativeCounter_counter_test", index=gpu_index
            )
            mock_counter.labels.return_value.inc.assert_called_with(30.0)
        except AssertionError:
            print(f"Assertion failed for GPU counter {gpu_index}")
            raise

    # Validate CPU counters
    for cpu_index, mock_counter in mock_cpu_counters.items():
        try:
            mock_counter.labels.assert_called_with(
                window="__EnergyCumulativeCounter_counter_test", index=cpu_index
            )
            mock_counter.labels.return_value.inc.assert_called_with(15.0)
        except AssertionError:
            print(f"Assertion failed for CPU counter {cpu_index}")
            raise

    # Validate DRAM counters
    for dram_index, mock_counter in mock_dram_counters.items():
        try:
            mock_counter.labels.assert_called_with(
                window="__EnergyCumulativeCounter_counter_test", index=dram_index
            )
            mock_counter.labels.return_value.inc.assert_called_with(7.5)
        except AssertionError:
            print(f"Assertion failed for DRAM counter {dram_index}")
            raise



def test_power_gauge():
    """
    Unit test for the PowerGauge class. This test checks that the power gauge
    measurement process starts and stops correctly, and that the mock power monitor
    provides valid power measurements during the window, and that the 'set' method
    of the Prometheus Gauge is called with the expected power values for GPU.
    """
    
    power_gauge_metric = PowerGauge(
            # power_monitor=mock_power_monitor,
            gpu_indices=mock_gpu_indices,
            update_period=2,
            prometheus_url='http://localhost:9091',
            job='test_power_gauge'
        )
    
    power_gauge_metric.begin_window("gauge_test")
    assert power_gauge_metric.proc is not None
    assert power_gauge_metric.proc.is_alive()
    
    for gpu_index in power_gauge_metric.gpu_gauges.keys():
        with patch.object(power_gauge_metric.gpu_gauges[gpu_index], 'set') as mock_set:
            for call_args in mock_set.return_value.labels.return_value.set.call_args_list:
                observed_value = call_args[0][0]
                assert observed_value == 300.0

    power_gauge_metric.end_window("gauge_test")
    power_gauge_metric.proc.join()  # Ensure the process has finished
    assert not power_gauge_metric.proc.is_alive()  # Process should be done
