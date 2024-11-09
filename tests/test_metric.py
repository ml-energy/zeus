from __future__ import annotations
from unittest.mock import patch

import pytest
from zeus.metric import EnergyHistogram, EnergyCumulativeCounter, PowerGauge

class MockMeasurement:
    """
    Mock object representing energy measurements for testing.
    Contains energy values for GPU, CPU, and DRAM.
    """
    def __init__(self, gpu_energy=None, cpu_energy=None, dram_energy=None):
        self.gpu_energy = gpu_energy or {}
        self.cpu_energy = cpu_energy or {}
        self.dram_energy = dram_energy or {}

class MockZeusMonitor:
    """
    Mock object to simulate an ZeusMonitor, which provides energy measurements
    for GPU, CPU, and DRAM for use in unit testing. The measurement values are fixed 
    to aid in validating the functionality of histogram metrics.
    """
    def __init__(self):
        self.gpu_indices = [0, 1, 2, 3]  # 4 GPUs in the mock
        self.cpu_indices = [0, 1]  # 2 CPUs in the mock
        self.dram_indices = [0, 1]

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
            gpu_energy={index: 30.0 for index in self.gpu_indices},  # Fixed value for all GPUs
            cpu_energy={index: 15.0 for index in self.cpu_indices},  # Fixed value for all CPUs
            dram_energy={index: 7.5 for index in self.dram_indices}  # Fixed value for all DRAMs
        )
        
class MockPowerMonitor:
    """
    Mock object to simulate a PowerMonitor, which provides power measurements for GPUs.
    The power values are randomized for testing purposes.
    """
    def __init__(self):
        self.gpu_indices = [0, 1, 2, 3]  # 4 GPUs

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

@pytest.fixture
def mock_energy_monitor():
    """
    Returns a mocked energy monitor instance for testing.
    """
    return MockZeusMonitor()

@pytest.fixture
def mock_power_monitor():
    """
    Returns a mocked power monitor instance for testing.
    """
    return MockPowerMonitor()

# Test Cases

def test_energy_histogram(mock_energy_monitor):
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
    histogram_metric = EnergyHistogram(
        energy_monitor=mock_energy_monitor,
        prometheus_url='http://localhost:9091',
        job='test_energy_histogram',
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

def test_energy_cumulative_counter(mock_energy_monitor):
    """
    Unit test for the EnergyCumulativeCounter class. This test ensures that the
    cumulative energy counter starts and stops correctly, and that the energy
    monitoring process is alive during the window, and that the 'inc' method
    of the Prometheus Counter is called with the expected incremental energy values for
    GPU, CPU, and DRAM.
    """
    cumulative_counter_metric = EnergyCumulativeCounter(
        energy_monitor=mock_energy_monitor,
        update_period=2,
        prometheus_url='http://localhost:9091',
        job='test_energy_counter'
    )

    # Start the window and check the process
    cumulative_counter_metric.begin_window("counter_test")
    assert cumulative_counter_metric.proc is not None
    assert cumulative_counter_metric.proc.is_alive()  # Check if the process is running

    for gpu_index in cumulative_counter_metric.gpu_counters.keys():
        with patch.object(cumulative_counter_metric.gpu_counters[gpu_index], 'inc') as mock_set:
            for call_args in mock_set.return_value.labels.return_value.set.call_args_list:
                observed_value = call_args[0][0]
                assert observed_value == 30.0

    for cpu_index in cumulative_counter_metric.cpu_counters.keys():
        with patch.object(cumulative_counter_metric.cpu_counters[cpu_index], 'inc') as mock_set:
            for call_args in mock_set.return_value.labels.return_value.set.call_args_list:
                observed_value = call_args[0][0]
                assert observed_value == 15.0

    for dram_index in cumulative_counter_metric.dram_counters.keys():
        with patch.object(cumulative_counter_metric.dram_counters[dram_index], 'inc') as mock_set:
            for call_args in mock_set.return_value.labels.return_value.set.call_args_list:
                observed_value = call_args[0][0]
                assert observed_value == 7.5

    # End the window and ensure the process has stopped
    cumulative_counter_metric.end_window("counter_test")
    cumulative_counter_metric.proc.join()  # Ensure the process has finished
    assert not cumulative_counter_metric.proc.is_alive()  # Process should be done

def test_power_gauge(mock_power_monitor):
    """
    Unit test for the PowerGauge class. This test checks that the power gauge
    measurement process starts and stops correctly, and that the mock power monitor
    provides valid power measurements during the window, and that the 'set' method
    of the Prometheus Gauge is called with the expected power values for GPU.
    """
    
    power_gauge_metric = PowerGauge(
            power_monitor=mock_power_monitor,
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
