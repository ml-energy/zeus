from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from zeus.metric import EnergyHistogram, EnergyCumulativeCounter, PowerGauge

@pytest.fixture
def mock_get_cpus():
    """Fixture to mock `get_cpus()` to avoid RAPL-related errors."""

    with patch("zeus.metric.get_cpus", autospec=True) as mock_get_cpus:
        mock_cpu = MagicMock()
        mock_cpu.cpus = []  
        mock_get_cpus.return_value = mock_cpu
        yield mock_get_cpus


@pytest.fixture
def mock_zeus_monitor():
    """Fixture to mock ZeusMonitor behavior."""

    with patch("zeus.metric.ZeusMonitor", autospec=True) as MockZeusMonitor:
        mock_instance = MockZeusMonitor.return_value
        mock_instance.end_window.return_value = MagicMock(
            gpu_energy={0: 30.0, 1: 35.0, 2: 40.0}, 
            cpu_energy={0: 20.0, 1: 25.0},  
            dram_energy={},  
        )
        mock_instance.gpu_indices = [0, 1, 2]  
        mock_instance.cpu_indices = [0, 1]  
        yield mock_instance

@pytest.fixture
def mock_histogram():
    """Fixture to mock Prometheus Histogram creation.

    Mocks the Histogram functionality to avoid real Prometheus interactions
    and to validate histogram-related method calls.
    """

    with patch("zeus.metric.Histogram", autospec=True) as MockHistogram:
        yield MockHistogram
        
def test_energy_histogram(mock_get_cpus, mock_zeus_monitor, mock_histogram):
    """Test EnergyHistogram class.

    Validates that GPU, CPU, and DRAM histograms are properly initialized,
    and that the correct energy values are recorded.

    Args:
        mock_get_cpus (MagicMock): Mocked `get_cpus` fixture.
        mock_zeus_monitor (MagicMock): Mocked ZeusMonitor fixture.
        mock_histogram (MagicMock): Mocked Prometheus Histogram fixture.
    """
    
    cpu_indices = [0, 1]   
    gpu_indices = [0, 1, 2]  
    prometheus_url = "http://localhost:9091"

    histogram_metric = EnergyHistogram(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        prometheus_url=prometheus_url,
        job="test_energy_histogram",
    )

    for gpu_index, gpu_histogram in histogram_metric.gpu_histograms.items():
        gpu_histogram.labels = MagicMock(return_value=gpu_histogram)
        gpu_histogram.observe = MagicMock()
    
    for cpu_index, cpu_histogram in histogram_metric.cpu_histograms.items():
        cpu_histogram.labels = MagicMock(return_value=cpu_histogram)
        cpu_histogram.observe = MagicMock()

    for dram_index, dram_histogram in histogram_metric.dram_histograms.items():
        dram_histogram.labels = MagicMock(return_value=dram_histogram)
        dram_histogram.observe = MagicMock()

    histogram_metric.begin_window("test_window")
    histogram_metric.end_window("test_window")

    # Assert GPU histograms were observed
    for gpu_index, energy in mock_zeus_monitor.return_value.end_window.return_value.gpu_energy.items():
        calls = [call[0][0] for call in histogram_metric.gpu_histograms[gpu_index].observe.call_args_list]
        print(f"Observed calls for GPU {gpu_index}: {calls}")
        assert energy in calls, f"Expected {energy} in {calls}"
    
    # Assert CPU histograms were observed
    for cpu_index, energy in mock_zeus_monitor.return_value.end_window.return_value.cpu_energy.items():
        calls = [call[0][0] for call in histogram_metric.cpu_histograms[cpu_index].observe.call_args_list]
        print(f"Observed CPU calls for CPU {cpu_index}: {calls}")
        assert energy in calls, f"Expected CPU energy {energy} in {calls}"
  
    # Assert DRAM histograms were observed
    for dram_index, energy in mock_zeus_monitor.return_value.end_window.return_value.dram_energy.items():
        calls = [call[0][0] for call in histogram_metric.dram_histograms[dram_index].observe.call_args_list]
        print(f"Observed DRAM calls for CPU {dram_index}: {calls}")
        assert energy in calls, f"Expected DRAM energy {energy} in {calls}"

def test_energy_cumulative_counter(mock_get_cpus, mock_zeus_monitor):
    """Test EnergyCumulativeCounter with mocked ZeusMonitor.

    Args:
        mock_get_cpus (MagicMock): Mocked `get_cpus` fixture.
        mock_zeus_monitor (MagicMock): Mocked ZeusMonitor fixture.    
    """

    cpu_indices = [0, 1]
    gpu_indices = [0, 1, 2]
    prometheus_url = "http://localhost:9091"

    cumulative_counter = EnergyCumulativeCounter(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        update_period=2,
        prometheus_url=prometheus_url,
        job="test_energy_counter",
    )

    for counters in [
        cumulative_counter.gpu_counters,
        cumulative_counter.cpu_counters,
    ]:
        for counter in counters.values():
            counter.labels = MagicMock(return_value=counter)
            counter.inc = MagicMock()

    cumulative_counter.begin_window("test_counter")
    cumulative_counter.end_window("test_counter")

    # Assert GPU counters
    for gpu_index, energy in mock_zeus_monitor.return_value.end_window.return_value.gpu_energy.items():
        assert gpu_index in cumulative_counter.gpu_counters, f"GPU counter for index {gpu_index} not initialized"
        cumulative_counter.gpu_counters[gpu_index].inc.assert_called_with(energy)

    # Assert CPU counters
    for cpu_index, energy in mock_zeus_monitor.return_value.end_window.return_value.cpu_energy.items():
        assert cpu_index in cumulative_counter.cpu_counters, f"CPU counter for index {cpu_index} not initialized"
        cumulative_counter.cpu_counters[cpu_index].inc.assert_called_with(energy)

@pytest.fixture
def mock_power_monitor():
    """Fixture to mock PowerMonitor."""

    with patch("zeus.metric.PowerMonitor", autospec=True) as MockPowerMonitor:
        mock_instance = MockPowerMonitor.return_value
        mock_instance.get_power.return_value = {
            0: 300.0,
            1: 310.0,
            2: 320.0,
        }
        yield mock_instance

@pytest.fixture
def mock_gauge():
    """Fixture to mock Prometheus Gauge creation."""
    
    with patch("zeus.metric.Gauge", autospec=True) as MockGauge:
        MockGauge.side_effect = lambda *args, **kwargs: MagicMock()
        yield MockGauge

@patch("prometheus_client.push_to_gateway")
@patch("zeus.device.gpu.get_gpus")
def test_power_gauge(
    mock_get_gpus: MagicMock, 
    mock_power_monitor: MagicMock, 
    mock_gauge: MagicMock,
    mock_push_to_gateway
) -> None:
    """Test PowerGauge with mocked PowerMonitor and Prometheus Gauges.

    Args:
        mock_get_gpus (MagicMock): Mocked `get_gpus` function to simulate available GPUs.
        mock_power_monitor (MagicMock): Mocked PowerMonitor to simulate GPU power data.
        mock_gauge (MagicMock): Mocked Prometheus Gauge creation.
    """

    gpu_indices = [0, 1, 2]
    prometheus_url = "http://localhost:9091"
    mock_push_to_gateway.return_value = None

    # Mock `get_gpus` to simulate available GPUs
    mock_get_gpus.return_value = MagicMock()
    mock_get_gpus.return_value.gpus = gpu_indices

    mock_gauge.side_effect = lambda *args, **kwargs: MagicMock()

    power_gauge = PowerGauge(
        gpu_indices=gpu_indices,
        update_period=2,
        prometheus_url=prometheus_url,
        job="test_power_gauge",
    )
    for gpu_index, gauge in power_gauge.gpu_gauges.items():
        gauge.labels = MagicMock(return_value=gauge)
        gauge.set = MagicMock()

    power_gauge.begin_window("test_power_window")
    power_gauge.end_window("test_power_window")

    # Assert that the gauges were set with the correct power values
    for gpu_index, power_value in mock_power_monitor.return_value.get_power.return_value.items():
        try:
            # Check if `labels` was called with the correct arguments
            power_gauge.gpu_gauges[gpu_index].labels.assert_called_once_with(
                gpu_index=gpu_index, window="test_power_window"
            )
            power_gauge.gpu_gauges[gpu_index].set.assert_called_once_with(power_value)
        except AssertionError as e:
            print(f"AssertionError for GPU {gpu_index}:")
            raise e
        
