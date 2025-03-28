"""Test metric.py."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from zeus.metric import EnergyHistogram, EnergyCumulativeCounter, PowerGauge


@pytest.fixture
def mock_get_cpus():
    """Fixture to mock `get_cpus()` to avoid RAPL-related errors."""
    with patch("zeus.metric.get_cpus", autospec=True) as mock_get_cpus:
        mock_cpu_0 = MagicMock()
        mock_cpu_0.supportsGetDramEnergyConsumption.return_value = True
        mock_cpu_1 = MagicMock()
        mock_cpu_1.supportsGetDramEnergyConsumption.return_value = False

        mock_get_cpus.return_value.cpus = [mock_cpu_0, mock_cpu_1]
        yield mock_get_cpus


@pytest.fixture
def mock_zeus_monitor():
    """Fixture to mock ZeusMonitor behavior."""
    with patch("zeus.metric.ZeusMonitor", autospec=True) as zeus_monitor:
        mock_instance = zeus_monitor.return_value
        mock_instance.end_window.return_value = MagicMock(
            gpu_energy={0: 50.0, 1: 100.0, 2: 200.0},
            cpu_energy={0: 40.0, 1: 50.0},
            dram_energy={0: 10},
        )
        mock_instance.gpu_indices = [0, 1, 2]
        mock_instance.cpu_indices = [0, 1]
        yield mock_instance


@pytest.fixture
def mock_power_monitor():
    """Fixture to mock PowerMonitor."""
    with patch("zeus.metric.PowerMonitor", autospec=True) as power_monitor:
        mock_instance = power_monitor.return_value
        mock_instance.get_power.return_value = {
            0: 300.0,
            1: 310.0,
            2: 320.0,
        }
        yield mock_instance


@pytest.fixture
def mock_histogram():
    """Fixture to mock Prometheus Histogram creation.

    Mocks the Histogram functionality to avoid real Prometheus interactions
    and to validate histogram-related method calls.
    """
    with patch("zeus.metric.Histogram", autospec=True) as histogram:
        yield histogram


def test_energy_histogram(
    mock_get_cpus: MagicMock,
    mock_zeus_monitor: MagicMock,
    mock_histogram: MagicMock,
) -> None:
    """Test EnergyHistogram class."""
    cpu_indices = [0, 1]
    gpu_indices = [0, 1, 2]
    pushgateway_url = "http://localhost:9091"
    window_name = "test_window"

    # Ensure mocked CPUs have the required method
    mock_get_cpus.return_value.cpus = [
        MagicMock(supportsGetDramEnergyConsumption=MagicMock(return_value=True)),
        MagicMock(supportsGetDramEnergyConsumption=MagicMock(return_value=False)),
    ]

    histogram_metric = EnergyHistogram(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        pushgateway_url=pushgateway_url,
        job="test_energy_histogram",
    )

    # Mock single Histogram objects for GPU, CPU, and DRAM
    gpu_mock_histogram = mock_histogram(
        name="gpu_energy_histogram",
        documentation="Mocked GPU histogram",
        labelnames=["window", "index"],
    )

    cpu_mock_histogram = mock_histogram(
        name="cpu_energy_histogram",
        documentation="Mocked CPU histogram",
        labelnames=["window", "index"],
    )

    dram_mock_histogram = mock_histogram(
        name="dram_energy_histogram",
        documentation="Mocked DRAM histogram",
        labelnames=["window", "index"],
    )

    # Attach mocked histograms to the metric
    histogram_metric.gpu_histograms = gpu_mock_histogram
    histogram_metric.cpu_histograms = cpu_mock_histogram
    histogram_metric.dram_histograms = dram_mock_histogram

    # Begin and end the monitoring window
    histogram_metric.begin_window(window_name, False)
    with patch("http.client.HTTPConnection", autospec=True) as mock_http:
        mock_http_instance = mock_http.return_value
        mock_http_instance.getresponse.return_value.code = 200
        mock_http_instance.getresponse.return_value.msg = "OK"
        mock_http_instance.getresponse.return_value.info = lambda: {}
        mock_http_instance.sock = MagicMock()
        histogram_metric.end_window(window_name, False)

    # Validate GPU histogram observations
    for (
        gpu_index,
        energy,
    ) in mock_zeus_monitor.return_value.end_window.return_value.gpu_energy.items():
        gpu_mock_histogram.labels.assert_any_call(window=window_name, index=gpu_index)
        gpu_mock_histogram.labels.return_value.observe.assert_any_call(energy)

    # Validate CPU histogram observations
    for (
        cpu_index,
        energy,
    ) in mock_zeus_monitor.return_value.end_window.return_value.cpu_energy.items():
        cpu_mock_histogram.labels.assert_any_call(window=window_name, index=cpu_index)
        cpu_mock_histogram.labels.return_value.observe.assert_any_call(energy)

    # Validate DRAM histogram observations
    for (
        dram_index,
        energy,
    ) in mock_zeus_monitor.return_value.end_window.return_value.dram_energy.items():
        dram_mock_histogram.labels.assert_any_call(window=window_name, index=dram_index)
        dram_mock_histogram.labels.return_value.observe.assert_any_call(energy)


@patch("zeus.metric.energy_monitoring_loop", autospec=True)
@patch("zeus.metric.mp.get_context", autospec=True)
def test_energy_cumulative_counter(
    mock_mp_context: MagicMock,
    mock_energy_monitoring_loop: MagicMock,
):
    """Test EnergyCumulativeCounter with mocked subprocess behavior."""
    cpu_indices = [0, 1]
    gpu_indices = [0, 1, 2]
    pushgateway_url = "http://localhost:9091"

    # Mock the context and queue
    mock_queue = MagicMock()
    mock_process = MagicMock()
    mock_mp_context.return_value.Queue.return_value = mock_queue
    mock_mp_context.return_value.Process.return_value = (
        mock_process  # Ensure Process returns mock_process
    )

    # Mock the behavior of subprocess
    mock_energy_monitoring_loop.return_value = (
        None  # Simulate the subprocess running without errors
    )

    # Create the EnergyCumulativeCounter instance
    cumulative_counter = EnergyCumulativeCounter(
        cpu_indices=cpu_indices,
        gpu_indices=gpu_indices,
        update_period=2,
        pushgateway_url=pushgateway_url,
        job="test_energy_counter",
    )

    # Begin a monitoring window
    cumulative_counter.begin_window("test_counter")

    # Assert that the subprocess was started with the correct arguments
    mock_mp_context.return_value.Process.assert_called_once_with(
        target=mock_energy_monitoring_loop,
        args=(
            "test_counter",
            mock_queue,
            cpu_indices,
            gpu_indices,
            2,
            pushgateway_url,
            "test_energy_counter",
        ),
    )
    mock_process.start.assert_called_once()

    # Assert the window state is updated correctly
    assert "test_counter" in cumulative_counter.window_state
    state = cumulative_counter.window_state["test_counter"]
    assert state.queue == mock_queue
    assert state.proc == mock_process

    # End the monitoring window
    cumulative_counter.end_window("test_counter")
    mock_queue.put.assert_called_once_with("stop")
    mock_process.join.assert_called_once()
    assert "test_counter" not in cumulative_counter.window_state


@patch("zeus.metric.power_monitoring_loop", autospec=True)
@patch("zeus.metric.mp.get_context", autospec=True)
def test_power_gauge(
    mock_mp_context: MagicMock,
    mock_power_monitoring_loop: MagicMock,
):
    """Test PowerGauge with mocked subprocess behavior."""
    gpu_indices = [0, 1, 2]
    pushgateway_url = "http://localhost:9091"

    # Mock the context and queue
    mock_queue = MagicMock()
    mock_process = MagicMock()
    mock_mp_context.return_value.Queue.return_value = mock_queue
    mock_mp_context.return_value.Process.return_value = mock_process

    # Mock the behavior of subprocess
    mock_power_monitoring_loop.return_value = (
        None  # Simulate the subprocess running without errors
    )

    # Create the EnergyCumulativeCounter instance
    power_gauge = PowerGauge(
        gpu_indices=gpu_indices,
        update_period=2,
        pushgateway_url=pushgateway_url,
        job="test_power_gauge",
    )

    # Begin a monitoring window
    power_gauge.begin_window("test_power_gauge")

    # Assert that the subprocess was started with the correct arguments
    mock_mp_context.return_value.Process.assert_called_once_with(
        target=mock_power_monitoring_loop,
        args=(
            "test_power_gauge",
            mock_queue,
            gpu_indices,
            2,
            pushgateway_url,
            "test_power_gauge",
        ),
    )
    mock_process.start.assert_called_once()

    # Assert the window state is updated correctly
    assert "test_power_gauge" in power_gauge.window_state
    state = power_gauge.window_state["test_power_gauge"]
    assert state.queue == mock_queue
    assert state.proc == mock_process

    # End the monitoring window
    power_gauge.end_window("test_power_gauge")
    mock_queue.put.assert_called_once_with("stop")
    mock_process.join.assert_called_once()
    assert "test_power_gauge" not in power_gauge.window_state
