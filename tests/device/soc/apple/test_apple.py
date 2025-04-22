import pytest
import sys
from unittest.mock import patch, MagicMock
from dataclasses import asdict


@pytest.fixture(autouse=True, scope="function")
def mock_optional_dep():
    mocked_import = MagicMock()
    mocked_import.AppleEnergyMonitor = MagicMock()
    mocked_import.AppleEnergyMetrics = MagicMock()

    with (
        patch.dict(sys.modules, {"zeus_apple_silicon": mocked_import}),
        patch("sys.platform", "darwin"),
        patch("platform.processor", return_value="arm"),
    ):
        yield


@patch("zeus.device.soc.apple.AppleEnergyMonitor")
def test_total_energy(mocked_energy_monitor):

    # These imports must happen at each test (i.e., function) instead of at
    # the top of the file because they have an optional dependency that must
    # be mocked. The `mock_optional_dep` fixture mocks the optional dependency
    # (zeus_apple_silicon) for all test functions.
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    # Mocking `AppleEnergyMonitor`, not `AppleSilicon`
    mock_monitor = MagicMock()
    mocked_energy_monitor.return_value = mock_monitor

    # Mocking `AppleEnergyMetrics`, not `AppleSiliconMeasurement`
    mock_metrics = MagicMock(
        cpu_total_mj=123,
        efficiency_cores_mj=[1, 2, 3],
        performance_cores_mj=[11, 22, 33],
        efficiency_core_manager_mj=123,
        performance_core_manager_mj=123,
        dram_mj=123,
        gpu_mj=123,
        gpu_sram_mj=123,
        ane_mj=123,
    )

    # Specifying return value of `get_cumulative_energy` in `AppleEnergyMonitor`
    mock_monitor.get_cumulative_energy.return_value = mock_metrics

    monitor = AppleSilicon()
    res = monitor.getTotalEnergyConsumption()

    res_dict = asdict(res)
    expected = {
        "cpu_total_mj": 123,
        "efficiency_cores_mj": [1, 2, 3],
        "performance_cores_mj": [11, 22, 33],
        "efficiency_core_manager_mj": 123,
        "performance_core_manager_mj": 123,
        "dram_mj": 123,
        "gpu_mj": 123,
        "gpu_sram_mj": 123,
        "ane_mj": 123,
    }

    assert res_dict == expected


@patch("zeus.device.soc.apple.AppleEnergyMonitor")
def test_interval_energy(mocked_energy_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    # Mocking `AppleEnergyMonitor`, not `AppleSilicon`
    mock_monitor = MagicMock()
    mocked_energy_monitor.return_value = mock_monitor

    # For `end_window`
    mock_metrics = MagicMock(
        cpu_total_mj=100,
        efficiency_cores_mj=[100, 100, 100],
        performance_cores_mj=[220, 220, 220],
        efficiency_core_manager_mj=10,
        performance_core_manager_mj=20,
        dram_mj=None,
        gpu_mj=200,
        gpu_sram_mj=None,
        ane_mj=None,
    )
    mock_monitor.end_window.return_value = mock_metrics

    monitor = AppleSilicon()
    monitor.beginWindow("test")
    res = monitor.endWindow("test")

    res_dict = asdict(res)
    expected = {
        "cpu_total_mj": 100,
        "efficiency_cores_mj": [100, 100, 100],
        "performance_cores_mj": [220, 220, 220],
        "efficiency_core_manager_mj": 10,
        "performance_core_manager_mj": 20,
        "dram_mj": None,
        "gpu_mj": 200,
        "gpu_sram_mj": None,
        "ane_mj": None,
    }

    assert res_dict == expected


@patch("zeus.device.soc.apple.AppleEnergyMonitor")
def test_overlapping_interval_energy(mocked_energy_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    # Mocking `AppleEnergyMonitor`, not `AppleSilicon`
    mock_monitor = MagicMock()
    mocked_energy_monitor.return_value = mock_monitor

    # For first `end_window`
    mock_metrics = MagicMock(
        cpu_total_mj=100,
        efficiency_cores_mj=[100, 100, 100],
        performance_cores_mj=[220, 220, 220],
        efficiency_core_manager_mj=10,
        performance_core_manager_mj=20,
        dram_mj=None,
        gpu_mj=200,
        gpu_sram_mj=None,
        ane_mj=None,
    )
    mock_monitor.end_window.return_value = mock_metrics

    monitor = AppleSilicon()
    monitor.beginWindow("test1")
    monitor.beginWindow("test2")

    res = monitor.endWindow("test1")
    res_dict = asdict(res)
    expected = {
        "cpu_total_mj": 100,
        "efficiency_cores_mj": [100, 100, 100],
        "performance_cores_mj": [220, 220, 220],
        "efficiency_core_manager_mj": 10,
        "performance_core_manager_mj": 20,
        "dram_mj": None,
        "gpu_mj": 200,
        "gpu_sram_mj": None,
        "ane_mj": None,
    }
    assert res_dict == expected

    # For second `end_window`
    mock_metrics = MagicMock(
        cpu_total_mj=200,
        efficiency_cores_mj=[200, 100, 100],
        performance_cores_mj=[220, 220, 220],
        efficiency_core_manager_mj=10,
        performance_core_manager_mj=20,
        dram_mj=None,
        gpu_mj=800,
        gpu_sram_mj=None,
        ane_mj=None,
    )
    mock_monitor.end_window.return_value = mock_metrics

    res = monitor.endWindow("test2")
    res_dict = asdict(res)
    expected = {
        "cpu_total_mj": 200,
        "efficiency_cores_mj": [200, 100, 100],
        "performance_cores_mj": [220, 220, 220],
        "efficiency_core_manager_mj": 10,
        "performance_core_manager_mj": 20,
        "dram_mj": None,
        "gpu_mj": 800,
        "gpu_sram_mj": None,
        "ane_mj": None,
    }
    assert res_dict == expected


@patch("zeus.device.soc.apple.AppleEnergyMonitor")
def test_available_metrics(mocked_energy_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    # Mocking `AppleEnergyMonitor`, not `AppleSilicon`
    mock_monitor = MagicMock()
    mocked_energy_monitor.return_value = mock_monitor

    # `AppleSilicon::getAvailableMetrics` relies on `AppleEnergyMonitor::get_cumulative_energy`.
    mock_metrics = MagicMock(
        cpu_total_mj=100,
        efficiency_cores_mj=[100, 100, 100],
        performance_cores_mj=None,
        efficiency_core_manager_mj=None,
        performance_core_manager_mj=20,
        dram_mj=None,
        gpu_mj=200,
        gpu_sram_mj=None,
        ane_mj=None,
    )
    mock_monitor.get_cumulative_energy.return_value = mock_metrics

    monitor = AppleSilicon()
    available = monitor.getAvailableMetrics()
    expected = {
        "soc.cpu_total_mj",
        "soc.efficiency_cores_mj",
        "soc.performance_core_manager_mj",
        "soc.gpu_mj",
    }

    assert available == expected


def test_metrics_subtraction():
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    metrics1 = AppleSiliconMeasurement(
        10, [10, 10, 10], [1, 1, 1], [30], 30, 5, 5, None, 5
    )
    metrics2 = AppleSiliconMeasurement(
        20, [100, 110, 100], [1, 1], 30, 40, 5, None, None, 10
    )

    diff = metrics2 - metrics1
    assert diff.cpu_total_mj == 10
    assert diff.efficiency_cores_mj == [90, 100, 90]
    assert diff.performance_cores_mj == None
    assert diff.efficiency_core_manager_mj == None
    assert diff.performance_core_manager_mj == 10
    assert diff.dram_mj == 0
    assert diff.gpu_mj is None
    assert diff.gpu_sram_mj is None
    assert diff.ane_mj == 5


def test_metrics_zero_out():
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    metrics = AppleSiliconMeasurement(10, [10, 10, 10], None, 30, 30, 5, None, None, 5)
    metrics.zeroAllFields()

    assert metrics.cpu_total_mj == 0
    assert metrics.efficiency_cores_mj == []
    assert metrics.performance_cores_mj == []
    assert metrics.efficiency_core_manager_mj == 0
    assert metrics.performance_core_manager_mj == 0
    assert metrics.dram_mj == 0
    assert metrics.gpu_mj == 0
    assert metrics.gpu_sram_mj == 0
    assert metrics.ane_mj == 0
