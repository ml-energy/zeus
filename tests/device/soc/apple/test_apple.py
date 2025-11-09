import pytest
import sys
from unittest.mock import patch, MagicMock
from dataclasses import asdict


@pytest.fixture(scope="function")
def mock_monitor():
    mocked_import = MagicMock()
    monitor = MagicMock()
    mocked_import.AppleEnergyMonitor.return_value = monitor
    mocked_import.AppleEnergyMetrics = MagicMock()

    # When tests in other files import zeus.device.soc, that will
    # cause zeus.device.soc.apple to be cached in sys.modules
    # because zeus.device.soc imports zeus.device.soc.apple.
    # Once cached, whenever we try to import zeus.device.soc.apple
    # in *this* test file, the cached version will be used. However,
    # since we want to *replace* zeus_apple_silicon which is used
    # inside zeus.device.soc.apple with our own new mocked version,
    # we must remove zeus.device.soc.apple from the cache and have it
    # later be reloaded with our mocked version of zeus_apple_silicon.
    sys.modules.pop("zeus.device.soc.apple", None)

    with (
        patch.dict(sys.modules, {"zeus_apple_silicon": mocked_import}),
        patch("sys.platform", "darwin"),
        patch("platform.processor", return_value="arm"),
    ):
        yield monitor


def test_total_energy(mock_monitor):
    # These imports must happen at each test (i.e., function) instead of at
    # the top of the file because they have an optional dependency that must
    # be mocked. The `mock_monitor` fixture mocks the optional dependency
    # (zeus_apple_silicon) for all test functions.
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

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
    res = monitor.get_total_energy_consumption()

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


def test_interval_energy(mock_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

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
    monitor.begin_window("test")
    res = monitor.end_window("test")

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


def test_overlapping_interval_energy(mock_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

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
    monitor.begin_window("test1")
    monitor.begin_window("test2")

    res = monitor.end_window("test1")
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

    res = monitor.end_window("test2")
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


def test_available_metrics(mock_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

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
    available = monitor.get_available_metrics()
    expected = {
        "cpu_total_mj",
        "efficiency_cores_mj",
        "performance_core_manager_mj",
        "gpu_mj",
    }

    assert available == expected


def test_metrics_subtraction(mock_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    metrics1 = AppleSiliconMeasurement(10, [10, 10, 10], [1, 0, 1], 10, 30, 5, None, None, 5)
    metrics2 = AppleSiliconMeasurement(20, [100, 110, 100], [1, 1, 1], 30, 40, 5, None, None, 10)
    diff = metrics2 - metrics1
    assert diff.cpu_total_mj == 10
    assert diff.efficiency_cores_mj == [90, 100, 90]
    assert diff.performance_cores_mj == [0, 1, 0]
    assert diff.efficiency_core_manager_mj == 20
    assert diff.performance_core_manager_mj == 10
    assert diff.dram_mj == 0
    assert diff.gpu_mj is None
    assert diff.gpu_sram_mj is None
    assert diff.ane_mj == 5


def test_metrics_bad_subtraction(mock_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    # Differing list lengths
    metrics1 = AppleSiliconMeasurement(10, [10, 10, 10], [1, 1], 10, 30, 5, None, None, 5)
    metrics2 = AppleSiliconMeasurement(20, [100, 110, 100], [1, 1, 1], 30, 40, 5, None, None, 10)
    with pytest.raises(ValueError):
        diff = metrics2 - metrics1

    # Differing field types
    metrics1 = AppleSiliconMeasurement(10, [10, 10, 10], [1, 1], 10, 30, 5, None, None, 5)
    metrics2 = AppleSiliconMeasurement([20], [100, 110, 100], [1, 1], 30, 40, 5, None, None, 10)
    with pytest.raises(ValueError):
        diff = metrics2 - metrics1

    # (None) - (not None)
    metrics1 = AppleSiliconMeasurement(10, [10, 10, 10], [1, 1], 10, 30, 5, None, None, 5)
    metrics2 = AppleSiliconMeasurement(None, [100, 110, 100], [1, 1], 30, 40, 5, None, None, 10)
    with pytest.raises(ValueError):
        diff = metrics2 - metrics1

    # (not None) - (None)
    metrics1 = AppleSiliconMeasurement(None, [10, 10, 10], [1, 1], 10, 30, 5, None, None, None)
    metrics2 = AppleSiliconMeasurement(None, [100, 110, 100], [1, 1], 30, 40, 5, None, None, 10)
    with pytest.raises(ValueError):
        diff = metrics2 - metrics1


def test_metrics_zero_out(mock_monitor):
    from zeus.device.soc.apple import AppleSilicon, AppleSiliconMeasurement

    metrics = AppleSiliconMeasurement(10, [10, 10, 10], None, 30, 30, 5, None, None, 5)
    metrics.zero_all_fields()

    assert metrics.cpu_total_mj == 0
    assert metrics.efficiency_cores_mj == []
    assert metrics.performance_cores_mj == None
    assert metrics.efficiency_core_manager_mj == 0
    assert metrics.performance_core_manager_mj == 0
    assert metrics.dram_mj == 0
    assert metrics.gpu_mj == None
    assert metrics.gpu_sram_mj == None
    assert metrics.ane_mj == 0


def test_availability(mock_monitor):
    from zeus.device.soc.apple import apple_silicon_is_available

    assert apple_silicon_is_available()
