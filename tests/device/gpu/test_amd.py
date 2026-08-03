"""Tests for AMD GPU index resolution.

These tests focus on the HIP-index to amdsmi-handle translation in
`AMDGPU._get_handle`. On some nodes (notably MI350X) the HIP index space
that PyTorch and `HIP_VISIBLE_DEVICES` use does not coincide with
`amdsmi_get_processor_handles()`'s BDF-sorted ordering, so the mapping
must come from amdsmi's enumeration or KFD topology information.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from zeus.utils.zeusd import GpuInfo


def _make_base_amdsmi_mock(handles: list[str]) -> MagicMock:
    """Build a fake `amdsmi` module without either index-mapping API."""
    amdsmi = MagicMock()
    amdsmi.amdsmi_get_processor_handles.return_value = handles
    del amdsmi.amdsmi_get_gpu_enumeration_info
    del amdsmi.amdsmi_get_gpu_kfd_info

    class _AmdSmiLibraryException(Exception):
        def get_error_code(self):
            return 31  # NOT_FOUND

        def get_error_info(self):
            return "mock error"

    amdsmi.AmdSmiLibraryException = _AmdSmiLibraryException
    return amdsmi


def _make_amdsmi_mock(hip_id_by_handle: dict[str, int]) -> MagicMock:
    """Build a fake `amdsmi` module with the enumeration-info API.

    Args:
        hip_id_by_handle: Maps amd-smi handle (a sentinel string standing in
            for an opaque processor handle) to its HIP index. The order of
            this dict is the order `amdsmi_get_processor_handles()` returns,
            i.e., amd-smi's own GPU index space (BDF-sorted on real hardware).
    """
    handles = list(hip_id_by_handle.keys())
    amdsmi = _make_base_amdsmi_mock(handles)
    amdsmi.amdsmi_get_gpu_enumeration_info = MagicMock(side_effect=lambda h: {"hip_id": hip_id_by_handle[h]})
    return amdsmi


def _make_kfd_amdsmi_mock(node_id_by_handle: dict[str, int | str]) -> MagicMock:
    """Build a fake `amdsmi` module with the KFD-info API."""
    handles = list(node_id_by_handle.keys())
    amdsmi = _make_base_amdsmi_mock(handles)
    amdsmi.amdsmi_get_gpu_kfd_info = MagicMock(side_effect=lambda h: {"node_id": node_id_by_handle[h]})
    return amdsmi


def _make_zeusd_client(gpus: list[GpuInfo]) -> MagicMock:
    """Build a fake ZeusdClient with GPU capabilities enabled."""
    client = MagicMock()
    client.gpus = gpus
    client.gpu_ids = [gpu.id for gpu in gpus]
    client.endpoint = "/run/zeusd/zeusd.sock"
    client.auth_error = None
    client.can_read_gpu = True
    client.can_control_gpu = True
    return client


@pytest.fixture
def fresh_amd_module():
    """Yield a callable that installs a fake `amdsmi` and (re)imports `zeus.device.gpu.amd`.

    `zeus.device.gpu.amd` binds `amdsmi` at import time, so we must drop any
    cached version before patching `sys.modules`.
    """
    started = []

    def _factory(amdsmi_mock: MagicMock):
        sys.modules.pop("zeus.device.gpu.amd", None)
        ctx = patch.dict(sys.modules, {"amdsmi": amdsmi_mock})
        ctx.start()
        started.append(ctx)
        import zeus.device.gpu.amd as amd_module

        return amd_module

    yield _factory

    for ctx in started:
        ctx.stop()
    sys.modules.pop("zeus.device.gpu.amd", None)


def test_get_handle_translates_hip_index_when_orderings_differ(fresh_amd_module):
    # MI350X-style reordering: amd-smi enumerates by BDF, but the HIP runtime
    # exposes them in a different order.
    amdsmi_mock = _make_amdsmi_mock(
        {
            "h_bdf0": 3,
            "h_bdf1": 2,
            "h_bdf2": 1,
            "h_bdf3": 0,
        }
    )
    amd = fresh_amd_module(amdsmi_mock)

    assert amd.AMDGPU(0).handle == "h_bdf3"
    assert amd.AMDGPU(1).handle == "h_bdf2"
    assert amd.AMDGPU(2).handle == "h_bdf1"
    assert amd.AMDGPU(3).handle == "h_bdf0"


def test_get_handle_identity_mapping(fresh_amd_module):
    # Most nodes: HIP index and amd-smi GPU index coincide.
    amdsmi_mock = _make_amdsmi_mock({"h0": 0, "h1": 1, "h2": 2, "h3": 3})
    amd = fresh_amd_module(amdsmi_mock)

    for i, expected in enumerate(["h0", "h1", "h2", "h3"]):
        assert amd.AMDGPU(i).handle == expected


def test_get_handle_raises_for_missing_hip_index(fresh_amd_module):
    amdsmi_mock = _make_amdsmi_mock({"h0": 0, "h1": 1})
    amd = fresh_amd_module(amdsmi_mock)

    import zeus.device.gpu.common as gpu_common

    with pytest.raises(gpu_common.ZeusGPUNotFoundError) as exc_info:
        amd.AMDGPU(7)

    # The error should list the available HIP indices so users can diagnose
    # `HIP_VISIBLE_DEVICES` mismatches.
    assert "HIP index 7" in str(exc_info.value)
    assert "[0, 1]" in str(exc_info.value)


def test_get_handle_maps_kfd_node_order_to_hip_indices(fresh_amd_module):
    amdsmi_mock = _make_kfd_amdsmi_mock({"h_a": 2, "h_b": 0, "h_c": 1})
    amd = fresh_amd_module(amdsmi_mock)

    assert amd.AMDGPU(0).handle == "h_b"
    assert amd.AMDGPU(1).handle == "h_c"
    assert amd.AMDGPU(2).handle == "h_a"


def test_get_handle_raises_for_non_integer_kfd_node_id(fresh_amd_module):
    amdsmi_mock = _make_kfd_amdsmi_mock({"h0": 0, "h1": "N/A"})
    amd = fresh_amd_module(amdsmi_mock)

    import zeus.device.gpu.common as gpu_common

    with pytest.raises(gpu_common.ZeusGPUInitError, match="N/A"):
        amd.AMDGPU(0)


def test_get_handle_requires_amdsmi_6_3_mapping_api(fresh_amd_module):
    amdsmi_mock = _make_base_amdsmi_mock(["h0"])
    amd = fresh_amd_module(amdsmi_mock)

    import zeus.device.gpu.common as gpu_common

    with pytest.raises(gpu_common.ZeusGPUInitError, match="6\\.3"):
        amd.AMDGPU(0)


def test_zeusd_gpu_matches_daemon_id_by_pci_address(fresh_amd_module):
    """Resolve a reordered daemon GPU ID by PCI address."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0, "h1": 1})
    amdsmi_mock.amdsmi_get_gpu_device_bdf.side_effect = {
        "h0": "0000:11:00.0",
        "h1": "0000:10:00.0",
    }.__getitem__
    amd = fresh_amd_module(amdsmi_mock)
    client = _make_zeusd_client(
        [
            GpuInfo(id=0, name="GPU 1", pci_address="0000:10:00.0", cumulative_energy_available=True),
            GpuInfo(id=2, name="GPU 0", pci_address="0000:11:00.0", cumulative_energy_available=True),
        ]
    )

    gpu = amd.ZeusdAMDGPU(0, client)
    gpu.set_gpu_locked_clocks(500, 1700, block=False)

    assert gpu._daemon_gpu_id == 2
    client.set_gpu_locked_clocks.assert_called_once_with([2], 500, 1700, False)


def test_zeusd_gpu_raises_when_local_pci_address_is_missing(fresh_amd_module):
    """Reject a local GPU that discovery does not contain."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0})
    amdsmi_mock.amdsmi_get_gpu_device_bdf.return_value = "0000:8e:00.0"
    amd = fresh_amd_module(amdsmi_mock)
    client = _make_zeusd_client(
        [GpuInfo(id=0, name="GPU 0", pci_address="0000:10:00.0", cumulative_energy_available=True)]
    )

    import zeus.device.gpu.common as gpu_common

    with pytest.raises(gpu_common.ZeusGPUInitError, match="0000:8e:00.0"):
        amd.ZeusdAMDGPU(0, client)


def test_zeusd_gpu_skips_unchanged_power_limit(fresh_amd_module):
    """Skip a daemon call when the requested power limit is current."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0})
    amdsmi_mock.amdsmi_get_gpu_device_bdf.return_value = "0000:8e:00.0"
    amdsmi_mock.amdsmi_get_power_cap_info.return_value = {"power_cap": 250_000_000}
    amd = fresh_amd_module(amdsmi_mock)
    client = _make_zeusd_client(
        [GpuInfo(id=3, name="GPU 0", pci_address="0000:8e:00.0", cumulative_energy_available=True)]
    )
    gpu = amd.ZeusdAMDGPU(0, client)

    gpu.set_power_management_limit(250_000, block=False)

    client.set_power_limit.assert_not_called()


def test_local_gpu_rejects_single_domain_clock_resets(fresh_amd_module):
    """Reject AMD single-domain reset requests without amdsmi calls."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0})
    amd = fresh_amd_module(amdsmi_mock)
    gpu = amd.AMDGPU(0)

    import zeus.device.gpu.common as gpu_common

    with pytest.raises(gpu_common.ZeusGPUNotSupportedError, match="cannot reset a single clock domain"):
        gpu.reset_gpu_locked_clocks(block=False)
    with pytest.raises(gpu_common.ZeusGPUNotSupportedError, match="cannot reset a single clock domain"):
        gpu.reset_memory_locked_clocks(block=False)

    amdsmi_mock.amdsmi_set_gpu_clk_range.assert_not_called()


def test_zeusd_gpu_rejects_single_domain_clock_resets(fresh_amd_module):
    """Reject AMD single-domain reset requests without daemon calls."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0})
    amdsmi_mock.amdsmi_get_gpu_device_bdf.return_value = "0000:8e:00.0"
    amd = fresh_amd_module(amdsmi_mock)
    client = _make_zeusd_client(
        [GpuInfo(id=3, name="GPU 0", pci_address="0000:8e:00.0", cumulative_energy_available=True)]
    )
    gpu = amd.ZeusdAMDGPU(0, client)

    import zeus.device.gpu.common as gpu_common

    with pytest.raises(gpu_common.ZeusGPUNotSupportedError, match="cannot reset a single clock domain"):
        gpu.reset_gpu_locked_clocks(block=False)
    with pytest.raises(gpu_common.ZeusGPUNotSupportedError, match="cannot reset a single clock domain"):
        gpu.reset_memory_locked_clocks(block=False)

    client.reset_gpu_locked_clocks.assert_not_called()
    client.reset_mem_locked_clocks.assert_not_called()


def test_zeusd_gpu_accepts_control_only_daemon(fresh_amd_module):
    """Initialize against a daemon that exposes only gpu-control."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0})
    amdsmi_mock.amdsmi_get_gpu_device_bdf.return_value = "0000:8e:00.0"
    amd = fresh_amd_module(amdsmi_mock)
    client = _make_zeusd_client(
        [GpuInfo(id=3, name="GPU 0", pci_address="0000:8e:00.0", cumulative_energy_available=True)]
    )
    client.can_read_gpu = False

    gpu = amd.ZeusdAMDGPU(0, client)

    assert gpu._daemon_gpu_id == 3


def test_zeusd_gpu_resets_all_clock_domains_through_daemon(fresh_amd_module):
    """Relay the combined clock reset to the daemon with the daemon GPU ID."""
    amdsmi_mock = _make_amdsmi_mock({"h0": 0})
    amdsmi_mock.amdsmi_get_gpu_device_bdf.return_value = "0000:8e:00.0"
    amd = fresh_amd_module(amdsmi_mock)
    client = _make_zeusd_client(
        [GpuInfo(id=3, name="GPU 0", pci_address="0000:8e:00.0", cumulative_energy_available=True)]
    )
    gpu = amd.ZeusdAMDGPU(0, client)

    gpu.reset_locked_clocks(block=False)

    client.reset_locked_clocks.assert_called_once_with([3], False)
