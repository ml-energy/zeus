"""Tests for AMD GPU index resolution.

These tests focus on the HIP-index → amdsmi-handle translation in
`AMDGPU._get_handle`. On some nodes (notably MI350X) the HIP index space
that PyTorch and `HIP_VISIBLE_DEVICES` use does not coincide with
`amdsmi_get_processor_handles()`'s BDF-sorted ordering, so the mapping
must come from `amdsmi_get_gpu_enumeration_info(handle)["hip_id"]`.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_amdsmi_mock(hip_id_by_handle: dict[str, int]) -> MagicMock:
    """Build a fake `amdsmi` module.

    Args:
        hip_id_by_handle: Maps amd-smi handle (a sentinel string standing in
            for an opaque processor handle) to its HIP index. The order of
            this dict is the order `amdsmi_get_processor_handles()` returns,
            i.e., amd-smi's own GPU index space (BDF-sorted on real hardware).
    """
    amdsmi = MagicMock()

    handles = list(hip_id_by_handle.keys())
    amdsmi.amdsmi_get_processor_handles.return_value = handles
    amdsmi.amdsmi_get_gpu_enumeration_info.side_effect = lambda h: {"hip_id": hip_id_by_handle[h]}

    class _AmdSmiLibraryException(Exception):
        def get_error_code(self):
            return 31  # NOT_FOUND

        def get_error_info(self):
            return "mock error"

    amdsmi.AmdSmiLibraryException = _AmdSmiLibraryException
    return amdsmi


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
