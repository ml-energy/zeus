"""Tests for the Raspberry Pi 5 PMIC machine backend.

None of these need a Pi. The parser runs against captured `pmic_read_adc`
output, and the monitor is driven through a fake `PMICReader` so the integration
and windowing logic can be checked deterministically.
"""

import sys
from unittest.mock import patch

import pytest

from zeus.device.machine.common import EmptyMachine, MachineMeasurement
from zeus.device.machine.rpi import (
    PMICReadError,
    PMICReader,
    RaspberryPi,
    RPIMeasurement,
    VcgencmdPMICReader,
    ZeusRaspberryPiInitError,
    _accumulate,
    default_reader,
    rpi_is_available,
)

# Captured verbatim from `vcgencmd pmic_read_adc` on a Raspberry Pi 5 Model B
# Rev 1.0 (2 GB), Raspberry Pi OS bookworm, with four busy cores. Under load
# both DRAM rails carry current: DDR_VDD2 at 0.225 A and DDR_VDDQ at 0.067 A.
# At idle both read exact zero, which is a property of the ADC at low current,
# so an idle capture would wrongly suggest DDR_VDDQ is never populated.
PMIC_OUTPUT = """ 3V7_WL_SW_A current(0)=0.09368929A
   3V3_SYS_A current(1)=0.08783370A
   1V8_SYS_A current(2)=0.18054710A
  DDR_VDD2_A current(3)=0.22543980A
  DDR_VDDQ_A current(4)=0.06666720A
   1V1_SYS_A current(5)=0.17566740A
    0V8_SW_A current(6)=0.37670900A
  VDD_CORE_A current(7)=6.22421000A
   3V3_DAC_A current(17)=0.00006105A
   3V3_ADC_A current(18)=0.00036630A
   0V8_AON_A current(16)=0.00305250A
      HDMI_A current(22)=0.02417580A
 3V7_WL_SW_V volt(8)=3.69977600V
   3V3_SYS_V volt(9)=3.31003300V
   1V8_SYS_V volt(10)=1.80512600V
  DDR_VDD2_V volt(11)=1.10476100V
  DDR_VDDQ_V volt(12)=0.60146460V
   1V1_SYS_V volt(13)=1.11025500V
    0V8_SW_V volt(14)=0.80146440V
  VDD_CORE_V volt(15)=0.86173300V
   3V3_DAC_V volt(20)=3.31318300V
   3V3_ADC_V volt(21)=3.30677300V
   0V8_AON_V volt(19)=0.80029220V
      HDMI_V volt(23)=5.08664000V
     EXT5V_V volt(24)=5.09066000V
      BATT_V volt(25)=0.00000000V
"""


class FakePMICReader(PMICReader):
    """Returns a scripted sequence of per-rail power readings, in mW."""

    def __init__(self, samples, fail_after=None):
        self.samples = list(samples)
        self.calls = 0
        self.fail_after = fail_after
        self.closed = False

    def read_rail_power_mw(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise PMICReadError("scripted failure")
        return dict(self.samples[min(self.calls - 1, len(self.samples) - 1)])

    def close(self):
        self.closed = True


class TestParsing:
    def test_only_rails_with_both_channels_yield_power(self):
        rails = PMICReader.parse(PMIC_OUTPUT)
        assert len(rails) == 12
        # Voltage-only channels cannot give power and must not appear.
        assert "EXT5V" not in rails
        assert "BATT" not in rails

    def test_power_is_current_times_voltage_in_mw(self):
        rails = PMICReader.parse(PMIC_OUTPUT)
        assert rails["VDD_CORE"] == pytest.approx(6.22421 * 0.861733 * 1000.0)

    def test_both_dram_rails_carry_current_under_load(self):
        # The idle capture this fixture replaced showed DDR_VDDQ at 0 A, which
        # invited the conclusion that it is never populated. It is.
        rails = PMICReader.parse(PMIC_OUTPUT)
        assert rails["DDR_VDD2"] > 0.0
        assert rails["DDR_VDDQ"] > 0.0

    def test_garbage_lines_are_ignored(self):
        assert PMICReader.parse("not an adc line\n\n") == {}

    def test_nul_separated_output_parses(self):
        # The mailbox returns one NUL-separated blob rather than newlines.
        blob = PMIC_OUTPUT.replace("\n", "\x00")
        assert len(PMICReader.parse(blob)) == 12


class TestMeasurement:
    def test_machine_energy_is_required_and_domains_are_not(self):
        m = RPIMeasurement(machine_energy_mj=5.0)
        assert m.machine_energy_mj == 5.0
        assert m.cpu_energy_mj is None
        assert m.dram_energy_mj is None

    def test_is_a_machine_measurement(self):
        assert isinstance(RPIMeasurement(machine_energy_mj=0.0), MachineMeasurement)

    def test_subtraction_is_field_wise(self):
        later = RPIMeasurement(machine_energy_mj=10.0, cpu_energy_mj=6.0, dram_energy_mj=1.0)
        earlier = RPIMeasurement(machine_energy_mj=4.0, cpu_energy_mj=2.0, dram_energy_mj=0.5)
        diff = later - earlier
        assert diff.machine_energy_mj == 6.0
        assert diff.cpu_energy_mj == 4.0
        assert diff.dram_energy_mj == 0.5

    def test_a_domain_unreadable_on_either_side_stays_none(self):
        later = RPIMeasurement(machine_energy_mj=10.0, cpu_energy_mj=6.0)
        earlier = RPIMeasurement(machine_energy_mj=4.0, cpu_energy_mj=2.0)
        assert (later - earlier).dram_energy_mj is None

    def test_subtracting_a_different_type_is_refused(self):
        with pytest.raises(TypeError):
            RPIMeasurement(machine_energy_mj=1.0) - object()  # type: ignore[operator]

    def test_zero_all_fields_leaves_unreadable_domains_alone(self):
        m = RPIMeasurement(machine_energy_mj=9.0, cpu_energy_mj=3.0)
        m.zero_all_fields()
        assert m.machine_energy_mj == 0.0
        assert m.cpu_energy_mj == 0.0
        assert m.dram_energy_mj is None


class TestAccumulate:
    def test_trapezoidal_over_one_interval(self):
        cumulative = RPIMeasurement(machine_energy_mj=0.0, cpu_energy_mj=0.0, dram_energy_mj=0.0)
        prev = {"VDD_CORE": 1000.0, "DDR_VDD2": 100.0}
        now = {"VDD_CORE": 3000.0, "DDR_VDD2": 300.0}
        _accumulate(cumulative, prev, now, 2.0)
        # mean power times dt, per rail
        assert cumulative.cpu_energy_mj == pytest.approx(2000.0 * 2.0)
        assert cumulative.dram_energy_mj == pytest.approx(200.0 * 2.0)
        assert cumulative.machine_energy_mj == pytest.approx(2000.0 * 2.0 + 200.0 * 2.0)

    def test_machine_total_covers_rails_outside_the_named_domains(self):
        cumulative = RPIMeasurement(machine_energy_mj=0.0)
        prev = {"VDD_CORE": 1000.0, "HDMI": 100.0}
        now = {"VDD_CORE": 1000.0, "HDMI": 100.0}
        _accumulate(cumulative, prev, now, 1.0)
        assert cumulative.machine_energy_mj == pytest.approx(1100.0)

    def test_a_rail_missing_from_one_sample_is_skipped(self):
        cumulative = RPIMeasurement(machine_energy_mj=0.0, dram_energy_mj=None)
        _accumulate(cumulative, {"VDD_CORE": 1000.0}, {}, 1.0)
        assert cumulative.machine_energy_mj == 0.0


class TestMonitor:
    def _monitor(self, reader):
        return RaspberryPi(reader=reader, poll_interval_s=0.01)

    def test_init_refuses_a_board_with_no_usable_rail(self):
        with pytest.raises(ZeusRaspberryPiInitError, match="no rail"):
            self._monitor(FakePMICReader([{}]))

    def test_init_surfaces_a_read_failure(self):
        with pytest.raises(ZeusRaspberryPiInitError, match="scripted failure"):
            self._monitor(FakePMICReader([{"VDD_CORE": 1.0}], fail_after=0))

    def test_windows_return_a_difference(self):
        reader = FakePMICReader([{"VDD_CORE": 1000.0, "DDR_VDD2": 100.0}])
        mon = self._monitor(reader)
        try:
            mon.begin_window("w")
            result = mon.end_window("w")
            assert isinstance(result, RPIMeasurement)
            assert result.machine_energy_mj >= 0.0
        finally:
            mon._stop_process()

    def test_duplicate_window_is_refused_unless_restarted(self):
        mon = self._monitor(FakePMICReader([{"VDD_CORE": 1000.0}]))
        try:
            mon.begin_window("w")
            with pytest.raises(KeyError):
                mon.begin_window("w")
            mon.begin_window("w", restart=True)
        finally:
            mon._stop_process()

    def test_unknown_window_is_refused(self):
        mon = self._monitor(FakePMICReader([{"VDD_CORE": 1000.0}]))
        try:
            with pytest.raises(KeyError):
                mon.end_window("never opened")
        finally:
            mon._stop_process()


class TestAvailability:
    def test_not_available_off_linux(self):
        with patch.object(sys, "platform", "darwin"):
            assert rpi_is_available() is False

    def test_not_available_on_x86(self):
        with (
            patch.object(sys, "platform", "linux"),
            patch("zeus.device.machine.rpi.platform.machine", return_value="x86_64"),
        ):
            assert rpi_is_available() is False

    def test_not_available_without_a_pi_model_string(self):
        with (
            patch.object(sys, "platform", "linux"),
            patch("zeus.device.machine.rpi.platform.machine", return_value="aarch64"),
            patch("zeus.device.machine.rpi.Path.read_bytes", return_value=b"Some Other Board"),
        ):
            assert rpi_is_available() is False


class TestReaderSelection:
    def test_falls_back_to_vcgencmd_when_the_mailbox_is_unusable(self):
        with patch(
            "zeus.device.machine.rpi.VcioPMICReader",
            side_effect=PMICReadError("no /dev/vcio"),
        ):
            assert isinstance(default_reader(), VcgencmdPMICReader)


class TestEmptyMachine:
    def test_every_operation_refuses(self):
        empty = EmptyMachine()
        assert empty.get_available_metrics() == set()
        for call in (
            empty.get_total_energy_consumption,
            lambda: empty.begin_window("w"),
            lambda: empty.end_window("w"),
        ):
            with pytest.raises(ValueError, match="No machine power meter"):
                call()
