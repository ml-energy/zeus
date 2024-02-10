from zeus.monitor import ZeusMonitor
from zeus.optimizer import GlobalPowerLimitOptimizer
from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer
import inspect
import unittest

# TESTCASES: HFGlobalPowerLimitOptimizer


class TestOptimizerSignatures(unittest.TestCase):
    def test_constructor_signature_equality(self):
        """Ensure that the constructor signatures of GPLO and HFGPLO are exactly the same."""
        gplo_signature = inspect.signature(GlobalPowerLimitOptimizer.__init__)
        hfgplo_signature = inspect.signature(HFGlobalPowerLimitOptimizer.__init__)

        self.assertEqual(gplo_signature, hfgplo_signature, "Constructor signatures do not match.")

    def test_HFGPLO_inherits_TrainerCallback(self):
        """Ensure that HFGPLO inherits from TrainerCallback."""
        self.assertTrue(issubclass(HFGlobalPowerLimitOptimizer, ZeusMonitor), "HFGPLO does not inherit from TrainerCallback.")
    
    def test_HFGPLO_usage(self):
        """Ensure that HFGPLO can be used as a HuggingFace TrainerCallback."""
        hfgplo = HFGlobalPowerLimitOptimizer()
