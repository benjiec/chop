import unittest

from utils.adaptive_alpha import compute_ssmd, AlphaTrendState, alpha_from_ssmd_trend


class TestAdaptiveAlphaUtils(unittest.TestCase):
    def test_compute_ssmd_basic(self):
        # mp=0.8, sp=0.1, mn=0.2, sn=0.1 => numerator=0.6, denom=sqrt(0.01+0.01)=~0.1414 => ~4.2426
        s = compute_ssmd(0.8, 0.1, 0.2, 0.1)
        self.assertTrue(s == s)  # not NaN
        self.assertGreater(s, 4.0)
        self.assertLess(s, 4.5)

    def test_alpha_trend_scheduler_pd_response(self):
        st = AlphaTrendState(ema_ssmd=2.0, prev_ema_ssmd=2.0, alpha=1.0)

        # Rising SSMD should reduce alpha toward alpha_min
        a1, st = alpha_from_ssmd_trend(ssmd_now=3.0, state=st, ssmd_target=3.3, k_p=0.2, k_d=0.1, beta=0.2, alpha0=1.0, alpha_min=0.05, hysteresis=1)
        self.assertLessEqual(a1, 1.0)
        self.assertGreaterEqual(a1, 0.05)

        a2, st = alpha_from_ssmd_trend(ssmd_now=3.6, state=st, ssmd_target=3.3, k_p=0.2, k_d=0.1, beta=0.2, alpha0=1.0, alpha_min=0.05, hysteresis=1)
        self.assertLessEqual(a2, a1)  # continued improvement lowers alpha
        self.assertGreaterEqual(a2, 0.05)

        # Falling SSMD should increase alpha, capped at alpha0
        a3, st = alpha_from_ssmd_trend(ssmd_now=2.5, state=st, ssmd_target=3.3, k_p=0.2, k_d=0.1, beta=0.2, alpha0=1.0, alpha_min=0.05, hysteresis=1)
        self.assertGreaterEqual(a3, a2)
        self.assertLessEqual(a3, 1.0)

