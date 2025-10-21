from utils.adaptive_alpha import alpha_from_ssmd_trend, AlphaTrendState

st = AlphaTrendState(ema_ssmd=2.15, prev_ema_ssmd=2.15, alpha=1.0)

a, st = alpha_from_ssmd_trend(ssmd_now=2.15, state=st, ssmd_target=1.8, alpha0=1.0, hysteresis=1)
print(a, st.ema_ssmd)
a, st = alpha_from_ssmd_trend(ssmd_now=2.15, state=st, ssmd_target=1.8, alpha0=1.0, hysteresis=1)
print(a, st.ema_ssmd)
a, st = alpha_from_ssmd_trend(ssmd_now=2.15, state=st, ssmd_target=1.8, alpha0=1.0, hysteresis=1)
print(a, st.ema_ssmd)
a, st = alpha_from_ssmd_trend(ssmd_now=1.8, state=st, ssmd_target=1.8, alpha0=1.0, hysteresis=1)
print(a, st.ema_ssmd)
a, st = alpha_from_ssmd_trend(ssmd_now=1.8, state=st, ssmd_target=1.8, alpha0=1.0, hysteresis=1)
print(a, st.ema_ssmd)
