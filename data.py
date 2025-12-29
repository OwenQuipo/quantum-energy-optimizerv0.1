# data.py
# 3-hour scenario tuned to stay tiny for QAOA while keeping an interesting
# peak/price trade-off (hour 1 spike + mild hour 2 bump).

T = 3  # hours

price = [0.05, 0.35, 0.09]
demand = [100, 230, 120]

E_MAX = 60     # kWh
P_MAX = 20     # kW
SOC_0 = 40     # kWh
