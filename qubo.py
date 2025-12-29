# qubo.py
"""QUBO construction utilities for the hybrid quantum/classical split.

Stage-1 focuses on a small quantum subproblem: selecting charge/discharge
hours subject to energy feasibility. Demand charges are handled via a
*proxy* term in the QUBO and then exactly evaluated classically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class QuboMeta:
    """Metadata needed to interpret a QUBO solution vector."""

    T: int
    discharge_budget_bits: int
    charge_budget_bits: int
    include_charge: bool
    demand_rate: float
    peak_weight: float

    @property
    def num_variables(self) -> int:
        count = self.T  # discharge
        if self.include_charge:
            count += self.T  # charge
        count += self.discharge_budget_bits
        count += self.charge_budget_bits
        return count


def _add_penalty_square(
    Q: List[List[float]], penalty: float, constant: float, terms: List[Tuple[int, float]]
) -> None:
    """Add a squared penalty ``penalty * (constant + sum(a_i x_i))^2`` to ``Q``."""

    for i, (idx_i, weight_i) in enumerate(terms):
        Q[idx_i][idx_i] += penalty * (weight_i**2 + 2 * constant * weight_i)
        for idx_j, weight_j in terms[i + 1 :]:
            Q[idx_i][idx_j] += 2 * penalty * weight_i * weight_j


def _budget_bits(budget: int) -> int:
    return max(1, math.ceil(math.log2(budget + 1)))


def build_qubo_hybrid(
    price: Sequence[float],
    demand: Sequence[float],
    SOC_0: float,
    E_MAX: float,
    P_MAX: float,
    dt: float = 1.0,
    demand_rate: float = 15.0,
    lambda_energy: float = 50.0,
    lambda_peak_proxy: float = 0.5,
    lambda_balance: float = 5.0,
    include_charge: bool = False,
) -> Tuple[List[List[float]], QuboMeta]:
    """Build a compact QUBO focused on discrete dispatch decisions.

    Variables (all binary):
    - d_t: discharge decision at time t (size T)
    - c_t: optional charge decision at time t (size T if enabled)
    - s^d_b: discharge budget slack bits enforcing sum(d_t) + slack = Kd
    - s^c_b: charge budget slack bits enforcing sum(c_t) + slack = Kc

    Constraints are encoded as squared penalties so the quantum layer directly
    enforces energy feasibility and mutual exclusion (if charging is enabled).
    """

    price = [float(p) for p in price]
    demand = [float(d) for d in demand]

    if len(price) != len(demand):
        raise ValueError("price and demand must have the same length")

    T = len(price)
    Kd = int(math.floor(SOC_0 / (P_MAX * dt)))
    if include_charge:
        Kc = int(math.floor((E_MAX - SOC_0) / (P_MAX * dt)))
    else:
        Kc = 0

    meta = QuboMeta(
        T=T,
        discharge_budget_bits=_budget_bits(max(Kd, 1)),
        charge_budget_bits=_budget_bits(max(Kc, 1)) if include_charge else 0,
        include_charge=include_charge,
        demand_rate=demand_rate,
        peak_weight=lambda_peak_proxy,
    )

    N = meta.num_variables
    Q = [[0.0 for _ in range(N)] for _ in range(N)]

    idx_discharge = list(range(T))
    if include_charge:
        idx_charge = list(range(T, 2 * T))
    else:
        idx_charge = []

    pointer = T + (T if include_charge else 0)
    idx_slack_discharge = list(range(pointer, pointer + meta.discharge_budget_bits))
    pointer += meta.discharge_budget_bits
    idx_slack_charge = list(range(pointer, pointer + meta.charge_budget_bits))

    # Objective: energy arbitrage (discharge reduces cost) + peak-shaving proxy.
    for t, idx in enumerate(idx_discharge):
        energy_savings = price[t] * P_MAX * dt
        peak_proxy = demand[t] * meta.peak_weight
        Q[idx][idx] += -(energy_savings + peak_proxy)

    if include_charge:
        for t, idx in enumerate(idx_charge):
            energy_cost = price[t] * P_MAX * dt
            Q[idx][idx] += energy_cost  # charging costs energy
            Q[idx][idx] += demand[t] * meta.peak_weight  # charging worsens peak
            # Mutual exclusion: c_t + d_t <= 1 -> (c + d)^2 penalty
            Q[idx][idx] += lambda_energy
            Q[idx_discharge[t]][idx_discharge[t]] += lambda_energy
            Q[idx][idx_discharge[t]] += 2 * lambda_energy

    # Discharge budget: sum(d_t) + slack = Kd
    discharge_terms = [(idx, 1.0) for idx in idx_discharge]
    for bit, idx in enumerate(idx_slack_discharge):
        discharge_terms.append((idx, float(2**bit)))
    _add_penalty_square(Q, lambda_energy, -float(Kd), discharge_terms)

    # Charge budget: sum(c_t) + slack = Kc (if enabled)
    if include_charge:
        charge_terms = [(idx, 1.0) for idx in idx_charge]
        for bit, idx in enumerate(idx_slack_charge):
            charge_terms.append((idx, float(2**bit)))
        _add_penalty_square(Q, lambda_energy, -float(Kc), charge_terms)

        # End-of-horizon SOC stabilization: (sum c - sum d)^2
        balance_terms = [(idx, 1.0) for idx in idx_charge]
        balance_terms.extend([(idx, -1.0) for idx in idx_discharge])
        _add_penalty_square(Q, lambda_balance, 0.0, balance_terms)

    return Q, meta


def decode_solution(x: Sequence[int], meta: QuboMeta) -> Dict[str, object]:
    """Decode a binary solution vector using ``meta``."""

    pointer = 0
    discharge = [int(val) for val in x[pointer : pointer + meta.T]]
    pointer += meta.T

    charge: List[int] = []
    if meta.include_charge:
        charge = [int(val) for val in x[pointer : pointer + meta.T]]
        pointer += meta.T

    discharge_slack_bits = [int(val) for val in x[pointer : pointer + meta.discharge_budget_bits]]
    pointer += meta.discharge_budget_bits
    charge_slack_bits: List[int] = []
    if meta.include_charge:
        charge_slack_bits = [int(val) for val in x[pointer : pointer + meta.charge_budget_bits]]

    discharge_slack = sum(bit * (2**idx) for idx, bit in enumerate(discharge_slack_bits))
    charge_slack = sum(bit * (2**idx) for idx, bit in enumerate(charge_slack_bits))

    return {
        "discharge": discharge,
        "charge": charge if meta.include_charge else None,
        "discharge_slack": discharge_slack,
        "charge_slack": charge_slack if meta.include_charge else None,
    }
