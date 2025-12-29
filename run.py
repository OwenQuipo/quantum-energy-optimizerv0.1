# run.py
from __future__ import annotations

import sys
import threading
import warnings
from typing import Dict, List

from data import E_MAX, P_MAX, SOC_0, demand, price
from qubo import build_qubo_hybrid, decode_solution
from solve_classical import solve_simulated_annealing
from solve_quantum import solve_qaoa

warnings.filterwarnings("ignore", category=FutureWarning)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def evaluate_schedule(discharge: List[int], charge: List[int] | None) -> Dict[str, float]:
    """Compute exact cost metrics for a fixed dispatch schedule."""

    net_demand = []
    for t, base in enumerate(demand):
        flow = -discharge[t] * P_MAX
        if charge:
            flow += charge[t] * P_MAX
        net = base + flow
        net_demand.append(net)

    peak_kW = max(net_demand)
    energy_cost = dot(price, net_demand)
    demand_charge = peak_kW * 15.0
    total = energy_cost + demand_charge
    return {
        "net_demand": net_demand,
        "peak_kW": peak_kW,
        "energy_cost": energy_cost,
        "demand_charge": demand_charge,
        "total": total,
    }


def main(include_charge: bool = False):
    Q, meta = build_qubo_hybrid(price, demand, SOC_0, E_MAX, P_MAX, include_charge=include_charge)

    progress_event = threading.Event()

    def report_progress(current_step: int, total_steps: int) -> None:
        percent = (current_step / total_steps) * 100
        print(f"Progress: {percent:.1f}%")

    def listen_for_enter() -> None:
        """Watch stdin for Enter presses and request a progress update."""

        for line in sys.stdin:
            if line.strip() == "":
                progress_event.set()

    listener = threading.Thread(target=listen_for_enter, daemon=True)
    listener.start()

    print("Press Enter at any time to display the current progress...")

    x_classical, _ = solve_simulated_annealing(
        Q, progress_event=progress_event, progress_reporter=report_progress
    )
    decoded_classical = decode_solution(x_classical, meta)

    try:
        x_quantum, _ = solve_qaoa(Q)
        decoded_quantum = decode_solution(x_quantum, meta)
        quantum_failed = False
    except ValueError as exc:
        decoded_quantum = None
        quantum_failed = True
        quantum_error = str(exc)

    # Classical evaluation of the schedules (ground truth scoring)
    class_score = evaluate_schedule(
        decoded_classical["discharge"], decoded_classical.get("charge")
    )
    print("Classical schedule:", decoded_classical)
    print(
        f"  Energy cost: ${class_score['energy_cost']:.2f}, Demand charge: ${class_score['demand_charge']:.2f}, Total: ${class_score['total']:.2f}"
    )

    if quantum_failed:
        print("Quantum run skipped:", quantum_error)
    else:
        quant_score = evaluate_schedule(
            decoded_quantum["discharge"], decoded_quantum.get("charge")
        )
        print("Quantum schedule:", decoded_quantum)
        print(
            f"  Energy cost: ${quant_score['energy_cost']:.2f}, Demand charge: ${quant_score['demand_charge']:.2f}, Total: ${quant_score['total']:.2f}"
        )


if __name__ == "__main__":
    main()
