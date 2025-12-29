# Quantum Battery Dispatch (Stage 1)

A **quantum-native optimization engine** for battery charge / discharge
scheduling. The core decision problem is formulated as a **QUBO (Quadratic
Unconstrained Binary Optimization)** and solved using **QAOA** via Qiskit, with
an exact classical verifier on the side.

## What changed in this drop
- The quantum subproblem is now a compact dispatch selector (≤20 binaries).
- Energy feasibility is enforced via binary budget constraints (no per-hour
  SOC variables yet).
- Demand-charge effects are represented with a peak-shaving proxy in the QUBO
  and then evaluated exactly in the classical scoring layer.

This keeps the quantum layer focused on the hard combinatorial choice while the
classical layer computes the precise bill.

## Inputs
- Hourly electricity prices
- Baseline demand per hour
- Battery limits: energy capacity ``E_MAX``, power ``P_MAX``, initial state of
  charge ``SOC_0``

## Outputs
- Binary discharge (and optional charge) decisions per hour
- Classical post-processing with true net demand, peak kW, energy cost, and
  demand charge

## Repository structure
- `data.py` — Hardcoded example scenario
- `qubo.py` — QUBO construction (quantum-native core)
- `solve_classical.py` — Classical simulated annealing solver for QUBOs
- `solve_quantum.py` — QAOA-based solver using Qiskit (optional)
- `run.py` — Entry point
- `README.md`

## Running the demo
The classical path has **no third-party dependencies**; invoke it directly:

```
python run.py
```

If Qiskit is installed, a QAOA run will be attempted as well. Otherwise the
quantum step is skipped with a clear message while the classical result is
still produced.

This is an early research-driven prototype.
