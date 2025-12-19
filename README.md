# PMD EV Experiments

This repository contains a collection of exploratory experiments built using
**PowerModelsDistribution.jl**, focused on understanding how low-voltage
distribution networks respond to increased electric vehicle (EV) charging load.

The work here is intentionally experimental. The goal is to build intuition,
test modelling assumptions, and explore network behaviour under stress,
rather than to produce polished or production-ready tools.

---

## Scope

The experiments in this repository explore topics such as:

- EV load injection at specific buses and phases
- Voltage magnitude violations and binding constraints
- Line and transformer loading under increased demand
- Sensitivity of network behaviour to EV placement and sizing
- Comparison between baseline and stressed network scenarios

Most experiments use small to medium LV network models and focus on
unbalanced power flow and OPF behaviour.

---

## Repository Structure

```text
pmd-ev-experiments/
├── pmd_ev_experiments/ # Core experiment scripts and scenarios
├── extensions/ # Custom helpers or model extensions (if used)
├── notebooks/ # Optional analysis or plotting notebooks
└── README.md
```


Large datasets and external data sources are intentionally excluded from
version control and are documented where required.

---

## Context 

This work is part of my personal learning and experimentation while working
with **PowerModelsDistribution.jl** in an applied research context.

---

## Status

This repository is a work in progress.

Experiments may change structure, be refactored, or be abandoned as new
insights are gained. Code quality and documentation will improve iteratively
as understanding deepens.

---

## Tools

- Julia
- PowerModelsDistribution.jl
- JuMP
- Ipopt (or other nonlinear solvers)

---

## Notes

If you are reading this as a future version of me:
be kind — this repo exists to capture thinking, not perfection.
