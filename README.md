# PMD EV Experiments

This repository contains exploratory experiments built using
**PowerModelsDistribution.jl**, focused on understanding how
low-voltage (LV) distribution networks respond to electric vehicle (EV)
charging load.

The work here is intentionally experimental. The aim is to build
technical intuition, test modelling assumptions, and explore voltage
and congestion behaviour under stress, not to produce production-ready
or validated planning tools.

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

## Data Sources and Licensing

This repository includes **small, curated datasets** that are explicitly
open-licensed and suitable for redistribution. Care has been taken to
respect licensing terms, attribution requirements, and data ownership
boundaries.

### Included datasets

#### 1. CSIRO Data Collection (csiro:62996)

- **Source:** https://data.csiro.au/collection/csiro:62996  
- **Provider:** Commonwealth Scientific and Industrial Research Organisation (CSIRO)  
- **License:** Creative Commons Attribution (CC-BY)  

This dataset is redistributed in accordance with the CC-BY licence.
Where data has been subset, processed, or reformatted, this repository
does not imply that CSIRO endorses the resulting analysis or conclusions.

Proper attribution to CSIRO is retained.

---

#### 2. Other public EV and LV load datasets

Additional datasets may be included where:

- redistribution is explicitly permitted under an open licence
  (e.g. CC-BY or equivalent), and  
- the data has been reduced, sampled, or curated for experimental use.

Original sources and licences are documented inline or alongside the
relevant data files.

---

### Data handling principles

- Only **open-licensed** data is committed to this repository  
- Large, raw, or upstream datasets are excluded  
- Internal or restricted network models are **not included**  
- Derived datasets reflect personal processing and interpretation  

These choices are intentional and aim to balance reproducibility,
ethical data use, and practical repository management.

---

## Context and Attribution

This repository reflects **personal experimentation and learning** while
working with PowerModelsDistribution.jl in an applied research context.

Although some experiments are inspired by real-world distribution
network research, **this repository is not an official CSIRO codebase,
deliverable, or endorsed work**.

All code, analysis, interpretations, and any errors are entirely my own.

---

## Status

This repository is a work in progress.

Experiments may evolve, be refactored, or be abandoned as understanding
improves. Code quality and documentation will be refined iteratively.

Nothing here should be interpreted as validated network planning advice.

---

## Tools

- Julia  
- PowerModelsDistribution.jl  
- JuMP  
---

## Notes to Future Me

This repo exists to capture thinking, not perfection.
If something looks messy, it probably reflects a real moment of learning.

