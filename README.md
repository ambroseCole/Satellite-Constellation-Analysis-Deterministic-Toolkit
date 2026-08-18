# Satellite Constellation Analysis Deterministic Toolkit (SCADT)

> An end-to-end simulation pipeline for analyzing collision avoidance maneuvers and their effects on mega-constellation geometry.

---

## Overview

This repository contains a simulation pipeline for generating individual case studies and aggregate statistics on satellite constellation effects under baseline conditions and during debris breakup events. Notable unique outputs from this pipeline include delta-v costs aggregated across collision-avoidance maneuvers (CAMs), phasing degradation analysis, and effective batching of large-scale TLE propagation. The files included do not constitute a plug-and-play system for most investigations, but rather a template to work from for research requiring deterministic simulation of satellite constellation configurations, CAMs, and inter-constellation distance degradation. Achievable data is limited by user hardware; SLURM bash files are included for those with access to High Performance Computing (HPC).

---

## Pipeline

The tools run in sequence, each producing inputs for the next:

```
download_tle.py       →  debris_catalog.tle
baseline_satsat.py    →  baseline separations
sat_debris_cam.py     →  conjunction screening + CAMs   [imports nasa_sbm.py]
propagate_perturbed.py →  degradation events
compare_satsat.py     →  cross-configuration summary
```

| Script | Purpose | Output |
|--------|---------|-----------|
| `download_tle.py` | Fetch debris catalog from Space-Track | `debris_catalog.tle` |
| `WalkerDeltaSGP.py` | Constellation generator (module) | Satrec objects |
| `baseline_satsat.py` | Reference intra-constellation separations | `results_*.csv` |
| `nasa_sbm.py` | Break-up fragment generator (module) | Satrec objects |
| `sat_debris_cam.py` | Conjunction screening + CAM computation | `results_cam_*.csv` |
| `propagate_perturbed.py` | CAM-induced degradation analysis | `results_prop_perp_*.csv` |
| `compare_satsat.py` | Cross-configuration cascade summary | `cascade_summary_*.csv` |

---

## Installation

```bash
git clone https://github.com/ambroseCole/Constellation-Analysis-Toolkit.git
cd Constellation-Analysis-Toolkit
pip install numpy scipy sgp4 spacetrack
```

Requires Python [version]. Dependencies: numpy, scipy, sgp4, spacetrack.

---

## Quick start

```bash
# 1. Download the debris catalog (requires a free Space-Track account)
python download_tle.py

# 2. Generate baseline separations for a constellation
python baseline_satsat.py --num_sats 1584 --num_planes 72 --output baseline_1584.csv

# 3. Screen for conjunctions and compute maneuvers
python sat_debris_cam.py --num_sats 1584 --num_planes 72 \
    --debris_file debris_catalog.tle --output cams_1584.csv

# 4. Analyze CAM-induced degradation
python propagate_perturbed.py --cam_file cams_1584.csv \
    --baseline_file baseline_1584.csv --num_sats 1584 --num_planes 72 \
    --output effects_1584.csv

# 5. Summarize
python compare_satsat.py --files effects_1584.csv --num_sats 1584
```

---

## Configuration

Key parameters and where they are set:

| Parameter | Where | Notes |
|-----------|-------|-------|
| Epoch | config block | **Must be identical across all scripts** |
| Constellation (t/p/f) | CLI args / config block | Key customization point |
| Altitude, inclination | `argConst` block | Helpful to sweep |
| Screening radius | config block | Tweak for scenario based on expected relative delta-v |
| Timestep (`dt_coarse`) | config block | Tweak for scenario and tolerances |
| Pc threshold | `sat_debris_cam.py` | Default is set quite high |
| Covariance (RTN σ) | `sat_debris_cam.py` | SOCRATES for sats, TLE-typical for debris |

> **Epoch consistency:** every script derives satellite positions from a fixed
> epoch. If you change it, change it everywhere, or propagation times will not
> align and results will be meaningless.

---

## Methodology

- **Propagation:** SGP4 (`sgp4` library), vectorized via `SatrecArray`.
- **Conjunction screening:** `scipy.spatial.cKDTree`, multi-stage refinement.
- **Collision probability:** Foster 2D method — Foster & Estes, NASA/JSC-25898 (1992).
- **Maneuvers:** Clohessy-Wiltshire tangential impulse — Clohessy & Wiltshire, *J. Aerospace Sci.* (1960).
- **Break-up model:** NASA Standard Break-Up Model — Johnson et al., *Adv. Space Res.* 28(9) (2001).
- **Covariance values:** SOCRATES — Kelso & Alfano, AAS 05-124 (2005).

---

## Limitations

- No active station-keeping model; results in slight constellation drift.
- SGP4 accuracy degrades slightly beyond ~2 weeks from epoch.
- Tangential-only maneuvers (no radial/cross-track component).
- Assumed diagonal RTN covariance; no time-varying uncertainty growth.

---

## Credentials

A free [Space-Track](https://www.space-track.org) account is required to
download the debris catalog. Set credentials in `download_tle.py`.

---

## Citation

If you use this work, please cite however you see fit.

## License

MIT License — see [LICENSE](LICENSE).
