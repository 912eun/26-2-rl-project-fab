# SMT2020 Full-Fab Benchmark — 30-Replication Simulation Output (P-30)

This dataset contains the raw event-level output of **30 independent
replications** of a four-year simulation of the **SMT2020 full-fab
benchmark model**, produced with the GoSoL discrete-event simulation engine.
It is the companion dataset to:

> I. Shibata, "Independent Reproduction of the SMT2020 Full-Fab Benchmark:
> Capacity Equivalence and Cycle-Time Sensitivity to an Underspecified
> Batch-Composition Policy," SimTrek Inc., Shizuoka, Japan, 2026.

- **DOI:** `10.5281/zenodo.20635151`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The underlying input master data is the publicly available SMT2020 testbed
(FernUniversität in Hagen,
<https://p2schedgen.fernuni-hagen.de/index.php/downloads/simulation>).
The GoSoL simulation engine itself is **not** part of this release.

## What is in this dataset

Each replication is one independent random realization of the same fab under
the same demand plan; replications differ only in the random seed driving the
stochastic equipment events (breakdowns and the duration of preventive
maintenance). The deterministic preventive-maintenance schedule is identical
across replications (visible as a near-constant PM event count in
`replications.csv`), while the stochastic breakdown realizations vary by seed.

No single replication is privileged: all reported KPIs are the mean (and
standard deviation) across the 30 replications, and the released
`kpis_by_replication.csv` and `wip_daily.csv` let the full confidence band be
reconstructed trial by trial without re-running the engine.

- **Simulation horizon:** 1460 days, 2018-01-01 to 2021-12-31 (UTC)
  (one-year warm-up + three-year measurement window, 2019-01-01 to 2021-12-31).
- **Replications:** 30 (`rep_01` … `rep_30`).

## Directory layout

```
.
├── README.md                  (this file)
├── DATA_DICTIONARY.md         column-level schema for every file
├── replications.csv           manifest: folder ↔ seed, event counts
├── kpis_by_replication.csv    per-replication KPIs (throughput, cycle time, WIP, per-product CT)
├── wip_daily.csv              daily in-system WIP for every replication
├── rep_01.zip                 → rep_01/{lot_log.parquet, equipment_stops.parquet}
├── rep_02.zip                 → rep_02/{…}
│   …
└── rep_30.zip                 → rep_30/{…}
```

Each replication is distributed as a single ZIP archive **`rep_NN.zip`**
(~650 MB) that unpacks to a folder `rep_NN/` containing exactly two Apache
Parquet files:

| file (inside `rep_NN.zip`) | description |
|---|---|
| `lot_log.parquet` | One row per lot per process step: which equipment unit processed it and when. ~38 M rows / replication. |
| `equipment_stops.parquet` | One row per equipment down interval, tagged `PM` (preventive maintenance) or `BD` (breakdown). ~0.47 M rows / replication. |

The two top-level summary files are derived from the per-replication logs and
are provided for convenience:

| file | description |
|---|---|
| `kpis_by_replication.csv` | One row per replication: throughput, mean cycle time (and its SD), Saleable/Engineering means, mean WIP, and per-product mean cycle time. The mean ± SD across the 30 rows reproduces the aggregate and per-product KPIs reported in the paper. |
| `wip_daily.csv` | Daily in-system WIP for every replication (`folder`, `date`, `wip_lots`); 30 × 1461 rows. Lets the WIP trajectory and its confidence band be reconstructed directly. |

`replications.csv` maps each `rep_NN` folder to the random seed used and lists
the PM / breakdown event counts for that replication.

See **`DATA_DICTIONARY.md`** for the full column-level schema.

## Notes on identifiers

- `CaseId` is constant (`"1"`) and `RepId` is constant (`"REP_001"`) across
  every file — they are fixed labels from the simulation runs. **The
  replication is identified by the folder name (`rep_NN`) and its seed in
  `replications.csv`**, not by `RepId`.
- `LotSize` in `lot_log.parquet` is `1.0` for every row; this dataset tracks
  lot-level flow, not per-lot piece counts.

## Loading the data

```python
import pyarrow.parquet as pq

# after unzipping rep_01.zip
lots  = pq.read_table("rep_01/lot_log.parquet")
stops = pq.read_table("rep_01/equipment_stops.parquet")

# breakdown vs preventive-maintenance downtime for one replication
import pyarrow.compute as pc
bd = stops.filter(pc.equal(stops["Type"], "BD"))
pm = stops.filter(pc.equal(stops["Type"], "PM"))
```

## Not included

- `buffer_log` (per-lot buffer-occupancy intervals) — not part of this release;
  it may be added in a later version.
- The GoSoL simulation engine and its dispatching logic.
- SMT2020 input master data — obtain from the source linked above.
