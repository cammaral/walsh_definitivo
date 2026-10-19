# Reproducibility protocol — Sparse Walsh–Pauli Correlation Encoding for MaxCut

**Manuscript:** *Sparse Walsh-Pauli Correlation Encoding for MaxCut: A Quantum-Inspired Formulation with Quantum-Hardware Validation* (manuscript supplied with the project).

**Reference source:** `eldorado_walsh_pce_codigo.zip`, inspected on 19 October 2026. The scientific implementations in `src/walsh_pce/` and the training settings of `stages/` are preserved. This distribution changes only nine scripts' absolute project paths to paths relative to this repository, replaces the absolute MQLib installation path with `MQLIB_ROOT`, and adds the present documentation and auxiliary verification scripts. **No optimization algorithm or published result was recalculated for the preparation of this distribution.**

## 1. Reproducibility scope and status

| Stage | Purpose | Entrypoint | Status in this distribution |
|---|---|---|---|
| 01a | Generate random graph bank | `stages/01_random_graphs/generate_graphs.py` | Source and 1,750 original graph files available |
| 01b | Obtain MQLib heuristic references | `stages/01_random_graphs/run_mqlib.py` | Script available; MQLib executable/revision and output CSV not included |
| 02 | Split and calibrate alpha, beta | `create_split.py`, `run_alpha.py`, `run_beta.py` | Scripts available; generated CSVs excluded |
| 03 | Optimize the complete Walsh basis | `run_complete_basis.py` | Script available; run-level results excluded |
| 04 | Base-mask support sweep | `prepare_support_grid.py`, `run_support_sweep.py` | Scripts available; sweep results excluded |
| 05 | Gset B=1 benchmark and selected coefficients | `run_gset.py` | Script/Gset graph files available; trained `.npz` excluded |
| 06 | Transfer selected coefficients to circuits | `stages/06_transfer/run.py` | Script available; requires Stage 05 outputs; partial original logs retained |
| 07 | Real-device experiment | `stages/07_hardware/run.py` | **Incomplete**: this entrypoint is empty; stored IBM measurement counts are available, but the acquisition/analysis pipeline is missing |
| Figures/tables | Publication graphics | `analysis/paper_figures_tables*.ipynb` | **Incomplete**: notebooks have absolute paths and reference some files not included; `.tex` table snapshots remain under `analysis/tables/` |

The project supports **methodological reproducibility** of Stages 01–06 after installation. Reproducing exactly the historical numerical values requires matching the original library versions, MQLib build, hardware, and execution environment. Stage 07 cannot be replayed end-to-end from this archive alone. A hardware rerun, even with the original code, would produce new noisy measurements rather than identical historical samples.

## Automated setup (new distribution feature)

From the repository root, run `bash scripts/setup.sh`. It creates `.venv` only if absent and does **not** modify an existing virtual environment. It clones and builds MQLib under the ignored `.external/MQLib` directory by default, records the checked-out revision at `results/reproducibility/mqlib_commit.txt`, and uses that location automatically in Stage 01. GNU C++ build tools (`g++`, `make`, `ar`) are prerequisites; on macOS install the Xcode Command Line Tools if missing.

**Important reproducibility limitation:** the original historical MQLib commit is not known. Without `MQLIB_COMMIT`, setup uses the default revision obtained at clone time and records it, which is *not* an exact historical lock. To pin a recovered historical revision, run `MQLIB_COMMIT=<full SHA> bash scripts/setup.sh`. The Python requirements file also remains unpinned until the historical environment is recovered. The script does not start training or overwrite experiment results.

## 2. Environment and dependencies

Run from the repository root, preferably in a fresh copy on Linux or macOS. Python, NumPy, pandas, NetworkX, PyTorch, tqdm, and PennyLane are required; the figure notebooks additionally require matplotlib and IPython/Jupyter. The package list is in `environment/requirements-reproduction.txt`.

```bash
cd /path/to/eldorado_walsh_pce_reproducivel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r environment/requirements-reproduction.txt
```

**Important:** the original `pip freeze`, Python version, CPU architecture, PyTorch/PennyLane versions, and MQLib Git revision were not contained in the supplied archive. Consequently the requirements list is **not** a historically exact lockfile. To recover bitwise numerical reproducibility, obtain that environment information from the original machine. Record the installed environment for each new execution with:

```bash
python scripts/capture_environment.py
```

The capture is saved to `results/reproducibility/environment.json` and `pip_freeze.txt`. Keep these files alongside every set of released results, including the MQLib commit hash. CPU-only versus GPU execution, parallelism, floating-point implementations, and wall-clock stopping conditions may affect the exact results. The scripts use double precision and PyTorch; calibration/optimization use `WORKERS=4` unless otherwise noted.

### External MQLib dependency

Install/build the MQLib command-line executable separately, using the upstream `MQLib` repository (see the manuscript references). The Stage-01 script expects the executable at `$MQLIB_ROOT/bin/MQLib` and runs the hyper-heuristic with arguments `-hh -fM <graph> -r 5 -s <seed>`.

```bash
export MQLIB_ROOT="$(pwd)/.external/MQLib"
test -x "$MQLIB_ROOT/bin/MQLib"  # must succeed
```

The original MQLib installation was outside the uploaded project. No commit hash or binary was present in the archive. MQLib uses **5 seconds per trial**, so its reference values are sensitive to machine performance and operating-system scheduling. The MQLib best-found cut is a heuristic reference, **not** a certified MaxCut optimum.

## 3. Original inputs and integrity checks

The repository includes 250 unweighted Erdős–Rényi instances per graph size for `m = 100, 200, 400, 500, 800, 2000, 7000` (1,750 graph files total) and the Gset inputs `G14.txt`, `G23.txt`, `G60.txt`, `G64.txt`.

The manifest `manifests/input_sha256.tsv` contains SHA-256 checksums of all **1,754** original graph files. Verify inputs **before** execution and again **after** regenerating Stage 01:

```bash
python scripts/check_input_hashes.py
```

A failed hash check means the graph bank is not the original input bank; **do not claim a numerical reproduction** until the discrepancy is understood. The original `data/graphs/graphs.csv` was removed from the provided ZIP and is reconstructed by Stage 01. Regeneration writes into `data/graphs/`, so work from a copy of the repository and do not overwrite the archived original data.

Gset reference cut values adopted in the paper are `G14=3064`, `G23=13344`, `G60=14188`, and `G64=8751`. Their inputs are fixed benchmark files; the reference values are published best-known cuts, not products of the MQLib runs on the random graph bank.

## 4. Exact execution order, stages 01–06

**Run the following from the repository root.** All commands produce files under `results/` or `configs/`. The original scripts resume some partially completed runs by reading existing CSVs. To reproduce from scratch, use a fresh copy with no prior stage CSVs or trained coefficients. Do not mix results from different protocols or environments.

### Stage 01 — random graphs and MQLib

```bash
python stages/01_random_graphs/generate_graphs.py
python scripts/check_input_hashes.py
python stages/01_random_graphs/run_mqlib.py
```

`generate_graphs.py`: master seed **20260623**; `nx.fast_gnp_random_graph`; target degree 6, `p = 6/(m−1)`; reject initial average degree below 3; connect surviving disconnected graphs by adding `c−1` unit-weight edges. The bank has **1,750** accepted instances. `run_mqlib.py`: seeds `1,2,...,10`, 5-second nominal time per seed, **8 concurrent workers**, giving **17,500 MQLib trials**. Keep the output `results/stage_01/mqlib_runs.csv` (all trials) and `mqlib_summary.csv` (highest cut per graph).

### Stage 02 — calibration split and hyperparameters

```bash
python stages/02_calibration/create_split.py
python stages/02_calibration/run_alpha.py
python stages/02_calibration/run_beta.py
```

`create_split.py` uses seed **20260909** and allocates **50 graphs per size to calibration** (350 total); the remaining **200 per size** are held out (1,400 total). Training calibration uses **five initializations `11,22,33,44,55`**, Adam `lr=1e-3`, **150 epochs**, the *complete* Walsh basis, and **4 workers**. Alpha is selected with `beta=0.5`; then beta is selected with alpha held fixed. The alpha candidate grids are defined explicitly in `run_alpha.py` and the beta candidate grid in `run_beta.py`.

Expected selections (`results/stage_02/selected_alpha.csv`, `selected_beta.csv`):

| (q,k) | alpha | beta |
|---|---:|---:|
| (9,4) | 27 | 0.75 |
| (10,5) | 22 | 0.75 |
| (11,5) | 22 | 0.75 |
| (12,6) | 36 | 0.75 |
| (14,7) | 42 | 0.50 |
| (16,8) | 80 | 0.50 |

If recalibration returns different values, investigate the environment and MQLib reference differences; do **not** silently substitute the published values and then label the run an exact replay.

### Stage 03 — complete Walsh basis

```bash
python stages/03_complete_basis/run_complete_basis.py
```

Uses **200 held-out graphs per size**, ten initializations `11,22,33,44,55,66,77,88,99,110`, Adam `lr=1e-3` and **500 epochs**. A total of **14,000 optimizations** is expected (1,400 graphs × 10 seeds). The full representation contains `2^q` coefficients, where `(q,k)` is selected using the encoding map in `src/walsh_pce/core.py` and the number of graph vertices.

Retain `results/stage_03/runs.csv` and `summary.csv`. Expected paper mean empirical AR by graph size: `100:0.9762`, `200:0.9759`, `400:0.9765`, `500:0.9752`, `800:0.9826`, `2000:0.9792`, `7000:0.9860` (printed to four decimal places).

### Stage 04 — structured Walsh support ablation

```bash
python stages/04_support_sweep/prepare_support_grid.py
python stages/04_support_sweep/run_support_sweep.py
```

`prepare_support_grid.py` creates `configs/support_grid.csv` using nested paired supports. Candidate base-mask counts are `B=1,2,4,8,16,32,64` plus the first full-basis saturation value for each size. **B is varied here because this stage is the support ablation.** `run_support_sweep.py` uses the **same 1,400 held-out graphs**, **five seeds** (`11,22,33,44,55`), **500 epochs**, `lr=1e-3`, and compares each sparse solution with the **paired Stage-03 full-basis run**. At saturation, the full-basis endpoint is imported from Stage 03 rather than retrained as another sparse model.

Retain `results/stage_04/runs.csv` and `summary.csv`. Published B=1 mean empirical AR values, ordered by `m = 100,200,400,500,800,2000,7000`, are `0.9770, 0.9769, 0.9787, 0.9771, 0.9804, 0.9835, 0.9850`.

### Stage 05 — final sparse Gset benchmark (**B=1**)

```bash
python stages/05_gset/run_gset.py
```

Uses the four fixed Gset instances, alpha/beta imported from Stage 02, **ten seeds** (`11,22,33,44,55,66,77,88,99,110`), `B=1`, Adam `lr=1e-3` and **500 epochs**. This is the **principal sparse configuration** used in the paper. For each seed, decode the trained correlator signs into a raw MaxCut partition. Refinement is one forward greedy bitflip sweep followed by one backward sweep (only strictly improving flips).

| Gset instance | Paper mean raw AR | Paper mean refined AR | Selected seed (highest raw AR) |
|---|---:|---:|---:|
| G14 | 0.9698 | 0.9828 | 77 |
| G23 | 0.9874 | 0.9926 | 44 |
| G60 | 0.9746 | 0.9781 | 110 |
| G64 | 0.9195 | 0.9369 | 110 |

**Keep all trained `.npz` coefficient files** at `results/stage_05/coefficients/<Gxx>/`, plus `runs.csv`, `summary.csv`, and `selected_best_raw.csv`. These are the provenance of the states prepared in Stage 06. The optional `run_gset_full.py` is a separate additional comparison and is **not** a replacement for `run_gset.py` in this paper's B=1 benchmark.

### Stage 06 — circuit-level parameter transfer

```bash
python stages/06_transfer/run.py --self-test
python stages/06_transfer/run.py --dry-run
python stages/06_transfer/run.py --statevector-only
```

Stage 06 reads the **selected best-raw Stage-05 coefficients**, without retraining. It evaluates the ancilla-based Walsh-loader circuit on the same four Gset instances. Its interference candidates are `epsilon = 0.02, 0.05, 0.10, 0.20, 0.40, 0.80`. In the uploaded final script the strict statevector choice is **epsilon=0.02**; the separate finite-shot/noise operating point is **epsilon=0.20**. Do not confuse these or substitute the older log using epsilon=0.05.

The manuscript's **Table 7** comes from the *strict statevector* result at epsilon 0.02, with target fidelities `G14=0.999981`, `G23=0.999949`, `G60=0.999950`, `G64=0.999953`, and 100% decision-sign agreement. The output is `results/stage_06/05_summary/statevector_summary.csv`. To additionally run the original finite-shot and noisy simulations, execute the script **without** `--statevector-only`:

```bash
python stages/06_transfer/run.py
```

The full Stage-06 run can consume substantial compute time, especially for the noise trajectories. Do not use `--skip-noise` if the intention is to reproduce the complete original simulated protocol. The manuscript's principal statevector table does not require the finite-shot/noise continuation.

## 5. Single-entry scripts (same individual stage commands)

These convenience wrappers do not modify the stage algorithms and deliberately require an explicit invocation:

```bash
# Execute stages 01–05 sequentially; long-running
bash scripts/reproduce_classical.sh --run

# Stage 06 after Stage 05 coefficient archives have been rebuilt
bash scripts/reproduce_transfer.sh --self-test
bash scripts/reproduce_transfer.sh --dry-run
bash scripts/reproduce_transfer.sh --statevector

# Optional complete shot/noise protocol
bash scripts/reproduce_transfer.sh --full
```

Do not run the wrapper in a directory containing partially reconstructed `results/` from a different version. When resuming an interrupted run, retain its existing CSV, NPZ, environment manifest, and graph files. The wrappers create execution logs under `results/reproducibility/`.

## 6. Numerical verification

```bash
python scripts/verify_paper_results.py --through 05
python scripts/verify_paper_results.py --through 06
```

The verifier checks the graph count/split, alpha-beta values, Stage-03 mean AR, Stage-04 B=1 mean AR, Stage-05 raw/refined mean AR, and Stage-06 statevector fidelities/sign agreement against **the rounded numbers printed in the manuscript**. The allowed tolerances reflect published rounding, **not** an assertion of exact bitwise identity. A failed comparison is a result requiring investigation, not a reason to manually edit any numerical output.

For each completed study, archive the stage `runs.csv`/summary files, trained `.npz`, original input hashes, `pip_freeze.txt`, `environment.json`, MQLib version, and any full-run logs. Maintain a source-code commit hash and an output manifest in the release. **This file alone is not a substitute for the missing historical environment lock.**

## 7. Hardware data, reconstruction limitations, and figures

The included `reference/hardware_n15_final/` contains the 15-vertex, 30-edge, 4-regular MaxCut specification and ten IBM hardware count JSONs for the `ibm_fez` experiment, together with some submission and layout metadata. The article reports data register `q=6`, one ancilla (7 qubits total), ten training seeds `0,...,9`, 500 epochs, Adam `lr=1e-3`, manually fixed `alpha=15` and `beta=0.1`, hardware interference `epsilon=0.6`, and 32,768 requested shots per seed. The reference is the **exact noiseless finite-epsilon circuit**; it is **not** the ideal epsilon→0 Walsh state.

The original measured post-selection rate was `0.169 ± 0.004`, sign agreement `0.927 ± 0.049`, raw AR `0.917 ± 0.068`, and refined AR `0.950 ± 0.043` (Table 8). The Stage-07 `run.py` file is empty. Without the original device-submission scripts, circuit transpilation/layout code, analysis scripts, saved original training coefficients, and readout-bit mapping, the hardware figure and full analysis **cannot be independently reproduced from this archive**. Reconstruct these exact scripts and artifacts from the original hardware repository; do not attempt to fill the gap with a newly designed pipeline and call it a reproduction. The original additional shot-budget submissions (8,192 and 65,536) also lack all required artifacts in this ZIP.

The original publication notebooks `analysis/paper_figures_tables.ipynb` and `analysis/paper_figures_tables(2).ipynb` are present. However, they refer to paths outside this repository and missing CSV data including the hardware correlator, shot sweep, and transpilation summaries. The saved `analysis/tables/*.tex` files can be used to inspect the published table snapshots. **Generating all paper figures from scratch remains blocked** until the missing analysis inputs are recovered and the notebooks are made path-independent.

## 8. Provenance and scientific caveats

- **Classical training ≠ quantum training.** The training and objective evaluation use classical normalized XOR autocorrelations. The circuit serves to validate parameter transfer, not to optimize the MaxCut objective on hardware.
- **Random-graph AR values are empirical** and divided by the best observed MQLib heuristic cut, not by a known optimum. Values slightly above 1 are possible.
- **B=1 is fixed for final sparse Gset results;** varying B is reserved for the Stage-04 ablation. The complete Walsh basis is the Stage-03 comparison reference.
- **Same method does not guarantee identical floats.** Exact historical replication additionally needs original library/OS/hardware versions, numerical backend settings, and MQLib build. Without these, report agreement within tolerances rather than asserting bitwise reproducibility.
- **Data access and code rights:** the manuscript identifies institutional restrictions on code and some data. Confirm authorization before distributing the archive outside the project or institution.
