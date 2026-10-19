from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ============================================================
# LOAD FROZEN STAGE-05 IMPLEMENTATION
# ============================================================

BASE_SCRIPT = (
    ROOT
    / "stages"
    / "05_gset"
    / "run_gset.py"
)

spec = importlib.util.spec_from_file_location(
    "stage05_base",
    BASE_SCRIPT,
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


from walsh_pce.initialization import (
    initialize_complete,
)
from walsh_pce.core import (
    DTYPE,
    decision_masks,
    load_graph,
    pce_loss,
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 500
LR = 1e-3
WORKERS = 4

MASTER_TRAIN_SEED = 20260909

INIT_SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 110,
]

CONFIG = base.CONFIG

OUT_DIR = (
    ROOT
    / "results"
    / "stage_05_full"
)

ARRAY_DIR = (
    OUT_DIR
    / "coefficients"
)

RUNS_CSV = (
    OUT_DIR
    / "runs.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "summary.csv"
)

SELECTED_CSV = (
    OUT_DIR
    / "selected_best_raw.csv"
)

COMPARISON_CSV = (
    ROOT
    / "results"
    / "stage_05_comparison.csv"
)


# ============================================================
# VALIDATE FULL-BASIS LOSS
# ============================================================

def validate_full_basis(
    instances,
):

    name = "G14"

    cfg = CONFIG[name]
    data = instances[name]

    n = cfg["n"]
    q = cfg["q"]
    k = cfg["k"]
    alpha = base.ALPHA_BY_QK[(q, k)]
    beta = base.BETA_BY_QK[(q, k)]

    graph = load_graph(
        str(data["path"])
    )

    masks = decision_masks(
        q=q,
        k=k,
        m=n,
    )

    N = 2**q

    seed = base.stable_seed(
        MASTER_TRAIN_SEED,
        n,
        name,
        INIT_SEEDS[0],
    )

    a = initialize_complete(
        n_coeffs=N,
        variable_masks=masks,
        seed=seed,
        dtype=DTYPE,
    )

    with torch.no_grad():

        core_loss, core_c = pce_loss(
            a=a,
            graph=graph,
            masks=masks,
            alpha=alpha,
            beta=beta,
        )

        new_loss, new_c, _ = (
            base.gset_loss(
                full=a,
                masks=masks,
                edge_u=data["edge_u_t"],
                edge_v=data["edge_v_t"],
                edge_w=data["edge_w_t"],
                alpha=alpha,
                beta=beta,
                nu=data["nu"],
            )
        )

    max_c_error = float(
        torch.max(
            torch.abs(
                core_c - new_c
            )
        ).item()
    )

    loss_error = abs(
        float(core_loss.item())
        - float(new_loss.item())
    )

    print()
    print("=" * 80)
    print("FULL-BASIS LOSS VALIDATION")
    print("=" * 80)

    print(
        f"max correlator error: "
        f"{max_c_error:.3e}"
    )

    print(
        f"loss error: "
        f"{loss_error:.3e}"
    )

    if max_c_error > 1e-10:
        raise RuntimeError(
            "Full-basis correlator validation failed."
        )

    if loss_error > 1e-8:
        raise RuntimeError(
            "Full-basis loss validation failed."
        )

    print(
        "Full Gset loss == frozen core loss: OK"
    )


# ============================================================
# TRAIN ONE FULL-BASIS MODEL
# ============================================================

def train_one(
    name,
    init_seed,
    data,
):

    cfg = CONFIG[name]

    n = cfg["n"]
    q = cfg["q"]
    k = cfg["k"]
    alpha = base.ALPHA_BY_QK[(q, k)]
    beta = base.BETA_BY_QK[(q, k)]
    reference = cfg["reference"]

    N = 2**q

    masks = decision_masks(
        q=q,
        k=k,
        m=n,
    )

    actual_seed = base.stable_seed(
        MASTER_TRAIN_SEED,
        n,
        name,
        init_seed,
    )

    # ========================================================
    # FULL BASIS
    # ========================================================

    a = initialize_complete(
        n_coeffs=N,
        variable_masks=masks,
        seed=actual_seed,
        dtype=DTYPE,
    )

    a.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [a],
        lr=LR,
    )

    start = time.perf_counter()

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    for _ in range(EPOCHS):

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, _, _ = base.gset_loss(
            full=a,
            masks=masks,
            edge_u=data["edge_u_t"],
            edge_v=data["edge_v_t"],
            edge_w=data["edge_w_t"],
            alpha=alpha,
            beta=beta,
            nu=data["nu"],
        )

        loss.backward()

        optimizer.step()

        with torch.no_grad():

            a /= torch.linalg.vector_norm(
                a
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    with torch.no_grad():

        final_loss, c, s = (
            base.gset_loss(
                full=a,
                masks=masks,
                edge_u=data["edge_u_t"],
                edge_v=data["edge_v_t"],
                edge_w=data["edge_w_t"],
                alpha=alpha,
                beta=beta,
                nu=data["nu"],
            )
        )

        final_norm = float(
            torch.linalg.vector_norm(
                a
            ).item()
        )

        coefficients = (
            a.detach()
            .cpu()
            .numpy()
            .copy()
        )

        correlators = (
            c.detach()
            .cpu()
            .numpy()
            .copy()
        )

        relaxed_spins = (
            s.detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ========================================================
    # RAW DECODING
    # ========================================================

    raw_spins = np.where(
        correlators >= 0.0,
        1,
        -1,
    ).astype(
        np.int8
    )

    raw_cut = base.cut_value(
        raw_spins,
        data["u"],
        data["v"],
        data["w"],
    )

    raw_AR = (
        raw_cut
        / reference
    )

    # ========================================================
    # SAME LOCAL SEARCH AS B1/B2
    # ========================================================

    (
        local_spins,
        flips_forward,
        flips_backward,
    ) = base.local_search_forward_backward(
        raw_spins,
        data["adjacency"],
    )

    local_cut = base.cut_value(
        local_spins,
        data["u"],
        data["v"],
        data["w"],
    )

    local_AR = (
        local_cut
        / reference
    )

    if local_cut + 1e-12 < raw_cut:

        raise RuntimeError(
            f"{name} seed {init_seed}: "
            f"local search decreased cut "
            f"{raw_cut} -> {local_cut}"
        )

    runtime = (
        time.perf_counter()
        - start
    )

    # ========================================================
    # SAVE
    # ========================================================

    instance_dir = (
        ARRAY_DIR
        / name
    )

    npz_path = (
        instance_dir
        / f"{name}_seed{init_seed:03d}.npz"
    )

    base.save_npz_atomic(
        npz_path,

        active_indices=np.arange(
            N,
            dtype=np.int64,
        ),

        coefficients=coefficients,

        correlators=correlators,

        relaxed_spins=relaxed_spins,

        raw_spins=raw_spins,

        local_spins=local_spins,
    )

    return {
        "instance": name,
        "n": n,
        "edges": len(data["w"]),
        "components": (
            data["n_components"]
        ),
        "q": q,
        "k": k,
        "support": "full",
        "active_size": N,
        "full_size": N,
        "active_fraction": 1.0,
        "alpha": alpha,
        "beta": beta,
        "nu": data["nu"],
        "weight_abs_sum": (
            data["weight_abs_sum"]
        ),
        "msf_abs": (
            data["msf_abs"]
        ),
        "epochs": EPOCHS,
        "lr": LR,
        "init_seed": init_seed,
        "actual_seed": actual_seed,
        "reference": reference,
        "raw_cut": raw_cut,
        "raw_AR": raw_AR,
        "local_cut": local_cut,
        "local_AR": local_AR,
        "local_gain": (
            local_cut
            - raw_cut
        ),
        "flips_forward": (
            flips_forward
        ),
        "flips_backward": (
            flips_backward
        ),
        "final_loss": float(
            final_loss.item()
        ),
        "final_norm": final_norm,
        "runtime_sec": runtime,
        "npz": str(
            npz_path.relative_to(
                ROOT
            )
        ),
    }


# ============================================================
# CSV
# ============================================================

FIELDS = [
    "instance",
    "n",
    "edges",
    "components",
    "q",
    "k",
    "support",
    "active_size",
    "full_size",
    "active_fraction",
    "alpha",
    "beta",
    "nu",
    "weight_abs_sum",
    "msf_abs",
    "epochs",
    "lr",
    "init_seed",
    "actual_seed",
    "reference",
    "raw_cut",
    "raw_AR",
    "local_cut",
    "local_AR",
    "local_gain",
    "flips_forward",
    "flips_backward",
    "final_loss",
    "final_norm",
    "runtime_sec",
    "npz",
]


def append_row(row):

    exists = RUNS_CSV.exists()

    with RUNS_CSV.open(
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def completed_keys():

    if not RUNS_CSV.exists():
        return set()

    df = pd.read_csv(
        RUNS_CSV
    )

    return set(
        zip(
            df["instance"].astype(str),
            df["init_seed"].astype(int),
        )
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary():

    runs = pd.read_csv(
        RUNS_CSV
    )

    summary = (
        runs
        .groupby(
            "instance",
            as_index=False,
        )
        .agg(
            n=("n", "first"),
            edges=("edges", "first"),
            components=(
                "components",
                "first",
            ),
            q=("q", "first"),
            k=("k", "first"),
            active_size=(
                "active_size",
                "first",
            ),
            reference=(
                "reference",
                "first",
            ),
            runs=(
                "init_seed",
                "size",
            ),
            mean_raw_cut=(
                "raw_cut",
                "mean",
            ),
            std_raw_cut=(
                "raw_cut",
                "std",
            ),
            best_raw_cut=(
                "raw_cut",
                "max",
            ),
            mean_raw_AR=(
                "raw_AR",
                "mean",
            ),
            std_raw_AR=(
                "raw_AR",
                "std",
            ),
            mean_local_cut=(
                "local_cut",
                "mean",
            ),
            std_local_cut=(
                "local_cut",
                "std",
            ),
            best_local_cut=(
                "local_cut",
                "max",
            ),
            mean_local_AR=(
                "local_AR",
                "mean",
            ),
            std_local_AR=(
                "local_AR",
                "std",
            ),
            mean_local_gain=(
                "local_gain",
                "mean",
            ),
            mean_runtime_sec=(
                "runtime_sec",
                "mean",
            ),
        )
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    selected = (
        runs
        .sort_values(
            [
                "instance",
                "raw_cut",
                "init_seed",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .groupby(
            "instance",
            as_index=False,
        )
        .head(1)
        .sort_values(
            "instance"
        )
        [
            [
                "instance",
                "init_seed",
                "raw_cut",
                "raw_AR",
                "local_cut",
                "local_AR",
                "npz",
            ]
        ]
    )

    selected.to_csv(
        SELECTED_CSV,
        index=False,
    )

    return summary, selected


# ============================================================
# B1 / B2 / FULL COMPARISON
# ============================================================

def build_comparison(
    full_summary,
):

    b1_path = (
        ROOT
        / "results"
        / "stage_05"
        / "summary.csv"
    )

    if not b1_path.exists():
        raise FileNotFoundError(
            b1_path
        )

    b1 = pd.read_csv(
        b1_path
    )

    base.validate_protocol_dataframe(
        b1,
        source=str(b1_path),
    )

    base.validate_protocol_dataframe(
        full_summary,
        source="full_summary",
    )

    expected = set(
        CONFIG.keys()
    )

    if set(
        b1["instance"].astype(str)
    ) != expected:
        raise RuntimeError(
            "B1 summary instance coverage mismatch."
        )

    if set(
        full_summary["instance"].astype(str)
    ) != expected:
        raise RuntimeError(
            "FULL summary instance coverage mismatch."
        )

    rows = []

    for row in b1.itertuples(
        index=False
    ):

        rows.append(
            {
                "protocol_id":
                    base.PROTOCOL_ID,
                "instance":
                    row.instance,
                "support":
                    "B1",
                "active_size":
                    row.active_size,
                "mean_raw_AR":
                    row.mean_raw_AR,
                "std_raw_AR":
                    row.std_raw_AR,
                "best_raw_cut":
                    row.best_raw_cut,
                "mean_local_AR":
                    row.mean_local_AR,
                "std_local_AR":
                    row.std_local_AR,
                "best_local_cut":
                    row.best_local_cut,
            }
        )

    for row in full_summary.itertuples(
        index=False
    ):

        rows.append(
            {
                "protocol_id":
                    base.PROTOCOL_ID,
                "instance":
                    row.instance,
                "support":
                    "FULL",
                "active_size":
                    row.active_size,
                "mean_raw_AR":
                    row.mean_raw_AR,
                "std_raw_AR":
                    row.std_raw_AR,
                "best_raw_cut":
                    row.best_raw_cut,
                "mean_local_AR":
                    row.mean_local_AR,
                "std_local_AR":
                    row.std_local_AR,
                "best_local_cut":
                    row.best_local_cut,
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    order = pd.CategoricalDtype(
        ["B1", "FULL"],
        ordered=True,
    )

    comparison["support"] = (
        comparison["support"]
        .astype(order)
    )

    comparison = (
        comparison
        .sort_values(
            [
                "instance",
                "support",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    comparison.to_csv(
        COMPARISON_CSV,
        index=False,
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARRAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(1)

    (
        base.ALPHA_BY_QK,
        base.BETA_BY_QK,
    ) = base.load_hyperparameters()

    # Same audited instances and nu as Stage 05.
    instances = base.prepare_instances()

    validate_full_basis(
        instances
    )

    completed = completed_keys()

    tasks = []

    for name in CONFIG:

        for init_seed in INIT_SEEDS:

            key = (
                name,
                init_seed,
            )

            if key in completed:
                continue

            expected_npz = (
                ARRAY_DIR
                / name
                / f"{name}_seed{init_seed:03d}.npz"
            )

            if expected_npz.exists():

                raise RuntimeError(
                    f"NPZ exists without CSV row: "
                    f"{expected_npz}"
                )

            tasks.append(
                (
                    name,
                    init_seed,
                )
            )

    expected = (
        len(CONFIG)
        * len(INIT_SEEDS)
    )

    print()
    print("=" * 80)
    print("STAGE 05 - GSET FULL BASIS")
    print("=" * 80)

    print(
        "Support: complete Walsh basis"
    )

    print(
        "Beta: calibrated by (q,k)"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LR}"
    )

    print(
        f"Seeds: {INIT_SEEDS}"
    )

    print(
        f"Workers: {WORKERS}"
    )

    print(
        f"Expected runs: {expected}"
    )

    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    print(
        f"Remaining: "
        f"{len(tasks)}"
    )

    print()

    if tasks:

        with ThreadPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    train_one,
                    name,
                    init_seed,
                    instances[name],
                ): (
                    name,
                    init_seed,
                )
                for (
                    name,
                    init_seed,
                ) in tasks
            }

            for future in tqdm(
                as_completed(
                    futures
                ),
                total=len(
                    futures
                ),
                desc="Gset full",
            ):

                name, seed = (
                    futures[future]
                )

                try:

                    result = (
                        future.result()
                    )

                except Exception:

                    print()
                    print(
                        f"FAILED: "
                        f"{name}, seed={seed}"
                    )

                    raise

                append_row(
                    result
                )

    runs = pd.read_csv(
        RUNS_CSV
    )

    if len(runs) != expected:

        raise RuntimeError(
            f"Expected {expected} runs, "
            f"found {len(runs)}"
        )

    if runs[
        [
            "instance",
            "init_seed",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate full-basis runs."
        )

    if not np.allclose(
        runs["final_norm"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Normalization audit failed."
        )

    if (
        runs["local_cut"]
        + 1e-12
        < runs["raw_cut"]
    ).any():

        raise RuntimeError(
            "Local-search audit failed."
        )

    summary, selected = (
        build_summary()
    )

    comparison = (
        build_comparison(
            summary
        )
    )

    print()
    print("=" * 105)
    print("FULL BASIS FINISHED")
    print("=" * 105)
    print()

    print(
        summary[
            [
                "instance",
                "n",
                "q",
                "k",
                "active_size",
                "reference",
                "mean_raw_cut",
                "std_raw_cut",
                "mean_raw_AR",
                "mean_local_cut",
                "std_local_cut",
                "mean_local_AR",
                "best_local_cut",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 105)
    print("B1 vs FULL")
    print("=" * 105)
    print()

    print(
        comparison[
            [
                "instance",
                "support",
                "active_size",
                "mean_raw_AR",
                "std_raw_AR",
                "mean_local_AR",
                "std_local_AR",
                "best_local_cut",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 105)
    print(
        "SELECTED BEST FULL-BASIS MODELS"
    )
    print("=" * 105)
    print()

    print(
        selected.to_string(
            index=False
        )
    )

    print()
    print(
        f"Runs: {RUNS_CSV}"
    )

    print(
        f"Summary: {SUMMARY_CSV}"
    )

    print(
        f"Comparison: {COMPARISON_CSV}"
    )

    print(
        f"Coefficients: {ARRAY_DIR}"
    )


if __name__ == "__main__":
    main()
