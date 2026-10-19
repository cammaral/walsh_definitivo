from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from walsh_pce.core import (
    SIZE_CONFIG,
    load_graph,
    train_complete,
)

from walsh_pce.protocol import (
    PROTOCOL_ID,
    validate_protocol_dataframe,
)


SELECTED_ALPHA_CSV = (
    ROOT
    / "results"
    / "stage_02"
    / "selected_alpha.csv"
)

SELECTED_BETA_CSV = (
    ROOT
    / "results"
    / "stage_02"
    / "selected_beta.csv"
)

INIT_SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 110,
]

EPOCHS = 500
LR = 1e-3

MASTER_TRAIN_SEED = 20260909
WORKERS = 4

GRAPHS_CSV = ROOT / "data" / "graphs" / "graphs.csv"
SPLIT_CSV = ROOT / "data" / "graphs" / "split.csv"

MQLIB_CSV = (
    ROOT
    / "results"
    / "stage_01"
    / "mqlib_summary.csv"
)

OUT_DIR = ROOT / "results" / "stage_03"

RUNS_CSV = OUT_DIR / "runs.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"


def stable_seed(*parts):

    text = "|".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        int(digest[:16], 16)
        % (2**32 - 1)
    )


def _validate_selected(
    df,
    value_column,
    source,
):

    validate_protocol_dataframe(
        df,
        source,
    )

    required = {
        "q",
        "k",
        value_column,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"{source} missing columns: "
            f"{sorted(missing)}"
        )

    if df[
        ["q", "k"]
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate (q,k) in {source}."
        )

    expected_pairs = set(
        SIZE_CONFIG.values()
    )

    actual_pairs = {
        (
            int(row.q),
            int(row.k),
        )
        for row in df.itertuples(
            index=False
        )
    }

    if actual_pairs != expected_pairs:

        raise RuntimeError(
            f"{source} has wrong (q,k) coverage. "
            f"Found {sorted(actual_pairs)}, "
            f"expected {sorted(expected_pairs)}."
        )


def load_hyperparameters():

    alpha_df = pd.read_csv(
        SELECTED_ALPHA_CSV
    )

    beta_df = pd.read_csv(
        SELECTED_BETA_CSV
    )

    _validate_selected(
        alpha_df,
        "alpha",
        "selected_alpha.csv",
    )

    _validate_selected(
        beta_df,
        "beta",
        "selected_beta.csv",
    )

    alpha_map = {
        (
            int(row.q),
            int(row.k),
        ):
        float(row.alpha)

        for row
        in alpha_df.itertuples(
            index=False
        )
    }

    beta_map = {
        (
            int(row.q),
            int(row.k),
        ):
        float(row.beta)

        for row
        in beta_df.itertuples(
            index=False
        )
    }

    if "alpha" in beta_df.columns:

        beta_alpha_map = {
            (
                int(row.q),
                int(row.k),
            ):
            float(row.alpha)

            for row
            in beta_df.itertuples(
                index=False
            )
        }

        for pair in alpha_map:

            if not np.isclose(
                beta_alpha_map[pair],
                alpha_map[pair],
                atol=0.0,
                rtol=0.0,
            ):

                raise RuntimeError(
                    "selected_beta.csv was produced "
                    "with a different alpha for "
                    f"{pair}: "
                    f"{beta_alpha_map[pair]} != "
                    f"{alpha_map[pair]}"
                )

    return (
        alpha_map,
        beta_map,
    )


def load_evaluation_table():

    graphs = pd.read_csv(
        GRAPHS_CSV
    )

    split = pd.read_csv(
        SPLIT_CSV
    )

    refs = pd.read_csv(
        MQLIB_CSV
    )

    df = graphs.merge(
        split,
        on=[
            "n",
            "graph_id",
        ],
        validate="one_to_one",
    )

    df = df[
        df["split"]
        == "evaluation"
    ].copy()

    df = df.merge(
        refs[
            [
                "n",
                "graph_id",
                "C_MQLib",
            ]
        ],
        on=[
            "n",
            "graph_id",
        ],
        validate="one_to_one",
    )

    counts = (
        df.groupby("n")
        .size()
        .to_dict()
    )

    for n in SIZE_CONFIG:

        if counts.get(n) != 200:

            raise RuntimeError(
                f"Expected 200 evaluation graphs "
                f"for n={n}, "
                f"found {counts.get(n)}"
            )

    return df


def run_one(
    n,
    graph_id,
    relative_path,
    c_ref,
    init_seed,
    alpha_map,
    beta_map,
):

    q, k = SIZE_CONFIG[n]

    alpha = float(
        alpha_map[
            (q, k)
        ]
    )

    beta = float(
        beta_map[
            (q, k)
        ]
    )

    actual_seed = stable_seed(
        MASTER_TRAIN_SEED,
        n,
        graph_id,
        init_seed,
    )

    graph_path = (
        ROOT
        / "data"
        / "graphs"
        / relative_path
    )

    graph = load_graph(
        str(graph_path)
    )

    start = time.perf_counter()

    result = train_complete(
        graph=graph,
        q=q,
        k=k,
        alpha=alpha,
        beta=beta,
        seed=actual_seed,
        epochs=EPOCHS,
        lr=LR,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    cut = float(
        result["cut"]
    )

    return {
        "protocol_id": PROTOCOL_ID,
        "n": int(n),
        "q": int(q),
        "k": int(k),
        "graph_id": str(graph_id),
        "init_seed": int(init_seed),
        "actual_seed": int(actual_seed),
        "alpha": alpha,
        "beta": beta,
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "cut": cut,
        "C_MQLib": float(c_ref),
        "AR": (
            cut
            / float(c_ref)
        ),
        "final_loss": float(
            result["loss"]
        ),
        "final_norm": float(
            result["norm"]
        ),
        "runtime_sec": float(
            elapsed
        ),
    }


FIELDS = [
    "protocol_id",
    "n",
    "q",
    "k",
    "graph_id",
    "init_seed",
    "actual_seed",
    "alpha",
    "beta",
    "epochs",
    "lr",
    "cut",
    "C_MQLib",
    "AR",
    "final_loss",
    "final_norm",
    "runtime_sec",
]


def completed_keys(
    alpha_map,
    beta_map,
):

    if not RUNS_CSV.exists():

        return set()

    df = pd.read_csv(
        RUNS_CSV
    )

    validate_protocol_dataframe(
        df,
        str(RUNS_CSV),
    )

    required = set(
        FIELDS
    )

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Existing Stage-03 runs have "
            "incompatible schema. Missing: "
            f"{sorted(missing)}"
        )

    for row in df.itertuples(
        index=False
    ):

        n = int(
            row.n
        )

        q, k = SIZE_CONFIG[n]

        if (
            int(row.q),
            int(row.k),
        ) != (
            q,
            k,
        ):

            raise RuntimeError(
                "Stage-03 q,k mismatch "
                f"for n={n}."
            )

        expected_alpha = float(
            alpha_map[
                (q, k)
            ]
        )

        expected_beta = float(
            beta_map[
                (q, k)
            ]
        )

        if not np.isclose(
            float(row.alpha),
            expected_alpha,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Existing Stage-03 alpha "
                "does not match current calibration."
            )

        if not np.isclose(
            float(row.beta),
            expected_beta,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Existing Stage-03 beta "
                "does not match current calibration."
            )

        if int(row.epochs) != EPOCHS:

            raise RuntimeError(
                "Existing Stage-03 epoch count "
                "does not match current protocol."
            )

        if not np.isclose(
            float(row.lr),
            LR,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Existing Stage-03 learning rate "
                "does not match current protocol."
            )

    return set(
        zip(
            df["n"].astype(int),
            df["graph_id"].astype(str),
            df["init_seed"].astype(int),
        )
    )


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

        writer.writerow(
            row
        )


def build_summary():

    runs = pd.read_csv(
        RUNS_CSV
    )

    validate_protocol_dataframe(
        runs,
        str(RUNS_CSV),
    )

    graph_best = (
        runs
        .groupby(
            [
                "n",
                "graph_id",
            ],
            as_index=False,
        )
        .agg(
            best_AR=(
                "AR",
                "max",
            ),
            mean_AR_graph=(
                "AR",
                "mean",
            ),
        )
    )

    base = (
        runs
        .groupby(
            [
                "n",
                "q",
                "k",
                "alpha",
                "beta",
            ],
            as_index=False,
        )
        .agg(
            graphs=(
                "graph_id",
                "nunique",
            ),
            runs=(
                "AR",
                "size",
            ),
            mean_AR=(
                "AR",
                "mean",
            ),
            std_AR=(
                "AR",
                "std",
            ),
            min_AR=(
                "AR",
                "min",
            ),
            max_AR=(
                "AR",
                "max",
            ),
            mean_runtime_sec=(
                "runtime_sec",
                "mean",
            ),
            max_norm_error=(
                "final_norm",
                lambda x:
                float(
                    (
                        x - 1.0
                    )
                    .abs()
                    .max()
                ),
            ),
        )
    )

    best_summary = (
        graph_best
        .groupby(
            "n",
            as_index=False,
        )
        .agg(
            mean_best_AR=(
                "best_AR",
                "mean",
            ),
            std_best_AR=(
                "best_AR",
                "std",
            ),
            mean_graph_AR=(
                "mean_AR_graph",
                "mean",
            ),
        )
    )

    summary = base.merge(
        best_summary,
        on="n",
        validate="one_to_one",
    )

    summary = (
        summary
        .sort_values("n")
        .reset_index(
            drop=True
        )
    )

    summary.insert(
        0,
        "protocol_id",
        PROTOCOL_ID,
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    return summary


def expected_run_count():

    return (
        200
        * len(SIZE_CONFIG)
        * len(INIT_SEEDS)
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(1)

    (
        alpha_map,
        beta_map,
    ) = load_hyperparameters()

    df = load_evaluation_table()

    completed = completed_keys(
        alpha_map,
        beta_map,
    )

    tasks = []

    for row in df.itertuples(
        index=False
    ):

        for init_seed in INIT_SEEDS:

            key = (
                int(row.n),
                str(row.graph_id),
                int(init_seed),
            )

            if key in completed:
                continue

            tasks.append(
                (
                    int(row.n),
                    str(row.graph_id),
                    str(row.path),
                    float(row.C_MQLib),
                    int(init_seed),
                    alpha_map,
                    beta_map,
                )
            )

    expected = expected_run_count()

    print("=" * 70)
    print("STAGE 03 - COMPLETE BASIS")
    print("=" * 70)

    print(
        f"Evaluation graphs: {len(df)}"
    )

    print(
        f"Seeds: {INIT_SEEDS}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LR}"
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

    print(
        "Alpha/beta by (q,k):"
    )

    for pair in sorted(
        alpha_map
    ):

        print(
            f"  {pair}: "
            f"alpha={alpha_map[pair]}, "
            f"beta={beta_map[pair]}"
        )

    print()

    if tasks:

        with ThreadPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            futures = [
                executor.submit(
                    run_one,
                    *task,
                )
                for task in tasks
            ]

            for future in tqdm(
                as_completed(
                    futures
                ),
                total=len(
                    futures
                ),
                desc="complete basis",
            ):

                append_row(
                    future.result()
                )

    runs = pd.read_csv(
        RUNS_CSV
    )

    validate_protocol_dataframe(
        runs,
        str(RUNS_CSV),
    )

    if len(runs) != expected:

        raise RuntimeError(
            f"Expected {expected} total runs, "
            f"found {len(runs)}."
        )

    if runs[
        [
            "n",
            "graph_id",
            "init_seed",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Stage-03 runs."
        )

    if not np.allclose(
        runs["final_norm"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Stage-03 normalization "
            "audit failed."
        )

    summary = build_summary()

    print()

    print(
        summary[
            [
                "n",
                "q",
                "k",
                "alpha",
                "beta",
                "graphs",
                "runs",
                "mean_AR",
                "std_AR",
                "mean_best_AR",
                "std_best_AR",
                "min_AR",
                "max_AR",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
