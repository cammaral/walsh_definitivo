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

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from walsh_pce.core import (
    SIZE_CONFIG,
    load_graph,
    train_complete,
)

from walsh_pce.protocol import (
    PROTOCOL_ID,
    validate_protocol_dataframe,
)


BETA = 0.5

INIT_SEEDS = [
    11,
    22,
    33,
    44,
    55,
]

EPOCHS = 150
LR = 1e-3

ALPHA_GRID_BY_QK = {
    (9, 4): [
        9, 18, 27, 36, 45,
    ],
    (10, 5): [
        11, 22, 33, 44, 55,
    ],
    (11, 5): [
        11, 22, 33, 44, 55,
    ],
    (12, 6): [
        12, 24, 36, 48, 60,
    ],
    (14, 7): [
        14, 28, 42, 56, 70,
    ],
    (16, 8): [
        16, 32, 48, 64,
        80, 96, 112, 128,
    ],
}

MASTER_TRAIN_SEED = 20260909

WORKERS = 4


GRAPHS_CSV = (
    ROOT
    / "data"
    / "graphs"
    / "graphs.csv"
)

SPLIT_CSV = (
    ROOT
    / "data"
    / "graphs"
    / "split.csv"
)

MQLIB_CSV = (
    ROOT
    / "results"
    / "stage_01"
    / "mqlib_summary.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "stage_02"
)

RUNS_CSV = (
    OUT_DIR
    / "alpha_runs.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "alpha_summary.csv"
)

SELECTED_CSV = (
    OUT_DIR
    / "selected_alpha.csv"
)


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


def load_calibration_table():

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
        == "calibration"
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

        if counts.get(n) != 50:

            raise RuntimeError(
                f"Expected 50 calibration "
                f"graphs for n={n}, "
                f"found {counts.get(n)}"
            )

    return df


def run_one(
    n,
    graph_id,
    relative_path,
    c_ref,
    init_seed,
    alpha,
):

    q, k = SIZE_CONFIG[n]

    if (q, k) not in ALPHA_GRID_BY_QK:

        raise RuntimeError(
            f"No alpha grid for "
            f"(q,k)=({q},{k})"
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
        alpha=float(alpha),
        beta=BETA,
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

    ar = (
        cut
        / float(c_ref)
    )

    return {
        "protocol_id": PROTOCOL_ID,
        "n": int(n),
        "q": int(q),
        "k": int(k),
        "graph_id": str(graph_id),
        "init_seed": int(init_seed),
        "actual_seed": int(actual_seed),
        "alpha": int(alpha),
        "beta": float(BETA),
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "cut": cut,
        "C_MQLib": float(c_ref),
        "AR": float(ar),
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


def completed_keys():

    if not RUNS_CSV.exists():
        return set()

    df = pd.read_csv(
        RUNS_CSV
    )

    validate_protocol_dataframe(
        df,
        str(RUNS_CSV),
    )

    required = {
        "n",
        "graph_id",
        "init_seed",
        "alpha",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Existing alpha_runs.csv has "
            "incompatible schema. Missing: "
            f"{sorted(missing)}"
        )

    return set(
        zip(
            df["n"].astype(int),
            df["graph_id"].astype(str),
            df["init_seed"].astype(int),
            df["alpha"].astype(int),
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

    required = {
        "q",
        "k",
        "alpha",
        "beta",
        "AR",
        "cut",
        "runtime_sec",
    }

    missing = (
        required
        - set(runs.columns)
    )

    if missing:

        raise RuntimeError(
            "alpha_runs.csv missing columns: "
            f"{sorted(missing)}"
        )

    summary = (
        runs
        .groupby(
            [
                "q",
                "k",
                "alpha",
                "beta",
            ],
            as_index=False,
        )
        .agg(
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
            mean_cut=(
                "cut",
                "mean",
            ),
            mean_runtime_sec=(
                "runtime_sec",
                "mean",
            ),
        )
    )

    p5 = (
        runs
        .groupby(
            [
                "q",
                "k",
                "alpha",
                "beta",
            ]
        )["AR"]
        .quantile(0.05)
        .reset_index(
            name="p5_AR"
        )
    )

    summary = summary.merge(
        p5,
        on=[
            "q",
            "k",
            "alpha",
            "beta",
        ],
        validate="one_to_one",
    )

    summary = summary.sort_values(
        [
            "q",
            "k",
            "alpha",
        ]
    ).reset_index(
        drop=True
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

    selected = (
        summary
        .sort_values(
            [
                "q",
                "k",
                "mean_AR",
                "alpha",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .groupby(
            [
                "q",
                "k",
            ],
            as_index=False,
        )
        .head(1)
        .sort_values(
            [
                "q",
                "k",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    selected[
        [
            "protocol_id",
            "q",
            "k",
            "alpha",
            "beta",
            "runs",
            "mean_AR",
            "std_AR",
            "p5_AR",
        ]
    ].to_csv(
        SELECTED_CSV,
        index=False,
    )

    return (
        summary,
        selected,
    )


def expected_run_count():

    total = 0

    for n, pair in SIZE_CONFIG.items():

        total += (
            50
            * len(INIT_SEEDS)
            * len(
                ALPHA_GRID_BY_QK[
                    pair
                ]
            )
        )

    return total


def validate_configuration():

    expected_pairs = {
        (9, 4),
        (10, 5),
        (11, 5),
        (12, 6),
        (14, 7),
        (16, 8),
    }

    actual_pairs = set(
        SIZE_CONFIG.values()
    )

    grid_pairs = set(
        ALPHA_GRID_BY_QK
    )

    if actual_pairs != expected_pairs:

        raise RuntimeError(
            "Unexpected SIZE_CONFIG pairs: "
            f"{sorted(actual_pairs)}"
        )

    if grid_pairs != expected_pairs:

        raise RuntimeError(
            "Alpha grid does not match "
            "SIZE_CONFIG pairs."
        )

    if expected_run_count() != 9500:

        raise RuntimeError(
            "Expected alpha calibration "
            "to contain exactly 9500 runs, "
            f"found {expected_run_count()}."
        )


def main():

    validate_configuration()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(1)

    df = load_calibration_table()

    completed = completed_keys()

    tasks = []

    for row in df.itertuples(
        index=False
    ):

        n = int(
            row.n
        )

        q, k = SIZE_CONFIG[n]

        for alpha in (
            ALPHA_GRID_BY_QK[
                (q, k)
            ]
        ):

            for init_seed in INIT_SEEDS:

                key = (
                    n,
                    str(
                        row.graph_id
                    ),
                    int(
                        init_seed
                    ),
                    int(
                        alpha
                    ),
                )

                if key in completed:
                    continue

                tasks.append(
                    (
                        n,
                        str(
                            row.graph_id
                        ),
                        str(
                            row.path
                        ),
                        float(
                            row.C_MQLib
                        ),
                        int(
                            init_seed
                        ),
                        int(
                            alpha
                        ),
                    )
                )

    expected = (
        expected_run_count()
    )

    print(
        "=" * 70
    )

    print(
        "STAGE 02A - "
        "ALPHA CALIBRATION BY (q,k)"
    )

    print(
        "=" * 70
    )

    print(
        "Calibration graphs: 350"
    )

    print(
        f"Initialisations: "
        f"{INIT_SEEDS}"
    )

    print(
        f"Beta fixed: {BETA}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LR}"
    )

    print(
        f"Workers: {WORKERS}"
    )

    print(
        f"Expected runs: "
        f"{expected}"
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
        "Alpha grid by (q,k):"
    )

    for pair in sorted(
        ALPHA_GRID_BY_QK
    ):

        print(
            f"  {pair}: "
            f"{ALPHA_GRID_BY_QK[pair]}"
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
                desc="alpha",
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
            f"Expected {expected} "
            f"alpha runs, "
            f"found {len(runs)}."
        )

    if runs[
        [
            "n",
            "graph_id",
            "init_seed",
            "alpha",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate alpha runs found."
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

    summary, selected = (
        build_summary()
    )

    if len(selected) != 6:

        raise RuntimeError(
            "Expected exactly 6 "
            "(q,k) alpha selections, "
            f"found {len(selected)}."
        )

    print()

    print(
        "=" * 70
    )

    print(
        "ALPHA CALIBRATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        summary[
            [
                "q",
                "k",
                "alpha",
                "runs",
                "mean_AR",
                "std_AR",
                "p5_AR",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Selected alpha by (q,k):"
    )

    print(
        selected[
            [
                "q",
                "k",
                "alpha",
                "mean_AR",
                "std_AR",
                "p5_AR",
            ]
        ].to_string(
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
        f"Selected: {SELECTED_CSV}"
    )


if __name__ == "__main__":
    main()
