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


INIT_SEEDS = [
    11,
    22,
    33,
    44,
    55,
]

BETA_GRID = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

EPOCHS = 150
LR = 1e-3
WORKERS = 4

MASTER_TRAIN_SEED = 20260909


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

ALPHA_CSV = (
    OUT_DIR
    / "selected_alpha.csv"
)

RUNS_CSV = (
    OUT_DIR
    / "beta_runs.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "beta_summary.csv"
)

SELECTED_CSV = (
    OUT_DIR
    / "selected_beta.csv"
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


def validate_selected_alpha_dataframe(
    df,
):

    validate_protocol_dataframe(
        df,
        "selected_alpha.csv",
    )

    required = {
        "q",
        "k",
        "alpha",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "selected_alpha.csv missing columns: "
            f"{sorted(missing)}"
        )

    if df[
        [
            "q",
            "k",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate (q,k) rows in "
            "selected_alpha.csv."
        )

    expected_pairs = set(
        SIZE_CONFIG.values()
    )

    actual_pairs = {
        (
            int(row.q),
            int(row.k),
        )
        for row
        in df.itertuples(
            index=False
        )
    }

    if actual_pairs != expected_pairs:

        raise RuntimeError(
            "selected_alpha.csv has wrong "
            "(q,k) coverage. "
            f"Found {sorted(actual_pairs)}, "
            f"expected {sorted(expected_pairs)}."
        )


def load_alpha_map():

    df = pd.read_csv(
        ALPHA_CSV
    )

    validate_selected_alpha_dataframe(
        df
    )

    return {
        (
            int(row.q),
            int(row.k),
        ):
        float(row.alpha)

        for row
        in df.itertuples(
            index=False
        )
    }


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
    beta,
    alpha_map,
):

    q, k = SIZE_CONFIG[n]

    alpha = alpha_map[
        (q, k)
    ]

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
        beta=float(beta),
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
        "alpha": float(alpha),
        "beta": float(beta),
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
        "beta",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Existing beta_runs.csv has "
            "incompatible schema. Missing: "
            f"{sorted(missing)}"
        )

    return set(
        zip(
            df["n"].astype(int),
            df["graph_id"].astype(str),
            df["init_seed"].astype(int),
            df["alpha"].astype(float),
            df["beta"].astype(float),
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
            p5_AR=(
                "AR",
                lambda x:
                np.quantile(
                    x,
                    0.05,
                ),
            ),
        )
    )

    summary = summary.sort_values(
        [
            "q",
            "k",
            "beta",
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
                "beta",
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

    return (
        50
        * len(SIZE_CONFIG)
        * len(INIT_SEEDS)
        * len(BETA_GRID)
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(1)

    alpha_map = (
        load_alpha_map()
    )

    df = (
        load_calibration_table()
    )

    completed = (
        completed_keys()
    )

    tasks = []

    for row in df.itertuples(
        index=False
    ):

        n = int(
            row.n
        )

        q, k = SIZE_CONFIG[n]

        alpha = float(
            alpha_map[
                (q, k)
            ]
        )

        for beta in BETA_GRID:

            for init_seed in INIT_SEEDS:

                key = (
                    n,
                    str(
                        row.graph_id
                    ),
                    int(
                        init_seed
                    ),
                    alpha,
                    float(
                        beta
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
                        float(
                            beta
                        ),
                        alpha_map,
                    )
                )

    expected = (
        expected_run_count()
    )

    print(
        "=" * 70
    )

    print(
        "STAGE 02B - "
        "BETA CALIBRATION BY (q,k)"
    )

    print(
        "=" * 70
    )

    print(
        "Selected alpha:"
    )

    for pair in sorted(
        alpha_map
    ):

        print(
            f"  (q,k)={pair}: "
            f"alpha="
            f"{alpha_map[pair]:g}"
        )

    print(
        f"Beta grid: "
        f"{BETA_GRID}"
    )

    print(
        f"Calibration graphs: "
        f"{len(df)}"
    )

    print(
        f"Seeds: "
        f"{INIT_SEEDS}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Workers: "
        f"{WORKERS}"
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
                desc="beta",
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
            f"Expected {expected} beta runs, "
            f"found {len(runs)}."
        )

    if runs[
        [
            "n",
            "graph_id",
            "init_seed",
            "alpha",
            "beta",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate beta runs found."
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
            "(q,k) beta selections, "
            f"found {len(selected)}."
        )

    print()

    print(
        "=" * 70
    )

    print(
        "BETA CALIBRATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Selected beta by (q,k):"
    )

    print(
        selected[
            [
                "q",
                "k",
                "alpha",
                "beta",
                "mean_AR",
                "std_AR",
                "p5_AR",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
