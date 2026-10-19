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
)
from walsh_pce.sparse import (
    train_sparse,
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
    11,
    22,
    33,
    44,
    55,
]

EPOCHS = 500
LR = 1e-3

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

COMPLETE_CSV = (
    ROOT
    / "results"
    / "stage_03"
    / "runs.csv"
)

GRID_CSV = (
    ROOT
    / "configs"
    / "support_grid.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "stage_04"
)

RUNS_CSV = (
    OUT_DIR
    / "runs.csv"
)

SUMMARY_CSV = (
    OUT_DIR
    / "summary.csv"
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
        for row
        in df.itertuples(
            index=False
        )
    }

    if actual_pairs != expected_pairs:

        raise RuntimeError(
            f"{source} has wrong (q,k) coverage."
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

        for row in beta_df.itertuples(
            index=False
        ):

            pair = (
                int(row.q),
                int(row.k),
            )

            if not np.isclose(
                float(row.alpha),
                alpha_map[pair],
                atol=0.0,
                rtol=0.0,
            ):

                raise RuntimeError(
                    "selected_beta.csv alpha "
                    "does not match selected_alpha.csv."
                )

    return (
        alpha_map,
        beta_map,
    )


def validate_grid(
    grid,
):

    validate_protocol_dataframe(
        grid,
        str(GRID_CSV),
    )

    required = {
        "n",
        "q",
        "k",
        "B",
        "active_size",
        "full_size",
        "active_fraction",
        "B_sat",
    }

    missing = (
        required
        - set(grid.columns)
    )

    if missing:

        raise RuntimeError(
            "Support grid missing columns: "
            f"{sorted(missing)}"
        )

    if grid[
        ["n", "B"]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate (n,B) in support grid."
        )

    for n, (q, k) in (
        SIZE_CONFIG.items()
    ):

        x = grid[
            grid["n"] == n
        ]

        if len(x) == 0:

            raise RuntimeError(
                f"Missing support grid for n={n}."
            )

        if not (
            x["q"].astype(int)
            == q
        ).all():

            raise RuntimeError(
                f"Wrong q in support grid for n={n}."
            )

        if not (
            x["k"].astype(int)
            == k
        ).all():

            raise RuntimeError(
                f"Wrong k in support grid for n={n}."
            )

        if (
            x["B_sat"].nunique()
            != 1
        ):

            raise RuntimeError(
                f"Multiple B_sat values for n={n}."
            )

        Bsat = int(
            x["B_sat"].iloc[0]
        )

        sat = x[
            x["B"] == Bsat
        ]

        if len(sat) != 1:

            raise RuntimeError(
                f"Missing saturation row for n={n}."
            )

        if int(
            sat["active_size"].iloc[0]
        ) != 2**q:

            raise RuntimeError(
                f"Saturation size wrong for n={n}."
            )


def validate_complete_runs(
    complete,
    alpha_map,
    beta_map,
):

    validate_protocol_dataframe(
        complete,
        str(COMPLETE_CSV),
    )

    required = {
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
        "AR",
        "final_norm",
    }

    missing = (
        required
        - set(complete.columns)
    )

    if missing:

        raise RuntimeError(
            "Stage-03 runs missing columns: "
            f"{sorted(missing)}"
        )

    if complete[
        [
            "n",
            "graph_id",
            "init_seed",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Stage-03 runs."
        )

    for row in complete.itertuples(
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
                f"Stage-03 q,k mismatch for n={n}."
            )

        if not np.isclose(
            float(row.alpha),
            alpha_map[(q, k)],
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Stage-03 alpha mismatch."
            )

        if not np.isclose(
            float(row.beta),
            beta_map[(q, k)],
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Stage-03 beta mismatch."
            )

        if int(
            row.epochs
        ) != EPOCHS:

            raise RuntimeError(
                "Stage-03 epochs mismatch."
            )

        if not np.isclose(
            float(row.lr),
            LR,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Stage-03 learning-rate mismatch."
            )

    if not np.allclose(
        complete["final_norm"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Stage-03 normalization audit failed."
        )


def load_data(
    alpha_map,
    beta_map,
):

    graphs = pd.read_csv(
        GRAPHS_CSV
    )

    split = pd.read_csv(
        SPLIT_CSV
    )

    refs = pd.read_csv(
        MQLIB_CSV
    )

    complete = pd.read_csv(
        COMPLETE_CSV
    )

    grid = pd.read_csv(
        GRID_CSV
    )

    validate_grid(
        grid
    )

    validate_complete_runs(
        complete,
        alpha_map,
        beta_map,
    )

    graphs = graphs.merge(
        split,
        on=[
            "n",
            "graph_id",
        ],
        validate="one_to_one",
    )

    graphs = graphs[
        graphs["split"]
        == "evaluation"
    ].copy()

    graphs = graphs.merge(
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
        graphs.groupby("n")
        .size()
        .to_dict()
    )

    for n in SIZE_CONFIG:

        if counts.get(n) != 200:

            raise RuntimeError(
                f"Expected 200 evaluation "
                f"graphs for n={n}."
            )

    complete = complete[
        complete["init_seed"].isin(
            INIT_SEEDS
        )
    ].copy()

    expected_complete = (
        200
        * len(SIZE_CONFIG)
        * len(INIT_SEEDS)
    )

    if len(complete) != expected_complete:

        raise RuntimeError(
            f"Expected {expected_complete} "
            "Stage-03 paired runs, "
            f"found {len(complete)}."
        )

    complete_map = {
        (
            int(row.n),
            str(row.graph_id),
            int(row.init_seed),
        ): (
            float(row.cut),
            float(row.AR),
            int(row.actual_seed),
        )

        for row
        in complete.itertuples(
            index=False
        )
    }

    grid_sparse = grid[
        grid["B"]
        < grid["B_sat"]
    ].copy()

    return (
        graphs,
        complete,
        complete_map,
        grid,
        grid_sparse,
    )


FIELDS = [
    "protocol_id",
    "n",
    "q",
    "k",
    "graph_id",
    "init_seed",
    "actual_seed",
    "B",
    "active_size",
    "full_size",
    "active_fraction",
    "alpha",
    "beta",
    "epochs",
    "lr",
    "cut",
    "C_MQLib",
    "AR",
    "complete_cut",
    "complete_AR",
    "recovery",
    "zero_cut",
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

    missing = (
        set(FIELDS)
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Existing Stage-04 runs have "
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

        if not np.isclose(
            float(row.alpha),
            alpha_map[(q, k)],
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Existing Stage-04 alpha mismatch."
            )

        if not np.isclose(
            float(row.beta),
            beta_map[(q, k)],
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Existing Stage-04 beta mismatch."
            )

        if int(
            row.epochs
        ) != EPOCHS:

            raise RuntimeError(
                "Existing Stage-04 epoch mismatch."
            )

        if not np.isclose(
            float(row.lr),
            LR,
            atol=0.0,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Existing Stage-04 lr mismatch."
            )

    return set(
        zip(
            df["n"].astype(int),
            df["graph_id"].astype(str),
            df["init_seed"].astype(int),
            df["B"].astype(int),
        )
    )


def append_row(
    row,
):

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


def run_one(
    n,
    graph_id,
    relative_path,
    c_ref,
    init_seed,
    B,
    active_size,
    full_size,
    active_fraction,
    complete_cut,
    complete_AR,
    complete_actual_seed,
    alpha_map,
    beta_map,
):

    q, k = SIZE_CONFIG[n]

    alpha = float(
        alpha_map[(q, k)]
    )

    beta = float(
        beta_map[(q, k)]
    )

    actual_seed = stable_seed(
        MASTER_TRAIN_SEED,
        n,
        graph_id,
        init_seed,
    )

    if (
        actual_seed
        != complete_actual_seed
    ):

        raise RuntimeError(
            "Sparse/full actual_seed mismatch."
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

    result = train_sparse(
        graph=graph,
        q=q,
        k=k,
        B=B,
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

    if (
        result["active_size"]
        != active_size
    ):

        raise RuntimeError(
            "Support-size mismatch: "
            f"n={n}, B={B}, "
            f"{result['active_size']} "
            f"!= {active_size}"
        )

    cut = float(
        result["cut"]
    )

    ar = (
        cut
        / float(c_ref)
    )

    recovery = (
        cut
        / complete_cut
        if complete_cut > 0
        else float("nan")
    )

    return {
        "protocol_id": PROTOCOL_ID,
        "n": int(n),
        "q": int(q),
        "k": int(k),
        "graph_id": str(graph_id),
        "init_seed": int(init_seed),
        "actual_seed": int(actual_seed),
        "B": int(B),
        "active_size": int(active_size),
        "full_size": int(full_size),
        "active_fraction":
            float(active_fraction),
        "alpha": alpha,
        "beta": beta,
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "cut": cut,
        "C_MQLib": float(c_ref),
        "AR": float(ar),
        "complete_cut":
            float(complete_cut),
        "complete_AR":
            float(complete_AR),
        "recovery":
            float(recovery),
        "zero_cut":
            int(cut == 0),
        "final_loss":
            float(result["loss"]),
        "final_norm":
            float(result["norm"]),
        "runtime_sec":
            float(elapsed),
    }


def build_summary(
    complete,
    grid,
):

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
                "n",
                "q",
                "k",
                "B",
                "active_size",
                "full_size",
                "active_fraction",
            ],
            as_index=False,
        )
        .agg(
            runs=(
                "recovery",
                "size",
            ),
            mean_recovery=(
                "recovery",
                "mean",
            ),
            median_recovery=(
                "recovery",
                "median",
            ),
            std_recovery=(
                "recovery",
                "std",
            ),
            mean_AR=(
                "AR",
                "mean",
            ),
            std_AR=(
                "AR",
                "std",
            ),
            zero_cuts=(
                "zero_cut",
                "sum",
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
                "n",
                "B",
            ]
        )["recovery"]
        .quantile(0.05)
        .reset_index(
            name="p5_recovery"
        )
    )

    summary = summary.merge(
        p5,
        on=[
            "n",
            "B",
        ],
        validate="one_to_one",
    )

    sat_rows = []

    for n in sorted(
        grid["n"].unique()
    ):

        n = int(n)

        q, k = SIZE_CONFIG[n]

        gr = grid[
            grid["n"] == n
        ]

        B_sat = int(
            gr["B_sat"].iloc[0]
        )

        c = complete[
            complete["n"] == n
        ]

        sat_rows.append(
            {
                "n": n,
                "q": q,
                "k": k,
                "B": B_sat,
                "active_size": 2**q,
                "full_size": 2**q,
                "active_fraction": 1.0,
                "runs": len(c),
                "mean_recovery": 1.0,
                "median_recovery": 1.0,
                "std_recovery": 0.0,
                "mean_AR":
                    c["AR"].mean(),
                "std_AR":
                    c["AR"].std(),
                "zero_cuts":
                    int(
                        (
                            c["cut"]
                            == 0
                        ).sum()
                    ),
                "mean_runtime_sec":
                    float("nan"),
                "p5_recovery": 1.0,
            }
        )

    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                sat_rows
            ),
        ],
        ignore_index=True,
    )

    summary = (
        summary
        .sort_values(
            [
                "n",
                "B",
            ]
        )
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


def expected_run_count(
    grid_sparse,
):

    return (
        len(grid_sparse)
        * 200
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

    (
        graphs,
        complete,
        complete_map,
        grid,
        grid_sparse,
    ) = load_data(
        alpha_map,
        beta_map,
    )

    completed = completed_keys(
        alpha_map,
        beta_map,
    )

    tasks = []

    for graph in graphs.itertuples(
        index=False
    ):

        n = int(
            graph.n
        )

        grid_n = grid_sparse[
            grid_sparse["n"]
            == n
        ]

        for support in (
            grid_n.itertuples(
                index=False
            )
        ):

            B = int(
                support.B
            )

            for init_seed in (
                INIT_SEEDS
            ):

                key = (
                    n,
                    str(
                        graph.graph_id
                    ),
                    init_seed,
                    B,
                )

                if key in completed:
                    continue

                complete_key = (
                    n,
                    str(
                        graph.graph_id
                    ),
                    init_seed,
                )

                if (
                    complete_key
                    not in complete_map
                ):

                    raise RuntimeError(
                        "Missing paired "
                        "Stage-03 run: "
                        f"{complete_key}"
                    )

                (
                    complete_cut,
                    complete_AR,
                    complete_actual_seed,
                ) = complete_map[
                    complete_key
                ]

                tasks.append(
                    (
                        n,
                        str(
                            graph.graph_id
                        ),
                        str(
                            graph.path
                        ),
                        float(
                            graph.C_MQLib
                        ),
                        init_seed,
                        B,
                        int(
                            support.active_size
                        ),
                        int(
                            support.full_size
                        ),
                        float(
                            support.active_fraction
                        ),
                        complete_cut,
                        complete_AR,
                        complete_actual_seed,
                        alpha_map,
                        beta_map,
                    )
                )

    expected = expected_run_count(
        grid_sparse
    )

    print("=" * 70)
    print("STAGE 04 - SUPPORT SWEEP")
    print("=" * 70)

    print(
        f"Evaluation graphs: "
        f"{len(graphs)}"
    )

    print(
        f"Seeds: {INIT_SEEDS}"
    )

    print(
        f"Sparse B configurations: "
        f"{len(grid_sparse)}"
    )

    print(
        f"Expected new runs: "
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
                desc="support sweep",
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
            f"Expected {expected} sparse runs, "
            f"found {len(runs)}."
        )

    if runs[
        [
            "n",
            "graph_id",
            "init_seed",
            "B",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Stage-04 runs."
        )

    if not np.allclose(
        runs["final_norm"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Stage-04 normalization "
            "audit failed."
        )

    summary = build_summary(
        complete,
        grid,
    )

    print()

    print(
        summary[
            [
                "n",
                "B",
                "active_size",
                "active_fraction",
                "runs",
                "mean_recovery",
                "median_recovery",
                "p5_recovery",
                "mean_AR",
                "zero_cuts",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
