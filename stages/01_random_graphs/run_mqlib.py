from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import csv

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

MQLIB_ROOT = Path(os.environ.get("MQLIB_ROOT", ROOT / ".external" / "MQLib")).expanduser().resolve()
MQLIB = MQLIB_ROOT / "bin" / "MQLib"

GRAPH_CSV = ROOT / "data" / "graphs" / "graphs.csv"

OUT_DIR = ROOT / "results" / "stage_01"
RUNS_CSV = OUT_DIR / "mqlib_runs.csv"
SUMMARY_CSV = OUT_DIR / "mqlib_summary.csv"

SEEDS = list(range(1, 11))
RUNTIME = 5
WORKERS = 8


def run_one(n, graph_id, graph_path, seed):
    graph_path = ROOT / "data" / "graphs" / graph_path

    cmd = [
        str(MQLIB),
        "-hh",
        "-fM",
        str(graph_path),
        "-r",
        str(RUNTIME),
        "-s",
        str(seed),
    ]

    result = subprocess.run(
        cmd,
        cwd=MQLIB_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed: n={n}, graph={graph_id}, seed={seed}\n"
            f"{result.stderr}"
        )

    line = result.stdout.strip().splitlines()[-1]

    fields = next(csv.reader([line]))

    if len(fields) < 5:
        raise RuntimeError(f"Unexpected MQLib output:\n{line}")

    heuristic = fields[1]
    cut = float(fields[3])
    runtime_actual = float(fields[4])

    if heuristic == "HH_":
        raise RuntimeError(
            f"Hyper-heuristic failed for {graph_path}, seed={seed}"
        )

    return {
        "n": int(n),
        "graph_id": graph_id,
        "seed": int(seed),
        "heuristic": heuristic,
        "cut": cut,
        "runtime_requested": RUNTIME,
        "runtime_actual": runtime_actual,
        "graph_path": str(graph_path.relative_to(ROOT)),
    }


def load_completed():
    if not RUNS_CSV.exists():
        return set()

    df = pd.read_csv(RUNS_CSV)

    return set(
        zip(
            df["n"].astype(int),
            df["graph_id"].astype(str),
            df["seed"].astype(int),
        )
    )


def append_result(row):
    exists = RUNS_CSV.exists()

    with RUNS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "graph_id",
                "seed",
                "heuristic",
                "cut",
                "runtime_requested",
                "runtime_actual",
                "graph_path",
            ],
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def build_summary():
    df = pd.read_csv(RUNS_CSV)

    idx = (
        df.groupby(["n", "graph_id"])["cut"]
        .idxmax()
    )

    best = df.loc[idx].copy()

    best = best.rename(
        columns={
            "seed": "best_seed",
            "heuristic": "best_heuristic",
            "cut": "C_MQLib",
            "runtime_actual": "best_run_runtime",
        }
    )

    best = best[
        [
            "n",
            "graph_id",
            "C_MQLib",
            "best_seed",
            "best_heuristic",
            "best_run_runtime",
            "graph_path",
        ]
    ]

    best = best.sort_values(
        ["n", "graph_id"]
    )

    best.to_csv(SUMMARY_CSV, index=False)

    return best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graphs = pd.read_csv(GRAPH_CSV)

    completed = load_completed()

    tasks = []

    for row in graphs.itertuples(index=False):
        for seed in SEEDS:
            key = (int(row.n), str(row.graph_id), seed)

            if key not in completed:
                tasks.append(
                    (
                        int(row.n),
                        str(row.graph_id),
                        str(row.path),
                        seed,
                    )
                )

    total_expected = len(graphs) * len(SEEDS)

    print("=" * 70)
    print("MQLib hyper-heuristic")
    print("=" * 70)
    print(f"Graphs: {len(graphs)}")
    print(f"Seeds: {SEEDS}")
    print(f"Runtime: {RUNTIME} s")
    print(f"Workers: {WORKERS}")
    print(f"Total expected runs: {total_expected}")
    print(f"Already completed: {len(completed)}")
    print(f"Runs remaining: {len(tasks)}")
    print()

    if tasks:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [
                executor.submit(run_one, *task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="MQLib",
            ):
                row = future.result()
                append_result(row)

    summary = build_summary()

    print()
    print("=" * 70)
    print("MQLib complete")
    print("=" * 70)
    print(f"Runs: {len(pd.read_csv(RUNS_CSV))}")
    print(f"Graphs summarized: {len(summary)}")
    print(f"Runs file: {RUNS_CSV}")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
