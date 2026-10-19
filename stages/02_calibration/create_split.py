from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GRAPHS_FILE = ROOT / "data" / "graphs" / "graphs.csv"
OUTPUT_FILE = ROOT / "data" / "graphs" / "split.csv"

SEED = 20260909
N_CALIBRATION = 50


def main():
    graphs = pd.read_csv(GRAPHS_FILE)

    rows = []

    for n in sorted(graphs["n"].unique()):

        group = graphs[graphs["n"] == n].copy()

        if len(group) != 250:
            raise RuntimeError(
                f"Expected 250 graphs for n={n}, found {len(group)}"
            )

        # Seed específica por tamanho.
        rng = np.random.default_rng(SEED + int(n))

        ids = group["graph_id"].to_numpy().copy()
        rng.shuffle(ids)

        calibration_ids = set(ids[:N_CALIBRATION])

        for row in group.itertuples(index=False):

            split = (
                "calibration"
                if row.graph_id in calibration_ids
                else "evaluation"
            )

            rows.append(
                {
                    "n": int(n),
                    "graph_id": row.graph_id,
                    "split": split,
                }
            )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["n", "graph_id"]
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("GRAPH SPLIT")
    print("=" * 60)

    print(
        result.groupby(["n", "split"])
        .size()
        .unstack(fill_value=0)
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
