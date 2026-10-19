from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from walsh_pce.core import SIZE_CONFIG
from walsh_pce.sparse import support_indices
from walsh_pce.protocol import PROTOCOL_ID


OUT_DIR = ROOT / "configs"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FILE = (
    OUT_DIR
    / "support_grid.csv"
)


CANDIDATE_B = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]


def find_saturation(
    n,
    q,
    k,
):

    N = 2**q

    for B in range(
        1,
        N + 1,
    ):

        active = support_indices(
            n=n,
            q=q,
            k=k,
            B=B,
        )

        if len(active) == N:
            return B

    raise RuntimeError(
        f"Support never saturated: "
        f"n={n}, q={q}, k={k}"
    )


def build_grid():

    rows = []

    for n, (q, k) in (
        SIZE_CONFIG.items()
    ):

        N = 2**q

        B_sat = find_saturation(
            n=n,
            q=q,
            k=k,
        )

        grid = sorted(
            {
                B
                for B
                in (
                    CANDIDATE_B
                    + [B_sat]
                )
                if B <= B_sat
            }
        )

        previous = set()

        for B in grid:

            active = support_indices(
                n=n,
                q=q,
                k=k,
                B=B,
            )

            active_set = set(
                int(x)
                for x in active.tolist()
            )

            if previous:

                if not (
                    previous
                    <= active_set
                ):

                    raise RuntimeError(
                        "Support nesting failed: "
                        f"n={n}, B={B}"
                    )

            previous = active_set

            active_size = len(
                active
            )

            rows.append(
                {
                    "protocol_id":
                        PROTOCOL_ID,
                    "n": int(n),
                    "q": int(q),
                    "k": int(k),
                    "B": int(B),
                    "active_size":
                        int(active_size),
                    "full_size":
                        int(N),
                    "active_fraction":
                        float(
                            active_size / N
                        ),
                    "B_sat":
                        int(B_sat),
                }
            )

    result = pd.DataFrame(
        rows
    )

    return (
        result
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


def main():

    result = build_grid()

    result.to_csv(
        OUT_FILE,
        index=False,
    )

    print(
        result.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
