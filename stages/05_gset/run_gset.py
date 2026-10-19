from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from walsh_pce.core import (
    DTYPE,
    SIZE_CONFIG,
    decision_masks,
    load_graph,
    pce_loss,
)

from walsh_pce.protocol import (
    PROTOCOL_ID,
    validate_protocol_dataframe,
)
from walsh_pce.initialization import (
    initialize_sparse,
)
from walsh_pce.sparse import (
    base_masks_for_support,
    support_indices,
)


# ============================================================
# CONFIGURATION
# ============================================================

B = 1
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
            f"{source} has wrong "
            "(q,k) coverage."
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
                    "does not match "
                    "selected_alpha.csv."
                )

    return (
        alpha_map,
        beta_map,
    )


ALPHA_BY_QK = None
BETA_BY_QK = None
EPOCHS = 500
LR = 1e-3
WORKERS = 4

MASTER_TRAIN_SEED = 20260909

INIT_SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 110,
]

CONFIG = {
    "G14": {
        "n": 800,
        "q": 12,
        "k": 6,
        "reference": 3064,
    },
    "G23": {
        "n": 2000,
        "q": 14,
        "k": 7,
        "reference": 13344,
    },
    "G60": {
        "n": 7000,
        "q": 16,
        "k": 8,
        "reference": 14188,
    },
    "G64": {
        "n": 7000,
        "q": 16,
        "k": 8,
        "reference": 8751,
    },
}

DATA_DIR = ROOT / "data" / "gset"

OUT_DIR = ROOT / "results" / "stage_05"
ARRAY_DIR = OUT_DIR / "coefficients"

RUNS_CSV = OUT_DIR / "runs.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
SELECTED_CSV = OUT_DIR / "selected_best_raw.csv"


# ============================================================
# SEEDS
# ============================================================

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


# ============================================================
# GSET PARSER
# ============================================================

def load_gset(path):

    with path.open("r") as f:

        first = f.readline().split()

        if len(first) < 2:
            raise ValueError(
                f"Invalid Gset header: {path}"
            )

        n = int(first[0])
        m_expected = int(first[1])

        edges = []

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                raise ValueError(
                    f"Invalid edge line in {path}: {line}"
                )

            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            w = float(parts[2])

            edges.append(
                (u, v, w)
            )

    if len(edges) != m_expected:
        raise RuntimeError(
            f"{path.name}: expected {m_expected} edges, "
            f"found {len(edges)}"
        )

    u = np.asarray(
        [e[0] for e in edges],
        dtype=np.int64,
    )

    v = np.asarray(
        [e[1] for e in edges],
        dtype=np.int64,
    )

    w = np.asarray(
        [e[2] for e in edges],
        dtype=np.float64,
    )

    if u.min() < 0 or v.min() < 0:
        raise RuntimeError(
            f"Negative vertex index in {path}"
        )

    if u.max() >= n or v.max() >= n:
        raise RuntimeError(
            f"Vertex index exceeds n in {path}"
        )

    return n, u, v, w


# ============================================================
# ABSOLUTE-WEIGHT MINIMUM SPANNING FOREST
# ============================================================

def mst_absolute_weight(
    n,
    u,
    v,
    w,
):
    """
    Kruskal minimum spanning forest using |w|.

    For connected graphs this is the minimum spanning tree.
    For disconnected graphs this is the minimum spanning forest.

    Returns
    -------
    total_weight : float
        Sum of |w| over the forest edges.

    n_components : int
        Number of connected components.
    """

    parent = np.arange(
        n,
        dtype=np.int64,
    )

    rank = np.zeros(
        n,
        dtype=np.int8,
    )

    def find(x):

        while parent[x] != x:

            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x

    def union(a, b):

        ra = find(a)
        rb = find(b)

        if ra == rb:
            return False

        if rank[ra] < rank[rb]:
            ra, rb = rb, ra

        parent[rb] = ra

        if rank[ra] == rank[rb]:
            rank[ra] += 1

        return True

    order = np.argsort(
        np.abs(w),
        kind="stable",
    )

    total = 0.0
    selected = 0

    for idx in order:

        a = int(u[idx])
        b = int(v[idx])

        if union(a, b):

            total += abs(
                float(w[idx])
            )

            selected += 1

    # For a spanning forest:
    #
    # selected_edges = n - number_of_components
    #
    n_components = n - selected

    return total, n_components


# ============================================================
# DIFFERENTIABLE FWHT
# ============================================================

def fwht(x):

    y = x

    N = y.numel()

    if N == 0 or (
        N & (N - 1)
    ):
        raise ValueError(
            "FWHT requires power-of-two length."
        )

    h = 1

    while h < N:

        y = y.reshape(
            -1,
            2,
            h,
        )

        left = y[:, 0, :]
        right = y[:, 1, :]

        y = torch.cat(
            (
                left + right,
                left - right,
            ),
            dim=1,
        )

        y = y.reshape(N)

        h *= 2

    return y


# ============================================================
# WALSH CORRELATORS
# ============================================================

def walsh_correlators(
    full,
    masks,
):

    N = full.numel()

    denominator = torch.sum(
        full * full
    )

    transformed = fwht(
        full
    )

    autocorrelation = (
        fwht(
            transformed * transformed
        )
        / N
    )

    return (
        autocorrelation[masks]
        / denominator
    )


# ============================================================
# GSET LOSS
# ============================================================

def gset_loss(
    full,
    masks,
    edge_u,
    edge_v,
    edge_w,
    alpha,
    beta,
    nu,
):

    c = walsh_correlators(
        full,
        masks,
    )

    s = torch.tanh(
        alpha * c
    )

    energy = torch.sum(
        edge_w
        * s[edge_u]
        * s[edge_v]
    )

    regularizer = (
        torch.mean(
            s * s
        )
        ** 2
    )

    loss = (
        energy
        + beta
        * nu
        * regularizer
    )

    return loss, c, s


# ============================================================
# CUT
# ============================================================

def cut_value(
    spins,
    u,
    v,
    w,
):

    crossed = (
        spins[u]
        != spins[v]
    )

    return float(
        np.sum(
            w[crossed]
        )
    )


# ============================================================
# LOCAL SEARCH
# ============================================================

def build_adjacency(
    n,
    u,
    v,
    w,
):

    adj = [
        []
        for _ in range(n)
    ]

    for a, b, weight in zip(
        u,
        v,
        w,
    ):

        a = int(a)
        b = int(b)
        weight = float(weight)

        adj[a].append(
            (b, weight)
        )

        adj[b].append(
            (a, weight)
        )

    return adj


def local_search_forward_backward(
    spins,
    adjacency,
):

    z = spins.copy()

    n = len(z)

    flips_forward = 0
    flips_backward = 0

    # --------------------------------------------------------
    # Forward sweep
    # --------------------------------------------------------

    for i in range(n):

        local_field = 0.0

        for j, weight in adjacency[i]:

            local_field += (
                weight
                * z[j]
            )

        # Exact cut improvement caused by flipping z_i.
        delta_cut = (
            z[i]
            * local_field
        )

        if delta_cut > 0.0:

            z[i] *= -1
            flips_forward += 1

    # --------------------------------------------------------
    # Backward sweep
    # --------------------------------------------------------

    for i in range(
        n - 1,
        -1,
        -1,
    ):

        local_field = 0.0

        for j, weight in adjacency[i]:

            local_field += (
                weight
                * z[j]
            )

        delta_cut = (
            z[i]
            * local_field
        )

        if delta_cut > 0.0:

            z[i] *= -1
            flips_backward += 1

    return (
        z,
        flips_forward,
        flips_backward,
    )


# ============================================================
# ATOMIC NPZ
# ============================================================

def save_npz_atomic(
    path,
    **arrays,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        raise FileExistsError(
            f"Refusing to overwrite: {path}"
        )

    arrays = dict(arrays)

    # Bind every coefficient artifact to the exact
    # experimental protocol that generated it.
    arrays["protocol_id"] = np.asarray(
        PROTOCOL_ID
    )

    tmp = path.with_name(
        path.stem
        + ".tmp.npz"
    )

    np.savez_compressed(
        tmp,
        **arrays,
    )

    os.replace(
        tmp,
        path,
    )


def validate_npz_protocol(
    path,
):

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise RuntimeError(
            f"Missing coefficient artifact: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if "protocol_id" not in data.files:
            raise RuntimeError(
                f"{path}: missing protocol_id"
            )

        value = str(
            data["protocol_id"].item()
        )

    if value != PROTOCOL_ID:
        raise RuntimeError(
            f"{path}: protocol mismatch: "
            f"{value!r} != {PROTOCOL_ID!r}"
        )

    return path


# ============================================================
# PRELOAD INSTANCES
# ============================================================

def prepare_instances():

    instances = {}

    print()
    print("=" * 80)
    print("GSET INSTANCE AUDIT")
    print("=" * 80)

    for name, cfg in CONFIG.items():

        path = (
            DATA_DIR
            / f"{name}.txt"
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        n, u, v, w = load_gset(
            path
        )

        if n != cfg["n"]:

            raise RuntimeError(
                f"{name}: expected n={cfg['n']}, "
                f"found n={n}"
            )

        # ----------------------------------------------------
        # Minimum spanning forest in |w|
        # ----------------------------------------------------

        msf_abs, n_components = (
            mst_absolute_weight(
                n,
                u,
                v,
                w,
            )
        )

        weight_abs_sum = float(
            np.abs(w).sum()
        )

        # Unified positive/signed convention:
        #
        # nu = 1/2 sum_e |w_e|
        #    + 1/4 w_abs(F_min)
        #
        nu = (
            0.5
            * weight_abs_sum
            + 0.25
            * msf_abs
        )

        positive = int(
            np.sum(w > 0)
        )

        negative = int(
            np.sum(w < 0)
        )

        zero = int(
            np.sum(w == 0)
        )

        adjacency = build_adjacency(
            n,
            u,
            v,
            w,
        )

        edge_u_t = torch.tensor(
            u,
            dtype=torch.long,
        )

        edge_v_t = torch.tensor(
            v,
            dtype=torch.long,
        )

        edge_w_t = torch.tensor(
            w,
            dtype=DTYPE,
        )

        instances[name] = {
            "path": path,
            "n": n,
            "u": u,
            "v": v,
            "w": w,
            "edge_u_t": edge_u_t,
            "edge_v_t": edge_v_t,
            "edge_w_t": edge_w_t,
            "adjacency": adjacency,
            "msf_abs": msf_abs,
            "weight_abs_sum": (
                weight_abs_sum
            ),
            "n_components": (
                n_components
            ),
            "nu": nu,
        }

        print()
        print(
            f"{name}: "
            f"n={n}, "
            f"edges={len(w)}, "
            f"+={positive}, "
            f"-={negative}, "
            f"0={zero}"
        )

        print(
            f"  sum|w| = "
            f"{weight_abs_sum:.6f}"
        )

        print(
            f"  MSF_abs = "
            f"{msf_abs:.6f}"
        )

        print(
            f"  components = "
            f"{n_components}"
        )

        print(
            f"  nu = "
            f"{nu:.6f}"
        )

    # --------------------------------------------------------
    # Explicit G60 audit
    # --------------------------------------------------------

    if instances["G60"]["n_components"] != 45:

        raise RuntimeError(
            "G60 component audit failed: "
            f"expected 45 components, found "
            f"{instances['G60']['n_components']}"
        )

    if abs(
        instances["G60"]["msf_abs"]
        - 6955.0
    ) > 1e-12:

        raise RuntimeError(
            "G60 MSF audit failed: "
            f"expected 6955, found "
            f"{instances['G60']['msf_abs']}"
        )

    print()
    print(
        "G60 original structure: "
        "45 components, MSF_abs=6955: OK"
    )

    return instances


# ============================================================
# VALIDATE NEW LOSS AGAINST FROZEN CORE
# ============================================================

def validate_positive_instance(
    instances,
):
    """
    On G14 all weights are positive and the graph is connected.

    Therefore the Gset loss must reproduce the frozen
    Stage-03/04 PCE implementation exactly.
    """

    name = "G14"

    cfg = CONFIG[name]
    data = instances[name]

    q = cfg["q"]
    k = cfg["k"]
    alpha = ALPHA_BY_QK[(q, k)]
    beta = BETA_BY_QK[(q, k)]
    n = cfg["n"]

    graph = load_graph(
        str(data["path"])
    )

    masks = decision_masks(
        q=q,
        k=k,
        m=n,
    )

    active_idx = support_indices(
        n=n,
        q=q,
        k=k,
        B=B,
    )

    seed = stable_seed(
        MASTER_TRAIN_SEED,
        n,
        name,
        INIT_SEEDS[0],
    )

    base_masks = base_masks_for_support(
        n=n,
        q=q,
        k=k,
        B=B,
    )

    a = initialize_sparse(
        active_idx=active_idx,
        base_masks=base_masks,
        seed=seed,
        n_coeffs=2**q,
        dtype=DTYPE,
    )

    full = torch.zeros(
        2**q,
        dtype=DTYPE,
    ).scatter(
        0,
        active_idx,
        a,
    )

    with torch.no_grad():

        core_loss, core_c = pce_loss(
            a=full,
            graph=graph,
            masks=masks,
            alpha=alpha,
            beta=beta,
        )

        new_loss, new_c, _ = (
            gset_loss(
                full=full,
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
                core_c
                - new_c
            )
        ).item()
    )

    loss_error = abs(
        float(
            core_loss.item()
        )
        - float(
            new_loss.item()
        )
    )

    print()
    print("=" * 80)
    print("POSITIVE-WEIGHT LOSS VALIDATION")
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
            "Walsh correlator validation failed."
        )

    if loss_error > 1e-8:

        raise RuntimeError(
            "Positive-weight loss validation failed."
        )

    print(
        "New loss == frozen positive-weight loss: OK"
    )


# ============================================================
# TRAIN ONE
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
    alpha = ALPHA_BY_QK[(q, k)]
    beta = BETA_BY_QK[(q, k)]
    reference = cfg["reference"]

    masks = decision_masks(
        q=q,
        k=k,
        m=n,
    )

    active_idx = support_indices(
        n=n,
        q=q,
        k=k,
        B=B,
    )

    actual_seed = stable_seed(
        MASTER_TRAIN_SEED,
        n,
        name,
        init_seed,
    )

    base_masks = base_masks_for_support(
        n=n,
        q=q,
        k=k,
        B=B,
    )

    a = initialize_sparse(
        active_idx=active_idx,
        base_masks=base_masks,
        seed=actual_seed,
        n_coeffs=2**q,
        dtype=DTYPE,
    )

    a.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [a],
        lr=LR,
    )

    N = 2**q

    start = time.perf_counter()

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    for _ in range(EPOCHS):

        optimizer.zero_grad(
            set_to_none=True
        )

        full = torch.zeros(
            N,
            dtype=DTYPE,
        ).scatter(
            0,
            active_idx,
            a,
        )

        loss, _, _ = gset_loss(
            full=full,
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

        full = torch.zeros(
            N,
            dtype=DTYPE,
        ).scatter(
            0,
            active_idx,
            a,
        )

        final_loss, c, s = gset_loss(
            full=full,
            masks=masks,
            edge_u=data["edge_u_t"],
            edge_v=data["edge_v_t"],
            edge_w=data["edge_w_t"],
            alpha=alpha,
            beta=beta,
            nu=data["nu"],
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

    raw_cut = cut_value(
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
    # LOCAL SEARCH
    # EXACTLY:
    #   0 -> n-1
    #   n-1 -> 0
    # ========================================================

    (
        local_spins,
        flips_forward,
        flips_backward,
    ) = local_search_forward_backward(
        raw_spins,
        data["adjacency"],
    )

    local_cut = cut_value(
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
    # SAVE COEFFICIENTS FOR STAGE 06
    # ========================================================

    instance_dir = (
        ARRAY_DIR
        / name
    )

    npz_path = (
        instance_dir
        / f"{name}_seed{init_seed:03d}.npz"
    )

    save_npz_atomic(
        npz_path,

        active_indices=(
            active_idx
            .cpu()
            .numpy()
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
        "B": B,
        "active_size": (
            len(active_idx)
        ),
        "full_size": N,
        "active_fraction": (
            len(active_idx)
            / N
        ),
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
    "protocol_id",
    "instance",
    "n",
    "edges",
    "components",
    "q",
    "k",
    "B",
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

    row = dict(row)
    row["protocol_id"] = PROTOCOL_ID

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


def completed_keys():

    if not RUNS_CSV.exists():
        return set()

    df = pd.read_csv(
        RUNS_CSV
    )

    validate_protocol_dataframe(
        df,
        source=str(RUNS_CSV),
    )

    required = {
        "instance",
        "q",
        "k",
        "B",
        "alpha",
        "beta",
        "epochs",
        "lr",
        "init_seed",
        "npz",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{RUNS_CSV}: missing columns "
            f"{sorted(missing)}"
        )

    if (
        ALPHA_BY_QK is None
        or BETA_BY_QK is None
    ):
        raise RuntimeError(
            "Hyperparameters must be loaded "
            "before resume validation."
        )

    duplicate = df.duplicated(
        subset=[
            "instance",
            "init_seed",
        ],
        keep=False,
    )

    if duplicate.any():
        bad = (
            df.loc[
                duplicate,
                [
                    "instance",
                    "init_seed",
                ],
            ]
            .drop_duplicates()
            .to_dict("records")
        )

        raise RuntimeError(
            f"Duplicate Stage 05 run keys: {bad}"
        )

    for row in df.itertuples(
        index=False
    ):

        q = int(row.q)
        k = int(row.k)
        qk = (q, k)

        if qk not in ALPHA_BY_QK:
            raise RuntimeError(
                f"Missing current alpha for {qk}"
            )

        if qk not in BETA_BY_QK:
            raise RuntimeError(
                f"Missing current beta for {qk}"
            )

        expected_alpha = float(
            ALPHA_BY_QK[qk]
        )

        expected_beta = float(
            BETA_BY_QK[qk]
        )

        if int(row.B) != int(B):
            raise RuntimeError(
                f"{row.instance}, seed={row.init_seed}: "
                f"B={row.B}, expected {B}"
            )

        if int(row.epochs) != int(EPOCHS):
            raise RuntimeError(
                f"{row.instance}, seed={row.init_seed}: "
                f"epochs={row.epochs}, expected {EPOCHS}"
            )

        if not np.isclose(
            float(row.lr),
            float(LR),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"{row.instance}, seed={row.init_seed}: "
                f"lr={row.lr}, expected {LR}"
            )

        if not np.isclose(
            float(row.alpha),
            expected_alpha,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{row.instance}, seed={row.init_seed}: "
                f"alpha={row.alpha}, "
                f"expected {expected_alpha}"
            )

        if not np.isclose(
            float(row.beta),
            expected_beta,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{row.instance}, seed={row.init_seed}: "
                f"beta={row.beta}, "
                f"expected {expected_beta}"
            )

        validate_npz_protocol(
            row.npz
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

    validate_protocol_dataframe(
        runs,
        source=str(RUNS_CSV),
    )

    summary = (
        runs
        .groupby(
            "instance",
            as_index=False,
        )
        .agg(
            n=(
                "n",
                "first",
            ),
            edges=(
                "edges",
                "first",
            ),
            components=(
                "components",
                "first",
            ),
            q=(
                "q",
                "first",
            ),
            k=(
                "k",
                "first",
            ),
            B=(
                "B",
                "first",
            ),
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

    summary.insert(
        0,
        "protocol_id",
        PROTOCOL_ID,
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # ========================================================
    # BEST RAW MODEL FOR STAGE 06
    # ========================================================

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
                "protocol_id",
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

    return (
        summary,
        selected,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global ALPHA_BY_QK
    global BETA_BY_QK

    (
        ALPHA_BY_QK,
        BETA_BY_QK,
    ) = load_hyperparameters()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARRAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(1)

    # ========================================================
    # INSTANCE AUDIT
    # ========================================================

    instances = prepare_instances()

    # ========================================================
    # LOSS VALIDATION
    # ========================================================

    validate_positive_instance(
        instances
    )

    # ========================================================
    # RESUME
    # ========================================================

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
    print("STAGE 05 - GSET")
    print("=" * 80)

    print(
        f"B: {B}"
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

    # ========================================================
    # RUN
    # ========================================================

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
                desc="Gset",
            ):

                name, seed = futures[
                    future
                ]

                try:

                    result = (
                        future.result()
                    )

                except Exception:

                    print()
                    print(
                        f"FAILED: "
                        f"{name}, "
                        f"seed={seed}"
                    )

                    raise

                append_row(
                    result
                )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

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
            "Duplicate Stage-05 runs found."
        )

    if not np.allclose(
        runs["final_norm"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Coefficient normalization audit failed."
        )

    if (
        runs["local_cut"]
        + 1e-12
        < runs["raw_cut"]
    ).any():

        raise RuntimeError(
            "Local-search monotonicity audit failed."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary, selected = (
        build_summary()
    )

    print()
    print("=" * 100)
    print("STAGE 05 FINISHED")
    print("=" * 100)

    print()

    print(
        summary[
            [
                "instance",
                "n",
                "edges",
                "components",
                "q",
                "k",
                "B",
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
    print("=" * 100)
    print(
        "SELECTED BEST RAW MODELS "
        "FOR STAGE 06"
    )
    print("=" * 100)
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
        f"Selected: {SELECTED_CSV}"
    )

    print(
        f"Coefficients: {ARRAY_DIR}"
    )


if __name__ == "__main__":
    main()
