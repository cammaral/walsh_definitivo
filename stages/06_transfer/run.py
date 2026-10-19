#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from walsh_pce.core import decision_masks
from walsh_pce.protocol import PROTOCOL_ID

from quantum_sub_extended import (
    analytical_loader_outputs,
    correlators_from_probabilities,
    dense_coefficients,
    postselected_histogram,
    resource_counts,
    run_loader_samples,
    run_loader_statevector,
    run_original_loader_statevector,
    run_pauli_trajectory_samples,
    state_fidelity,
    target_walsh_outputs,
    total_variation,
)


# ============================================================
# INPUT / OUTPUT
# ============================================================

STAGE5_SELECTED = (
    ROOT
    / "results"
    / "stage_05"
    / "selected_best_raw.csv"
)

STAGE5_SUMMARY = (
    ROOT
    / "results"
    / "stage_05"
    / "summary.csv"
)

DATA_DIR = (
    ROOT
    / "data"
    / "gset"
)

OUT_DIR = (
    ROOT
    / "results"
    / "stage_06"
)


# ============================================================
# FINAL INSTANCES
# ============================================================

INSTANCES = [
    "G14",
    "G23",
    "G60",
    "G64",
]


# ============================================================
# EPSILON / ER
# ============================================================

# Used to determine the strict statevector validation point.
ER_CANDIDATES = [
    0.02,
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
]


# Strict validation criteria.
#
# This reproduces the frozen Stage-05 solution before finite-shot
# sampling is considered.
MIN_STATE_FIDELITY = 0.999
MAX_TARGET_TV = 0.02
MIN_CORRELATOR_SIGN_AGREEMENT = 1.0
MIN_RAW_SPIN_AGREEMENT = 1.0

CLASSICAL_CORRELATOR_ATOL = 1.0e-10
CIRCUIT_ANALYTIC_ATOL = 1.0e-10


# ============================================================
# FINITE-SHOT OPERATING POINT
# ============================================================

# The epsilon sweep showed that er=0.20 gives approximately
#
#     P_success ~ 0.01
#
# while preserving the Stage-05 objective values extremely well:
#
# max |Delta AR_raw|   < 3.3e-4
# max |Delta AR_local| < 6.0e-4
#
# Therefore:
#
# - chosen_er     -> strict statevector validation
# - SAMPLING_ER   -> finite shots and noise
#
# This is NOT a retraining parameter.
SAMPLING_ER = 0.20


# ============================================================
# SHOTS
# ============================================================

# One circuit execution is performed at the maximum budget.
# Smaller budgets use prefixes of the same sample sequence.
SHOT_BUDGETS = [
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
]

SHOT_SEED = 20260912


# ============================================================
# NOISE
# ============================================================

NOISE_LEVELS = {
    "low": {
        "p1": 1.0e-4,
        "p2": 1.0e-3,
        "p_readout": 1.0e-3,
    },
    "moderate": {
        "p1": 1.0e-3,
        "p2": 1.0e-2,
        "p_readout": 1.0e-2,
    },
}

NOISE_TRAJECTORIES = 32
NOISE_SHOTS_PER_TRAJECTORY = 2048
NOISE_SEED = 20260913


# ============================================================
# DEVICE
# ============================================================

DEVICE_PREFERENCE = [
    "lightning.qubit",
    "default.qubit",
]

SCRIPT_VERSION = 3


# ============================================================
# IO
# ============================================================

def atomic_csv(path, frame):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )


def atomic_json(path, payload):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def write_npz(path, arrays):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        **arrays,
    )


# ============================================================
# GSET LOADER
# ============================================================

def load_gset(path):

    path = Path(path)

    with path.open("r") as f:

        first = f.readline().split()

        n = int(first[0])
        m = int(first[1])

        edges = []

        for line in f:

            line = line.strip()

            if not line:
                continue

            a, b, w = line.split()[:3]

            edges.append(
                (
                    int(a) - 1,
                    int(b) - 1,
                    float(w),
                )
            )

    if len(edges) != m:

        raise RuntimeError(
            f"{path}: "
            f"expected {m} edges, "
            f"found {len(edges)}"
        )

    u = np.asarray(
        [edge[0] for edge in edges],
        dtype=np.int64,
    )

    v = np.asarray(
        [edge[1] for edge in edges],
        dtype=np.int64,
    )

    w = np.asarray(
        [edge[2] for edge in edges],
        dtype=np.float64,
    )

    return n, u, v, w


# ============================================================
# CUT
#
# EXACT STAGE-05 CONVENTION
# ============================================================

def cut_value(
    spins,
    u,
    v,
    w,
):

    spins = np.asarray(
        spins,
        dtype=np.int8,
    )

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
# ADJACENCY
# ============================================================

def build_adjacency(
    n,
    u,
    v,
    w,
):

    adjacency = [
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

        adjacency[a].append(
            (
                b,
                weight,
            )
        )

        adjacency[b].append(
            (
                a,
                weight,
            )
        )

    return adjacency


# ============================================================
# LOCAL SEARCH
#
# EXACT STAGE-05 ORDER:
#
#   0 -> n-1
#   n-1 -> 0
# ============================================================

def local_search_forward_backward(
    spins,
    adjacency,
):

    z = np.asarray(
        spins,
        dtype=np.int8,
    ).copy()

    n = len(z)

    flips_forward = 0
    flips_backward = 0

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    for i in range(n):

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

            flips_forward += 1

    # --------------------------------------------------------
    # BACKWARD
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
# AGREEMENT
# ============================================================

def sign_vector(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return np.where(
        values >= 0.0,
        1,
        -1,
    ).astype(
        np.int8
    )


def sign_agreement(
    reference,
    estimate,
):

    return float(
        np.mean(
            sign_vector(reference)
            ==
            sign_vector(estimate)
        )
    )


def spin_agreement(
    reference,
    estimate,
):

    return float(
        np.mean(
            np.asarray(reference)
            ==
            np.asarray(estimate)
        )
    )


# ============================================================
# LOAD STAGE 05 SELECTION
# ============================================================

def load_selected():

    if not STAGE5_SELECTED.is_file():

        raise FileNotFoundError(
            STAGE5_SELECTED
        )

    if not STAGE5_SUMMARY.is_file():

        raise FileNotFoundError(
            STAGE5_SUMMARY
        )

    selected = pd.read_csv(
        STAGE5_SELECTED
    )

    summary = pd.read_csv(
        STAGE5_SUMMARY
    )

    required_selected = {
        "protocol_id",
        "instance",
        "init_seed",
        "raw_cut",
        "raw_AR",
        "local_cut",
        "local_AR",
        "npz",
    }

    missing = (
        required_selected
        - set(selected.columns)
    )

    if missing:

        raise RuntimeError(
            "selected_best_raw.csv missing columns: "
            f"{sorted(missing)}"
        )

    selected = selected[
        selected[
            "instance"
        ].isin(
            INSTANCES
        )
    ].copy()

    if (
        set(
            selected[
                "instance"
            ]
        )
        !=
        set(
            INSTANCES
        )
    ):

        raise RuntimeError(
            "Wrong Stage-05 instance coverage."
        )

    if not (
        selected[
            "protocol_id"
        ]
        ==
        PROTOCOL_ID
    ).all():

        raise RuntimeError(
            "Stage-05 CSV protocol mismatch."
        )

    required_summary = {
        "instance",
        "n",
        "q",
        "k",
        "reference",
    }

    missing = (
        required_summary
        - set(summary.columns)
    )

    if missing:

        raise RuntimeError(
            "stage_05/summary.csv missing columns: "
            f"{sorted(missing)}"
        )

    metadata = summary[
        [
            "instance",
            "n",
            "q",
            "k",
            "reference",
        ]
    ].copy()

    selected = selected.merge(
        metadata,
        on="instance",
        how="left",
        validate="one_to_one",
    )

    return (
        selected
        .sort_values(
            "instance"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOAD FROZEN STAGE-05 STATE
# ============================================================

def load_record(row):

    instance = str(
        row["instance"]
    )

    q = int(
        row["q"]
    )

    k = int(
        row["k"]
    )

    n = int(
        row["n"]
    )

    reference = float(
        row["reference"]
    )

    path = Path(
        row["npz"]
    )

    if not path.is_absolute():

        path = (
            ROOT
            / path
        )

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "active_indices",
            "coefficients",
            "correlators",
            "relaxed_spins",
            "raw_spins",
            "local_spins",
            "protocol_id",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:

            raise RuntimeError(
                f"{path} missing "
                f"{sorted(missing)}"
            )

        npz_protocol = str(
            data[
                "protocol_id"
            ].item()
        )

        if (
            npz_protocol
            !=
            PROTOCOL_ID
        ):

            raise RuntimeError(
                f"{path}: "
                f"protocol mismatch "
                f"{npz_protocol!r} "
                f"!= {PROTOCOL_ID!r}"
            )

        active_indices = np.asarray(
            data[
                "active_indices"
            ],
            dtype=np.int64,
        ).copy()

        coefficients = np.asarray(
            data[
                "coefficients"
            ],
            dtype=np.float64,
        ).copy()

        classical_correlators = np.asarray(
            data[
                "correlators"
            ],
            dtype=np.float64,
        ).copy()

        raw_spins = np.asarray(
            data[
                "raw_spins"
            ],
            dtype=np.int8,
        ).copy()

        local_spins = np.asarray(
            data[
                "local_spins"
            ],
            dtype=np.int8,
        ).copy()

    # ========================================================
    # DENSE WALSH VECTOR
    # ========================================================

    dense_raw = dense_coefficients(
        coefficients,
        active_indices,
        q,
    )

    coefficient_norm = float(
        np.linalg.norm(
            dense_raw
        )
    )

    if coefficient_norm <= 0.0:

        raise RuntimeError(
            f"{instance}: "
            "null coefficient vector."
        )

    # Global scale does not modify the normalized target state
    # or the Walsh correlators.
    dense = (
        dense_raw
        /
        coefficient_norm
    )

    target = target_walsh_outputs(
        dense
    )

    # ========================================================
    # DECISION MASKS
    # ========================================================

    masks = (
        decision_masks(
            q=q,
            k=k,
            m=n,
        )
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )

    # ========================================================
    # CLASSICAL WALSH RECONSTRUCTION AUDIT
    # ========================================================

    target_correlators = (
        correlators_from_probabilities(
            target[
                "probabilities"
            ],
            masks,
        )
    )

    max_error = float(
        np.max(
            np.abs(
                target_correlators
                -
                classical_correlators
            )
        )
    )

    if (
        max_error
        >
        CLASSICAL_CORRELATOR_ATOL
    ):

        raise RuntimeError(
            f"{instance}: "
            "Walsh reconstruction does not "
            "reproduce Stage 05. "
            f"max error={max_error:.3e}"
        )

    # ========================================================
    # GRAPH
    # ========================================================

    graph_path = (
        DATA_DIR
        / f"{instance}.txt"
    )

    (
        graph_n,
        u,
        v,
        w,
    ) = load_gset(
        graph_path
    )

    if graph_n != n:

        raise RuntimeError(
            f"{instance}: "
            f"graph n={graph_n}, "
            f"Stage-05 n={n}"
        )

    adjacency = build_adjacency(
        n,
        u,
        v,
        w,
    )

    # ========================================================
    # RAW DECODING AUDIT
    # ========================================================

    reconstructed_raw = sign_vector(
        classical_correlators
    )

    if not np.array_equal(
        reconstructed_raw,
        raw_spins,
    ):

        raise RuntimeError(
            f"{instance}: "
            "raw spin reconstruction mismatch."
        )

    raw_cut = cut_value(
        raw_spins,
        u,
        v,
        w,
    )

    if not np.isclose(
        raw_cut,
        float(
            row[
                "raw_cut"
            ]
        ),
        atol=1.0e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{instance}: "
            "raw cut mismatch."
        )

    # ========================================================
    # LOCAL SEARCH AUDIT
    # ========================================================

    (
        reconstructed_local,
        flips_forward,
        flips_backward,
    ) = local_search_forward_backward(
        raw_spins,
        adjacency,
    )

    if not np.array_equal(
        reconstructed_local,
        local_spins,
    ):

        raise RuntimeError(
            f"{instance}: "
            "local search reconstruction mismatch."
        )

    local_cut = cut_value(
        local_spins,
        u,
        v,
        w,
    )

    if not np.isclose(
        local_cut,
        float(
            row[
                "local_cut"
            ]
        ),
        atol=1.0e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{instance}: "
            "local cut mismatch."
        )

    return {
        "instance":
            instance,

        "seed":
            int(
                row[
                    "init_seed"
                ]
            ),

        "n":
            n,

        "q":
            q,

        "k":
            k,

        "reference":
            reference,

        "npz":
            path,

        "active_indices":
            active_indices,

        "coefficients":
            coefficients,

        "dense_coefficients":
            dense,

        "coefficient_norm":
            coefficient_norm,

        "classical_correlators":
            classical_correlators,

        "raw_spins":
            raw_spins,

        "local_spins":
            local_spins,

        "raw_cut_stage5":
            raw_cut,

        "local_cut_stage5":
            local_cut,

        "u":
            u,

        "v":
            v,

        "w":
            w,

        "adjacency":
            adjacency,

        "masks":
            masks,

        "target_state":
            np.asarray(
                target[
                    "state"
                ],
                dtype=np.complex128,
            ),

        "target_probabilities":
            np.asarray(
                target[
                    "probabilities"
                ],
                dtype=np.float64,
            ),

        "max_abs_function":
            float(
                target[
                    "max_abs_function"
                ]
            ),

        "max_classical_correlator_error":
            max_error,

        "resources":
            resource_counts(
                active_indices,
                q,
            ),

        "flips_forward_stage5":
            flips_forward,

        "flips_backward_stage5":
            flips_backward,
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibration_case(
    record,
    er,
):

    analytical = (
        analytical_loader_outputs(
            record[
                "dense_coefficients"
            ],
            er,
        )
    )

    correlators = (
        correlators_from_probabilities(
            analytical[
                "probabilities"
            ],
            record[
                "masks"
            ],
        )
    )

    raw_spins = sign_vector(
        correlators
    )

    return {
        "instance":
            record[
                "instance"
            ],

        "seed":
            record[
                "seed"
            ],

        "er":
            float(er),

        "max_phase":
            float(er)
            *
            record[
                "max_abs_function"
            ],

        "success_probability":
            float(
                analytical[
                    "success_probability"
                ]
            ),

        "state_fidelity_to_target":
            state_fidelity(
                record[
                    "target_state"
                ],
                analytical[
                    "state"
                ],
            ),

        "tv_to_target":
            total_variation(
                record[
                    "target_probabilities"
                ],
                analytical[
                    "probabilities"
                ],
            ),

        "correlator_mae":
            float(
                np.mean(
                    np.abs(
                        correlators
                        -
                        record[
                            "classical_correlators"
                        ]
                    )
                )
            ),

        "correlator_maxae":
            float(
                np.max(
                    np.abs(
                        correlators
                        -
                        record[
                            "classical_correlators"
                        ]
                    )
                )
            ),

        "correlator_sign_agreement":
            sign_agreement(
                record[
                    "classical_correlators"
                ],
                correlators,
            ),

        "raw_spin_agreement":
            spin_agreement(
                record[
                    "raw_spins"
                ],
                raw_spins,
            ),
    }


def choose_global_er(
    calibration,
):

    rows = []

    for er, group in calibration.groupby(
        "er",
        sort=True,
    ):

        valid = bool(

            (
                group[
                    "state_fidelity_to_target"
                ]
                >=
                MIN_STATE_FIDELITY
            ).all()

            and

            (
                group[
                    "tv_to_target"
                ]
                <=
                MAX_TARGET_TV
            ).all()

            and

            (
                group[
                    "correlator_sign_agreement"
                ]
                >=
                MIN_CORRELATOR_SIGN_AGREEMENT
            ).all()

            and

            (
                group[
                    "raw_spin_agreement"
                ]
                >=
                MIN_RAW_SPIN_AGREEMENT
            ).all()
        )

        rows.append(
            {
                "er":
                    float(er),

                "valid_global":
                    valid,

                "min_state_fidelity":
                    float(
                        group[
                            "state_fidelity_to_target"
                        ].min()
                    ),

                "max_tv":
                    float(
                        group[
                            "tv_to_target"
                        ].max()
                    ),

                "max_phase":
                    float(
                        group[
                            "max_phase"
                        ].max()
                    ),

                "min_success_probability":
                    float(
                        group[
                            "success_probability"
                        ].min()
                    ),

                "mean_success_probability":
                    float(
                        group[
                            "success_probability"
                        ].mean()
                    ),

                "min_sign_agreement":
                    float(
                        group[
                            "correlator_sign_agreement"
                        ].min()
                    ),

                "min_raw_spin_agreement":
                    float(
                        group[
                            "raw_spin_agreement"
                        ].min()
                    ),
            }
        )

    summary = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "er"
        )
        .reset_index(
            drop=True
        )
    )

    valid = summary[
        summary[
            "valid_global"
        ]
    ]

    if valid.empty:

        raise RuntimeError(
            "No globally valid strict er candidate."
        )

    chosen = float(
        valid.iloc[-1][
            "er"
        ]
    )

    summary[
        "selected"
    ] = np.isclose(
        summary[
            "er"
        ],
        chosen,
    )

    return (
        chosen,
        summary,
    )


# ============================================================
# DECODE QUANTUM PROBABILITIES
# ============================================================

def decode_probabilities(
    record,
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    correlators = (
        correlators_from_probabilities(
            probabilities,
            record[
                "masks"
            ],
        )
    )

    raw_spins = sign_vector(
        correlators
    )

    raw_cut = cut_value(
        raw_spins,
        record[
            "u"
        ],
        record[
            "v"
        ],
        record[
            "w"
        ],
    )

    (
        local_spins,
        flips_forward,
        flips_backward,
    ) = local_search_forward_backward(
        raw_spins,
        record[
            "adjacency"
        ],
    )

    local_cut = cut_value(
        local_spins,
        record[
            "u"
        ],
        record[
            "v"
        ],
        record[
            "w"
        ],
    )

    reference = record[
        "reference"
    ]

    delta_c = (
        correlators
        -
        record[
            "classical_correlators"
        ]
    )

    return {
        "correlator_mae":
            float(
                np.mean(
                    np.abs(
                        delta_c
                    )
                )
            ),

        "correlator_maxae":
            float(
                np.max(
                    np.abs(
                        delta_c
                    )
                )
            ),

        "correlator_rmse":
            float(
                np.sqrt(
                    np.mean(
                        delta_c
                        *
                        delta_c
                    )
                )
            ),

        "correlator_sign_agreement":
            sign_agreement(
                record[
                    "classical_correlators"
                ],
                correlators,
            ),

        "raw_spin_agreement":
            spin_agreement(
                record[
                    "raw_spins"
                ],
                raw_spins,
            ),

        "local_spin_agreement":
            spin_agreement(
                record[
                    "local_spins"
                ],
                local_spins,
            ),

        "raw_cut":
            raw_cut,

        "raw_AR":
            raw_cut
            /
            reference,

        "local_cut":
            local_cut,

        "local_AR":
            local_cut
            /
            reference,

        "raw_cut_difference_stage5":
            raw_cut
            -
            record[
                "raw_cut_stage5"
            ],

        "local_cut_difference_stage5":
            local_cut
            -
            record[
                "local_cut_stage5"
            ],

        "flips_forward":
            flips_forward,

        "flips_backward":
            flips_backward,

        "_correlators":
            correlators,

        "_raw_spins":
            raw_spins,

        "_local_spins":
            local_spins,
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    n = 3

    indices = np.asarray(
        [
            0,
            1,
            3,
            5,
            6,
        ],
        dtype=np.int64,
    )

    coeffs = np.asarray(
        [
            0.7,
            -0.2,
            0.3,
            0.1,
            -0.15,
        ],
        dtype=np.float64,
    )

    dense = dense_coefficients(
        coeffs,
        indices,
        n,
    )

    dense /= np.linalg.norm(
        dense
    )

    er = 0.2

    analytical = (
        analytical_loader_outputs(
            dense,
            er,
        )
    )

    original = (
        run_original_loader_statevector(
            dense,
            indices,
            n,
            er,
        )
    )

    optimized = (
        run_loader_statevector(
            dense,
            indices,
            n,
            er,
            device_preference=[
                "default.qubit"
            ],
        )
    )

    fidelity_original = state_fidelity(
        analytical[
            "state"
        ],
        original[
            "postselected_state"
        ],
    )

    fidelity_optimized = state_fidelity(
        analytical[
            "state"
        ],
        optimized[
            "postselected_state"
        ],
    )

    fidelity_between = state_fidelity(
        original[
            "postselected_state"
        ],
        optimized[
            "postselected_state"
        ],
    )

    success_error = abs(
        float(
            analytical[
                "success_probability"
            ]
        )
        -
        float(
            optimized[
                "success_probability_ancilla_1"
            ]
        )
    )

    if (
        1.0
        -
        fidelity_original
        >
        1.0e-12
    ):

        raise RuntimeError(
            "Original WSL failed."
        )

    if (
        1.0
        -
        fidelity_optimized
        >
        1.0e-12
    ):

        raise RuntimeError(
            "Optimized WSL failed."
        )

    if (
        1.0
        -
        fidelity_between
        >
        1.0e-12
    ):

        raise RuntimeError(
            "Original and optimized WSL differ."
        )

    if success_error > 1.0e-12:

        raise RuntimeError(
            "WSL success probability mismatch."
        )

    print()
    print("=" * 70)
    print("STAGE 06 SELF TEST")
    print("=" * 70)

    print(
        f"original fidelity  = "
        f"{fidelity_original:.16f}"
    )

    print(
        f"optimized fidelity = "
        f"{fidelity_optimized:.16f}"
    )

    print(
        f"between circuits   = "
        f"{fidelity_between:.16f}"
    )

    print(
        f"success prob error = "
        f"{success_error:.3e}"
    )

    print()
    print("SELF-TEST OK")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--statevector-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-noise",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # SELF TEST
    # ========================================================

    if args.self_test:

        self_test()

        return

    # ========================================================
    # LOAD STAGE 05
    # ========================================================

    selected = load_selected()

    records = {}

    print()
    print("=" * 90)
    print("STAGE 06 — QUANTUM TRANSFER")
    print("=" * 90)

    for _, row in selected.iterrows():

        record = load_record(
            row
        )

        records[
            record[
                "instance"
            ]
        ] = record

        print(
            f"{record['instance']}: "
            f"seed={record['seed']} | "
            f"n={record['n']} | "
            f"q={record['q']} | "
            f"active={len(record['active_indices'])} | "
            f"raw={record['raw_cut_stage5']:.1f} | "
            f"local={record['local_cut_stage5']:.1f} | "
            f"Walsh error="
            f"{record['max_classical_correlator_error']:.3e}"
        )

    # ========================================================
    # DRY RUN
    # ========================================================

    if args.dry_run:

        print()
        print("=" * 90)
        print("CIRCUIT RESOURCES")
        print("=" * 90)

        rows = []

        for instance in INSTANCES:

            record = records[
                instance
            ]

            rows.append(
                {
                    "instance":
                        instance,

                    "seed":
                        record[
                            "seed"
                        ],

                    **asdict(
                        record[
                            "resources"
                        ]
                    ),
                }
            )

        print(
            pd.DataFrame(
                rows
            ).to_string(
                index=False
            )
        )

        print()

        print(
            "strict er candidates:",
            ER_CANDIDATES,
        )

        print(
            "sampling er:",
            SAMPLING_ER,
        )

        print(
            "shot budgets:",
            SHOT_BUDGETS,
        )

        print()

        print(
            "Expected accepted shots "
            "at sampling er:"
        )

        for instance in INSTANCES:

            record = records[
                instance
            ]

            analytical = analytical_loader_outputs(
                record[
                    "dense_coefficients"
                ],
                SAMPLING_ER,
            )

            p_success = float(
                analytical[
                    "success_probability"
                ]
            )

            values = [
                int(
                    round(
                        budget
                        *
                        p_success
                    )
                )
                for budget in SHOT_BUDGETS
            ]

            print(
                f"  {instance}: "
                f"P_success={p_success:.8f} | "
                f"{values}"
            )

        return

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    calibration_dir = (
        OUT_DIR
        / "01_calibration"
    )

    statevector_dir = (
        OUT_DIR
        / "02_statevector"
    )

    shots_dir = (
        OUT_DIR
        / "03_ideal_shots"
    )

    noise_dir = (
        OUT_DIR
        / "04_noise"
    )

    summary_dir = (
        OUT_DIR
        / "05_summary"
    )

    for directory in [
        calibration_dir,
        statevector_dir,
        shots_dir,
        noise_dir,
        summary_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # STRICT ER CALIBRATION
    # ========================================================

    calibration_rows = []

    for instance in INSTANCES:

        for er in ER_CANDIDATES:

            calibration_rows.append(
                calibration_case(
                    records[
                        instance
                    ],
                    er,
                )
            )

    calibration = pd.DataFrame(
        calibration_rows
    )

    (
        _automatic_er,
        calibration_summary,
    ) = choose_global_er(
        calibration
    )

    # Final statevector-transfer protocol.
    # Same epsilon used in the manuscript validation.
    chosen_er = 0.02

    calibration_summary["selected"] = np.isclose(
        calibration_summary["er"],
        chosen_er,
    )

    atomic_csv(
        calibration_dir
        / "epsilon_calibration_all.csv",
        calibration,
    )

    atomic_csv(
        calibration_dir
        / "epsilon_calibration_summary.csv",
        calibration_summary,
    )

    atomic_json(
        calibration_dir
        / "epsilon_selection.json",
        {
            "validation_er":
                chosen_er,

            "sampling_er":
                SAMPLING_ER,

            "strict_criteria":
                {
                    "min_state_fidelity":
                        MIN_STATE_FIDELITY,

                    "max_target_tv":
                        MAX_TARGET_TV,

                    "min_correlator_sign_agreement":
                        MIN_CORRELATOR_SIGN_AGREEMENT,

                    "min_raw_spin_agreement":
                        MIN_RAW_SPIN_AGREEMENT,
                },
        },
    )

    print()

    print(
        f"Selected strict validation er = "
        f"{chosen_er}"
    )

    print(
        f"Finite-shot sampling er       = "
        f"{SAMPLING_ER}"
    )

    # ========================================================
    # STATEVECTOR
    #
    # STRICT VALIDATION POINT
    # ========================================================

    statevector_rows = []

    for instance in INSTANCES:

        record = records[
            instance
        ]

        print()
        print(
            f"STATEVECTOR: {instance}"
        )

        analytical = (
            analytical_loader_outputs(
                record[
                    "dense_coefficients"
                ],
                chosen_er,
            )
        )

        start = time.perf_counter()

        result = (
            run_loader_statevector(
                record[
                    "dense_coefficients"
                ],
                record[
                    "active_indices"
                ],
                record[
                    "q"
                ],
                chosen_er,
                device_preference=
                    DEVICE_PREFERENCE,
            )
        )

        wall = (
            time.perf_counter()
            -
            start
        )

        probabilities = np.asarray(
            result[
                "postselected_probabilities"
            ],
            dtype=np.float64,
        )

        fidelity_analytic = (
            state_fidelity(
                result[
                    "postselected_state"
                ],
                analytical[
                    "state"
                ],
            )
        )

        tv_analytic = (
            total_variation(
                probabilities,
                analytical[
                    "probabilities"
                ],
            )
        )

        if (
            1.0
            -
            fidelity_analytic
            >
            CIRCUIT_ANALYTIC_ATOL
        ):

            raise RuntimeError(
                f"{instance}: "
                "circuit fidelity failure."
            )

        if (
            tv_analytic
            >
            CIRCUIT_ANALYTIC_ATOL
        ):

            raise RuntimeError(
                f"{instance}: "
                "circuit TV failure."
            )

        decoded = (
            decode_probabilities(
                record,
                probabilities,
            )
        )

        row = {
            "instance":
                instance,

            "seed":
                record[
                    "seed"
                ],

            "er":
                chosen_er,

            "device":
                result[
                    "device"
                ],

            "success_probability":
                float(
                    result[
                        "success_probability_ancilla_1"
                    ]
                ),

            "state_fidelity_analytic":
                fidelity_analytic,

            "state_fidelity_target":
                state_fidelity(
                    result[
                        "postselected_state"
                    ],
                    record[
                        "target_state"
                    ],
                ),

            "tv_analytic":
                tv_analytic,

            "tv_target":
                total_variation(
                    probabilities,
                    record[
                        "target_probabilities"
                    ],
                ),

            "wall_s":
                wall,

            **{
                key: value
                for key, value
                in decoded.items()
                if not key.startswith("_")
            },
        }

        statevector_rows.append(
            row
        )

        write_npz(
            statevector_dir
            /
            f"{instance}_statevector.npz",

            {
                "probabilities":
                    probabilities,

                "state":
                    np.asarray(
                        result[
                            "postselected_state"
                        ],
                        dtype=np.complex128,
                    ),

                "correlators":
                    decoded[
                        "_correlators"
                    ],

                "raw_spins":
                    decoded[
                        "_raw_spins"
                    ],

                "local_spins":
                    decoded[
                        "_local_spins"
                    ],
            },
        )

        print(
            f"  F(target)="
            f"{row['state_fidelity_target']:.8f}"
            f" | raw AR="
            f"{row['raw_AR']:.6f}"
            f" | local AR="
            f"{row['local_AR']:.6f}"
        )

    statevector_df = pd.DataFrame(
        statevector_rows
    )

    atomic_csv(
        summary_dir
        / "statevector_summary.csv",
        statevector_df,
    )

    if args.statevector_only:

        print()
        print(
            "Statevector-only finished."
        )

        return

    # ========================================================
    # EXACT SAMPLING REFERENCES
    #
    # Finite-shot and noisy results must be compared against
    # the exact loader distribution at SAMPLING_ER, not against
    # the strict statevector distribution at chosen_er.
    # ========================================================

    sampling_exact_probabilities = {}
    sampling_reference_rows = []

    print()
    print("=" * 90)
    print(
        f"SAMPLING REFERENCE — er={SAMPLING_ER}"
    )
    print("=" * 90)

    for instance in INSTANCES:

        record = records[
            instance
        ]

        sampling_analytical = (
            analytical_loader_outputs(
                record[
                    "dense_coefficients"
                ],
                SAMPLING_ER,
            )
        )

        probabilities = np.asarray(
            sampling_analytical[
                "probabilities"
            ],
            dtype=np.float64,
        )

        sampling_exact_probabilities[
            instance
        ] = probabilities

        decoded = decode_probabilities(
            record,
            probabilities,
        )

        p_success = float(
            sampling_analytical[
                "success_probability"
            ]
        )

        sampling_reference_rows.append(
            {
                "instance":
                    instance,

                "seed":
                    record[
                        "seed"
                    ],

                "er":
                    SAMPLING_ER,

                "success_probability":
                    p_success,

                "state_fidelity_target":
                    state_fidelity(
                        record[
                            "target_state"
                        ],
                        sampling_analytical[
                            "state"
                        ],
                    ),

                "tv_target":
                    total_variation(
                        record[
                            "target_probabilities"
                        ],
                        probabilities,
                    ),

                **{
                    key: value
                    for key, value
                    in decoded.items()
                    if not key.startswith("_")
                },
            }
        )

        expected_200k = (
            200_000
            *
            p_success
        )

        print(
            f"{instance}: "
            f"P_success={p_success:.8f} | "
            f"expected accepted@200k="
            f"{expected_200k:.1f} | "
            f"raw AR={decoded['raw_AR']:.6f} | "
            f"local AR={decoded['local_AR']:.6f}"
        )

    sampling_reference_df = pd.DataFrame(
        sampling_reference_rows
    )

    atomic_csv(
        summary_dir
        / "sampling_reference_summary.csv",
        sampling_reference_df,
    )

    # ========================================================
    # IDEAL SHOTS
    #
    # OPERATING POINT: SAMPLING_ER
    # ========================================================

    shot_rows = []

    for instance in INSTANCES:

        record = records[
            instance
        ]

        max_shots = max(
            SHOT_BUDGETS
        )

        print()
        print(
            f"IDEAL SHOTS: {instance}"
        )

        print(
            f"  er={SAMPLING_ER}, "
            f"max shots={max_shots}"
        )

        start = time.perf_counter()

        result = run_loader_samples(
            record[
                "dense_coefficients"
            ],
            record[
                "active_indices"
            ],
            record[
                "q"
            ],
            SAMPLING_ER,
            shots=max_shots,
            seed=(
                SHOT_SEED
                +
                record[
                    "seed"
                ]
            ),
            device_preference=
                DEVICE_PREFERENCE,
        )

        wall = (
            time.perf_counter()
            -
            start
        )

        samples = np.asarray(
            result[
                "samples"
            ],
            dtype=np.int8,
        )

        # Keep one full sample artifact per instance.
        write_npz(
            shots_dir
            /
            f"{instance}_shots.npz",
            {
                "samples":
                    samples,
            },
        )

        for budget in SHOT_BUDGETS:

            histogram = (
                postselected_histogram(
                    samples[
                        :budget
                    ],
                    record[
                        "q"
                    ],
                )
            )

            accepted = int(
                histogram[
                    "accepted_shots"
                ]
            )

            if accepted == 0:

                print(
                    f"  {budget}: "
                    "accepted=0"
                )

                shot_rows.append(
                    {
                        "instance":
                            instance,

                        "seed":
                            record[
                                "seed"
                            ],

                        "er":
                            SAMPLING_ER,

                        "shot_budget":
                            budget,

                        "accepted_shots":
                            0,

                        "success_probability":
                            0.0,

                        "tv_exact":
                            np.nan,

                        "correlator_mae":
                            np.nan,

                        "correlator_maxae":
                            np.nan,

                        "correlator_rmse":
                            np.nan,

                        "correlator_sign_agreement":
                            np.nan,

                        "raw_spin_agreement":
                            np.nan,

                        "local_spin_agreement":
                            np.nan,

                        "raw_cut":
                            np.nan,

                        "raw_AR":
                            np.nan,

                        "local_cut":
                            np.nan,

                        "local_AR":
                            np.nan,

                        "raw_cut_difference_stage5":
                            np.nan,

                        "local_cut_difference_stage5":
                            np.nan,

                        "flips_forward":
                            0,

                        "flips_backward":
                            0,

                        "wall_s_full_sample":
                            wall,
                    }
                )

                continue

            probabilities = np.asarray(
                histogram[
                    "probabilities"
                ],
                dtype=np.float64,
            )

            decoded = (
                decode_probabilities(
                    record,
                    probabilities,
                )
            )

            row = {
                "instance":
                    instance,

                "seed":
                    record[
                        "seed"
                    ],

                "er":
                    SAMPLING_ER,

                "shot_budget":
                    budget,

                "accepted_shots":
                    accepted,

                "success_probability":
                    float(
                        histogram[
                            "observed_success_probability"
                        ]
                    ),

                "tv_exact":
                    total_variation(
                        probabilities,
                        sampling_exact_probabilities[
                            instance
                        ],
                    ),

                "wall_s_full_sample":
                    wall,

                **{
                    key: value
                    for key, value
                    in decoded.items()
                    if not key.startswith("_")
                },
            }

            shot_rows.append(
                row
            )

            print(
                f"  {budget}: "
                f"accepted={accepted} | "
                f"raw AR="
                f"{decoded['raw_AR']:.6f} | "
                f"local AR="
                f"{decoded['local_AR']:.6f}"
            )

    shots_df = pd.DataFrame(
        shot_rows
    )

    atomic_csv(
        summary_dir
        / "ideal_shots_summary.csv",
        shots_df,
    )

    # ========================================================
    # NOISE
    #
    # SAME OPERATING POINT USED FOR FINITE SHOTS
    # ========================================================

    noise_rows = []

    if not args.skip_noise:

        for instance in INSTANCES:

            record = records[
                instance
            ]

            for (
                level_name,
                level,
            ) in NOISE_LEVELS.items():

                print()
                print(
                    f"NOISE: "
                    f"{instance} / "
                    f"{level_name}"
                )

                offset = sum(
                    ord(char)
                    for char
                    in instance
                )

                start = time.perf_counter()

                result = (
                    run_pauli_trajectory_samples(

                        record[
                            "dense_coefficients"
                        ],

                        record[
                            "active_indices"
                        ],

                        record[
                            "q"
                        ],

                        SAMPLING_ER,

                        trajectories=
                            NOISE_TRAJECTORIES,

                        shots_per_trajectory=
                            NOISE_SHOTS_PER_TRAJECTORY,

                        base_seed=(
                            NOISE_SEED
                            +
                            1000
                            *
                            record[
                                "seed"
                            ]
                            +
                            100_000
                            *
                            offset
                            +
                            (
                                1
                                if
                                level_name
                                ==
                                "low"
                                else
                                2
                            )
                        ),

                        p1=
                            level[
                                "p1"
                            ],

                        p2=
                            level[
                                "p2"
                            ],

                        p_readout=
                            level[
                                "p_readout"
                            ],

                        device_preference=
                            DEVICE_PREFERENCE,
                    )
                )

                wall = (
                    time.perf_counter()
                    -
                    start
                )

                samples = np.asarray(
                    result[
                        "samples"
                    ],
                    dtype=np.int8,
                )

                write_npz(
                    noise_dir
                    /
                    (
                        f"{instance}_"
                        f"{level_name}.npz"
                    ),
                    {
                        "samples":
                            samples,
                    },
                )

                histogram = (
                    postselected_histogram(
                        samples,
                        record[
                            "q"
                        ],
                    )
                )

                accepted = int(
                    histogram[
                        "accepted_shots"
                    ]
                )

                if accepted == 0:

                    print(
                        "  accepted=0"
                    )

                    noise_rows.append(
                        {
                            "instance":
                                instance,

                            "seed":
                                record[
                                    "seed"
                                ],

                            "er":
                                SAMPLING_ER,

                            "noise_level":
                                level_name,

                            "p1":
                                level[
                                    "p1"
                                ],

                            "p2":
                                level[
                                    "p2"
                                ],

                            "p_readout":
                                level[
                                    "p_readout"
                                ],

                            "total_shots":
                                int(
                                    histogram[
                                        "total_shots"
                                    ]
                                ),

                            "accepted_shots":
                                0,

                            "success_probability":
                                0.0,

                            "tv_exact":
                                np.nan,

                            "raw_AR":
                                np.nan,

                            "local_AR":
                                np.nan,

                            "wall_s":
                                wall,
                        }
                    )

                    continue

                probabilities = np.asarray(
                    histogram[
                        "probabilities"
                    ],
                    dtype=np.float64,
                )

                decoded = (
                    decode_probabilities(
                        record,
                        probabilities,
                    )
                )

                row = {
                    "instance":
                        instance,

                    "seed":
                        record[
                            "seed"
                        ],

                    "er":
                        SAMPLING_ER,

                    "noise_level":
                        level_name,

                    "p1":
                        level[
                            "p1"
                        ],

                    "p2":
                        level[
                            "p2"
                        ],

                    "p_readout":
                        level[
                            "p_readout"
                        ],

                    "trajectories":
                        NOISE_TRAJECTORIES,

                    "shots_per_trajectory":
                        NOISE_SHOTS_PER_TRAJECTORY,

                    "total_shots":
                        int(
                            histogram[
                                "total_shots"
                            ]
                        ),

                    "accepted_shots":
                        accepted,

                    "success_probability":
                        float(
                            histogram[
                                "observed_success_probability"
                            ]
                        ),

                    "tv_exact":
                        total_variation(
                            probabilities,
                            sampling_exact_probabilities[
                                instance
                            ],
                        ),

                    "wall_s":
                        wall,

                    **{
                        key: value
                        for key, value
                        in decoded.items()
                        if not key.startswith("_")
                    },
                }

                noise_rows.append(
                    row
                )

                print(
                    f"  accepted="
                    f"{accepted}"
                    f" | raw AR="
                    f"{decoded['raw_AR']:.6f}"
                    f" | local AR="
                    f"{decoded['local_AR']:.6f}"
                )

    noise_df = pd.DataFrame(
        noise_rows
    )

    atomic_csv(
        summary_dir
        / "noise_summary.csv",
        noise_df,
    )

    # ========================================================
    # CIRCUIT RESOURCES
    # ========================================================

    resource_rows = []

    for instance in INSTANCES:

        record = records[
            instance
        ]

        resource_rows.append(
            {
                "instance":
                    instance,

                "seed":
                    record[
                        "seed"
                    ],

                **asdict(
                    record[
                        "resources"
                    ]
                ),
            }
        )

    resources_df = pd.DataFrame(
        resource_rows
    )

    atomic_csv(
        summary_dir
        / "circuit_resources.csv",
        resources_df,
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    atomic_json(
        summary_dir
        / "manifest.json",

        {
            "script_version":
                SCRIPT_VERSION,

            "protocol_id":
                PROTOCOL_ID,

            "training_performed":
                False,

            "instances":
                INSTANCES,

            "validation_er":
                chosen_er,

            "sampling_er":
                SAMPLING_ER,

            "er_candidates":
                ER_CANDIDATES,

            "shot_budgets":
                SHOT_BUDGETS,

            "stage5_source":
                str(
                    STAGE5_SELECTED
                ),

            "quantum_loader":
                "Walsh Series Loader",

            "classical_postprocessing":
                (
                    "exact Stage-05 "
                    "forward/backward local search"
                ),

            "weighted_gset":
                True,

            "signed_weights_supported":
                True,

            "sampling_er_rationale":
                (
                    "Finite-shot operating point selected "
                    "after exact epsilon diagnostic. "
                    "At er=0.20, post-selection probability "
                    "is approximately 1%, while the observed "
                    "maximum deviations from the frozen "
                    "Stage-05 results are below 3.3e-4 "
                    "in raw AR and 6.0e-4 in local AR."
                ),

            "noise_model":
                (
                    "stochastic Pauli trajectories "
                    "plus independent readout bit flips"
                ),
        },
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("STAGE 06 FINISHED")
    print("=" * 90)

    print()
    print(
        "STRICT STATEVECTOR VALIDATION"
    )

    print(
        statevector_df[
            [
                "instance",
                "er",
                "success_probability",
                "state_fidelity_target",
                "raw_AR",
                "local_AR",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "FINITE-SHOT SAMPLING REFERENCE"
    )

    print(
        sampling_reference_df[
            [
                "instance",
                "er",
                "success_probability",
                "state_fidelity_target",
                "raw_AR",
                "local_AR",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Results: {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
