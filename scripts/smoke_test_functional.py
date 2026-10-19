#!/usr/bin/env python3

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from walsh_pce.protocol import PROTOCOL_ID


def load_stage(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name, result=None):
    msg = f"[PASS] {name}"
    if result is not None:
        msg += f" | {result}"
    print(msg, flush=True)


def validate(row, *columns):
    for column in columns:
        value = float(row[column])
        assert math.isfinite(value), f"Invalid {column}: {value}"

    assert row.get("protocol_id", PROTOCOL_ID) == PROTOCOL_ID


def main():

    torch.set_num_threads(1)

    with tempfile.TemporaryDirectory(
        prefix="walsh_pce_functional_"
    ) as dirname:

        tmp = Path(dirname)
        data_dir = tmp / "data" / "graphs"
        data_dir.mkdir(parents=True)

        graph_rel = "n100/g0000.txt"

        # ==================================================
        # STAGE 01A - GENERATE ONE GRAPH
        # ==================================================

        s01 = load_stage(
            "smoke_generate",
            "stages/01_random_graphs/generate_graphs.py"
        )

        s01.SIZES = [100]
        s01.GRAPHS_PER_SIZE = 1
        s01.OUTPUT_ROOT = data_dir
        s01.SUMMARY_FILE = data_dir / "graphs.csv"

        s01.generate_bank()

        assert (data_dir / graph_rel).is_file()

        check("Stage 01A - Graph generation")

        # ==================================================
        # STAGE 01B - MQLIB (ONE RUN, 2 SECONDS)
        # ==================================================

        s01m = load_stage(
            "smoke_mqlib",
            "stages/01_random_graphs/run_mqlib.py"
        )

        assert s01m.MQLIB.is_file(), (
            f"MQLib executable missing: {s01m.MQLIB}"
        )

        s01m.ROOT = tmp
        s01m.RUNTIME = 2

        ref = s01m.run_one(
            100, "g0000", graph_rel, 1
        )

        c_ref = float(ref["cut"])

        assert math.isfinite(c_ref) and c_ref > 0

        check("Stage 01B - MQLib", f"cut={c_ref:.0f}")

        # ==================================================
        # STAGE 02A - CALIBRATION/EVALUATION SPLIT
        # ==================================================

        metadata = pd.DataFrame(
            {
                "n": n,
                "graph_id": f"g{i:04d}",
                "path": f"n{n}/g{i:04d}.txt"
            }
            for n in [100, 200, 400, 500, 800, 2000, 7000]
            for i in range(250)
        )

        # Synthetic metadata to test the split logic.
        # No additional graph files are generated.
        metadata.to_csv(
            data_dir / "graphs.csv",
            index=False
        )

        split_mod = load_stage(
            "smoke_split",
            "stages/02_calibration/create_split.py"
        )

        split_mod.GRAPHS_FILE = data_dir / "graphs.csv"
        split_mod.OUTPUT_FILE = data_dir / "split.csv"

        split_mod.main()

        split = pd.read_csv(split_mod.OUTPUT_FILE)

        counts = (
            split.groupby(["n", "split"])
            .size()
            .unstack(fill_value=0)
        )

        assert len(split) == 1750
        assert (counts["calibration"] == 50).all()
        assert (counts["evaluation"] == 200).all()

        check("Stage 02A - Calibration split")

        # ==================================================
        # STAGE 02B - ALPHA CALIBRATION
        # ==================================================

        alpha = 27.0
        beta = 0.75

        alpha_map = {(9, 4): alpha}
        beta_map = {(9, 4): beta}

        s02a = load_stage(
            "smoke_alpha",
            "stages/02_calibration/run_alpha.py"
        )

        s02a.ROOT = tmp
        s02a.EPOCHS = 2

        result_alpha = s02a.run_one(
            100, "g0000", graph_rel,
            c_ref, 11, int(alpha)
        )

        validate(
            result_alpha,
            "cut", "AR", "final_loss", "final_norm"
        )

        check(
            "Stage 02B - Alpha optimization",
            f"cut={result_alpha['cut']:.0f}"
        )

        # ==================================================
        # STAGE 02C - BETA CALIBRATION
        # ==================================================

        s02b = load_stage(
            "smoke_beta",
            "stages/02_calibration/run_beta.py"
        )

        s02b.ROOT = tmp
        s02b.EPOCHS = 2

        result_beta = s02b.run_one(
            100, "g0000", graph_rel,
            c_ref, 11, beta, alpha_map
        )

        validate(
            result_beta,
            "cut", "AR", "final_loss", "final_norm"
        )

        check(
            "Stage 02C - Beta optimization",
            f"cut={result_beta['cut']:.0f}"
        )

        # ==================================================
        # STAGE 03 - COMPLETE WALSH BASIS
        # ==================================================

        s03 = load_stage(
            "smoke_full",
            "stages/03_complete_basis/run_complete_basis.py"
        )

        s03.ROOT = tmp
        s03.EPOCHS = 2

        full = s03.run_one(
            100, "g0000", graph_rel,
            c_ref, 11, alpha_map, beta_map
        )

        validate(
            full,
            "cut", "AR", "final_loss", "final_norm"
        )

        check(
            "Stage 03 - Full basis optimization",
            f"cut={full['cut']:.0f}"
        )

        # ==================================================
        # STAGE 04A - SUPPORT GRID
        # ==================================================

        grid_mod = load_stage(
            "smoke_grid",
            "stages/04_support_sweep/prepare_support_grid.py"
        )

        grid_mod.SIZE_CONFIG = {100: (9, 4)}
        grid_mod.OUT_FILE = (
            tmp / "configs" / "support_grid.csv"
        )

        grid_mod.OUT_FILE.parent.mkdir(
            parents=True
        )

        grid_mod.main()

        grid = pd.read_csv(grid_mod.OUT_FILE)

        entry = grid.loc[
            grid["B"] == 1
        ].iloc[0]

        active_size = int(entry["active_size"])
        full_size = int(entry["full_size"])

        assert active_size < full_size

        check(
            "Stage 04A - Support construction",
            f"B=1, active={active_size}/{full_size}"
        )

        # ==================================================
        # STAGE 04B - SPARSE OPTIMIZATION
        # ==================================================

        s04 = load_stage(
            "smoke_sparse",
            "stages/04_support_sweep/run_support_sweep.py"
        )

        s04.ROOT = tmp
        s04.EPOCHS = 2

        sparse = s04.run_one(
            100,
            "g0000",
            graph_rel,
            c_ref,
            11,
            1,
            active_size,
            full_size,
            float(entry["active_fraction"]),
            float(full["cut"]),
            float(full["AR"]),
            int(full["actual_seed"]),
            alpha_map,
            beta_map
        )

        validate(
            sparse,
            "cut", "AR", "final_loss",
            "final_norm", "recovery"
        )

        check(
            "Stage 04B - Sparse B=1 optimization",
            f"cut={sparse['cut']:.0f}"
        )

        # ==================================================
        # STAGE 05 - GSET, BITFLIP AND COEFFICIENTS
        # ==================================================

        s05 = load_stage(
            "smoke_gset",
            "stages/05_gset/run_gset.py"
        )

        # Reads original Gset inputs, without editing them.
        instances = s05.prepare_instances()

        # All coefficient files go to the temporary directory.
        s05.ROOT = tmp
        s05.ARRAY_DIR = (
            tmp / "stage_05" / "coefficients"
        )

        s05.ALPHA_BY_QK = {(12, 6): 36.0}
        s05.BETA_BY_QK = {(12, 6): 0.75}
        s05.EPOCHS = 2

        gset = s05.train_one(
            "G14",
            11,
            instances["G14"]
        )

        validate(
            gset,
            "raw_cut", "raw_AR",
            "local_cut", "local_AR",
            "final_loss", "final_norm"
        )

        assert gset["local_cut"] >= gset["raw_cut"]

        saved = s05.validate_npz_protocol(
            gset["npz"]
        )

        with np.load(
            saved,
            allow_pickle=False
        ) as archive:

            assert (
                len(archive["coefficients"])
                == gset["active_size"]
            )

        check(
            "Stage 05 - Gset + bitflip + NPZ",
            f"raw={gset['raw_cut']:.0f}, "
            f"refined={gset['local_cut']:.0f}"
        )

        # ==================================================
        # FINAL SUMMARY
        # ==================================================

        print()
        print("=" * 60)
        print("FUNCTIONAL SMOKE TEST SUMMARY")
        print("=" * 60)
        print("STAGES 01-05: PASS")
        print("Original experimental results: UNCHANGED")
        print("Temporary test outputs: REMOVED")
        print("=" * 60)


if __name__ == "__main__":
    main()
