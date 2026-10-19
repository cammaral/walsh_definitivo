#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Funções compartilhadas pelos scripts da Etapa 1.

Este módulo não reimplementa o método Walsh/PCE. Ele importa do arquivo
``experiment_1_full_walsh_random.py`` as rotinas de geração/validação dos
grafos, MQLib, máscaras, FWHT, inicialização, treinamento e salvamento.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

import artigo.results.stage1_config as cfg

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

# Os novos scripts devem ficar na raiz do projeto, ao lado do experimento base.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import experiment_1_full_walsh_random as base
except ImportError as exc:
    raise SystemExit(
        "Não encontrei experiment_1_full_walsh_random.py. Coloque os arquivos "
        "deste pacote na raiz do projeto, ao lado desse script original."
    ) from exc


def setup_reproducibility() -> None:
    """Usa exatamente as seeds globais do experimento original."""
    torch.set_num_threads(base.TORCH_NUM_THREADS)
    random.seed(base.MASTER_SEED)
    np.random.seed(base.MASTER_SEED)
    torch.manual_seed(base.MASTER_SEED)


def resolve_output_root(value: str | Path | None) -> Path:
    if value is None:
        path = PROJECT_DIR / cfg.OUTPUT_ROOT
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
    return path.resolve()


def unique_sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})


def encoding_choices(n_values: Iterable[int]) -> list[Any]:
    """Codificação de menor número de qubits, igual ao experimento original."""
    return [
        base.choose_minimum_qubit_encoding(int(n))
        for n in unique_sorted_ints(n_values)
    ]


def encoding_frame(choices: Iterable[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for choice in choices:
        rows.append(
            {
                "n_vertices": int(choice.n_vertices),
                "q_qubits": int(choice.q_qubits),
                "k_order": int(choice.k_order),
                "capacity": int(choice.capacity),
                "unused_capacity": int(choice.capacity - choice.n_vertices),
                "full_basis_size": int(choice.full_basis_size),
            }
        )
    return pd.DataFrame(rows).sort_values("n_vertices").reset_index(drop=True)


def choices_by_k(choices: Iterable[Any]) -> dict[int, list[Any]]:
    groups: dict[int, list[Any]] = {}
    for choice in choices:
        groups.setdefault(int(choice.k_order), []).append(choice)
    for group in groups.values():
        group.sort(key=lambda item: int(item.n_vertices))
    return dict(sorted(groups.items()))


def alpha_grid_for_group(group: list[Any]) -> pd.DataFrame:
    """Grade de alpha absoluto compartilhada por todos os tamanhos do mesmo k."""
    if not group:
        raise ValueError("Grupo de codificações vazio.")
    k = int(group[0].k_order)
    if any(int(choice.k_order) != k for choice in group):
        raise ValueError("Todas as codificações do grupo devem ter o mesmo k.")

    q_reference = max(int(choice.q_qubits) for choice in group)
    rows = []
    for multiplier in cfg.ALPHA_Q_MULTIPLIERS:
        rows.append(
            {
                "k_order": k,
                "q_reference": q_reference,
                "alpha_multiplier": float(multiplier),
                "alpha": float(multiplier) * float(q_reference),
            }
        )
    return pd.DataFrame(rows)


def full_hyperparameter_grid(choices: Iterable[Any]) -> pd.DataFrame:
    frames = []
    for k, group in choices_by_k(choices).items():
        alpha_frame = alpha_grid_for_group(group)
        for beta in cfg.BETA_VALUES:
            part = alpha_frame.copy()
            part["beta"] = float(beta)
            frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["k_order", "alpha", "beta"]).reset_index(drop=True)


def runtime_config(
    *,
    output_dir: Path,
    n_values: Iterable[int],
    n_graphs: int,
    training_seeds: Iterable[int],
    steps: int,
    learning_rate: float,
    mqlib_runtime_seconds: float,
) -> Any:
    return base.RuntimeConfig(
        project_dir=PROJECT_DIR,
        output_dir=output_dir,
        n_values=unique_sorted_ints(n_values),
        n_graphs=int(n_graphs),
        training_seeds=[int(seed) for seed in training_seeds],
        steps=int(steps),
        learning_rate=float(learning_rate),
        eval_every=int(cfg.EVAL_EVERY),
        mqlib_runtime_seconds=float(mqlib_runtime_seconds),
        device=base.DEVICE_NAME,
        dtype_name=base.DTYPE_NAME,
    )


def graph_bank_dir(output_root: Path) -> Path:
    return output_root / "01_graph_bank"


def calibration_dir(output_root: Path) -> Path:
    return output_root / "02_calibration_100_epochs"


def final_dir(output_root: Path) -> Path:
    return output_root / "03_final_500_epochs"


def load_graph_metadata(
    output_root: Path,
    n_vertices: int,
    expected_graphs: int | None = None,
) -> pd.DataFrame:
    bank = graph_bank_dir(output_root)
    meta_dir = bank / "graph_metadata"
    pattern = f"random_n{int(n_vertices)}_g*.json"
    rows: list[dict[str, Any]] = []
    for path in sorted(meta_dir.glob(pattern)):
        try:
            row = base.read_json(path)
            graph_file = Path(str(row["graph_file"]))
            if graph_file.is_file():
                rows.append(row)
        except Exception as error:
            print(f"[aviso] metadado inválido {path}: {error}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("graph_index").reset_index(drop=True)

    if expected_graphs is not None and len(frame) < int(expected_graphs):
        raise RuntimeError(
            f"Banco incompleto para n={n_vertices}: "
            f"{len(frame)}/{expected_graphs} grafos. Rode 01_prepare_graphs.py."
        )
    if expected_graphs is not None:
        frame = frame.iloc[: int(expected_graphs)].copy()
    return frame


def deterministic_calibration_selection(
    graph_frame: pd.DataFrame,
    n_select: int,
) -> pd.DataFrame:
    """Seleciona índices uniformemente espalhados no banco validado."""
    if graph_frame.empty:
        raise ValueError("Banco de grafos vazio.")
    n_select = min(int(n_select), len(graph_frame))
    positions = np.linspace(0, len(graph_frame) - 1, n_select, dtype=int)
    positions = np.unique(positions)
    selected = graph_frame.iloc[positions].copy()
    selected["selection_position"] = np.arange(len(selected), dtype=int)
    return selected.reset_index(drop=True)


def ensure_decision_masks(choice: Any, directory: Path) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"decision_masks_n{choice.n_vertices}_q{choice.q_qubits}_"
        f"k{choice.k_order}.npy"
    )
    if path.is_file():
        masks = np.load(path)
    else:
        masks = base.choose_decision_masks(choice)
        np.save(path, masks)
    return np.asarray(masks, dtype=np.int64)


def save_pipeline_config(
    output_root: Path,
    *,
    n_values: Iterable[int],
    n_graphs: int,
    calibration_graphs: int,
    training_seeds: Iterable[int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    choices = encoding_choices(n_values)
    payload = {
        "n_values": unique_sorted_ints(n_values),
        "n_graphs_per_size": int(n_graphs),
        "calibration_graphs_per_size": int(calibration_graphs),
        "training_seeds": [int(seed) for seed in training_seeds],
        "calibration_steps": int(cfg.CALIBRATION_STEPS),
        "final_steps": int(cfg.FINAL_STEPS),
        "learning_rate": float(cfg.LEARNING_RATE),
        "eval_every": int(cfg.EVAL_EVERY),
        "alpha_q_multipliers": [float(v) for v in cfg.ALPHA_Q_MULTIPLIERS],
        "beta_values": [float(v) for v in cfg.BETA_VALUES],
        "mqlib_runtime_seconds": float(cfg.MQLIB_RUNTIME_SECONDS),
        "keep_final_arrays": cfg.KEEP_FINAL_ARRAYS,
        "encodings": encoding_frame(choices).to_dict(orient="records"),
        "hyperparameter_grid": full_hyperparameter_grid(choices).to_dict(
            orient="records"
        ),
    }
    base.write_json_atomic(output_root / "stage1_config_resolved.json", payload)
    base.write_csv_atomic(output_root / "encoding_plan.csv", encoding_frame(choices))
    base.write_csv_atomic(
        output_root / "hyperparameter_grid.csv",
        full_hyperparameter_grid(choices),
    )


def collect_json_rows(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            rows.append(base.read_json(path))
        except Exception as error:
            print(f"[aviso] resultado inválido {path}: {error}")
    return pd.DataFrame(rows)


def write_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Tabela Markdown sem depender do pacote opcional ``tabulate``."""
    if frame.empty:
        return "_Sem dados._"
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "_Sem colunas disponíveis._"
    view = frame[available].copy()

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def zip_selected_files(
    root: Path,
    zip_path: Path,
    *,
    exclude_directory_names: set[str] | None = None,
) -> None:
    exclude_directory_names = exclude_directory_names or set()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in exclude_directory_names for part in relative.parts):
                continue
            if path == zip_path:
                continue
            archive.write(path, arcname=str(relative))


def update_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    row = base.read_json(path)
    row.update(updates)
    base.write_json_atomic(path, row)
    return row


def remove_path(path_value: str | Path | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def expected_calibration_runs(
    choices: Iterable[Any],
    calibration_graphs_per_size: int,
    seeds: Iterable[int],
) -> int:
    grid = full_hyperparameter_grid(choices)
    total = 0
    seed_count = len(list(seeds))
    for k, group in choices_by_k(choices).items():
        pair_count = len(grid[grid["k_order"] == k])
        total += pair_count * len(group) * int(calibration_graphs_per_size) * seed_count
    return int(total)


def expected_final_runs(
    choices: Iterable[Any],
    graphs_per_size: int,
    seeds: Iterable[int],
) -> int:
    return int(len(list(choices)) * int(graphs_per_size) * len(list(seeds)))
