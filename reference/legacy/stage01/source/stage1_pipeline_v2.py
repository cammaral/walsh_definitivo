#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline seguro da Etapa 1 Walsh/PCE.

O método Walsh/PCE, a FWHT, a inicialização, o treinamento e a interface do
MQLib são importados de ``experiment_1_full_walsh_random.py``. Este arquivo
organiza as três fases, corrige a geração de grafos grandes e implementa uma
retomada que nunca depende de caminhos absolutos antigos.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch

import artigo.results.stage1_config_v2 as cfg

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# CONTEXTO E UTILITÁRIOS
# =============================================================================

@dataclass(frozen=True)
class Context:
    project_root: Path
    output_root: Path
    base: ModuleType
    n_values: tuple[int, ...]
    n_graphs: int
    calibration_graphs: int
    training_seeds: tuple[int, ...]
    mqlib_runtime: float
    calibration_steps: int
    final_steps: int
    keep_arrays: str


def _candidate_roots(explicit: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser().resolve())
    for start in [Path.cwd().resolve(), SCRIPT_DIR.resolve()]:
        candidates.append(start)
        candidates.extend(start.parents)
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def find_project_root(explicit: str | Path | None = None) -> Path:
    for candidate in _candidate_roots(explicit):
        if (candidate / "experiment_1_full_walsh_random.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Não encontrei experiment_1_full_walsh_random.py. Extraia esta pasta "
        "dentro do projeto ou use --project-root /caminho/do/projeto."
    )


def import_base(project_root: Path) -> ModuleType:
    module_path = project_root / "experiment_1_full_walsh_random.py"
    spec = importlib.util.spec_from_file_location("walsh_stage1_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_output_root(project_root: Path, value: str | Path | None) -> Path:
    path = Path(value).expanduser() if value is not None else cfg.OUTPUT_ROOT
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if path in {project_root, project_root.parent, Path("/")}:
        raise ValueError(f"output-root inseguro: {path}")
    return path


def unique_ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def build_context(args: argparse.Namespace) -> Context:
    project_root = find_project_root(args.project_root)
    output_root = resolve_output_root(project_root, args.output_root)
    base = import_base(project_root)

    torch.set_num_threads(base.TORCH_NUM_THREADS)
    random.seed(cfg.MASTER_SEED)
    np.random.seed(cfg.MASTER_SEED)
    torch.manual_seed(cfg.MASTER_SEED)

    keep_arrays = str(getattr(args, "keep_arrays", cfg.KEEP_FINAL_ARRAYS))
    if keep_arrays not in {"all", "best", "none"}:
        raise ValueError("keep-arrays deve ser all, best ou none")

    return Context(
        project_root=project_root,
        output_root=output_root,
        base=base,
        n_values=unique_ints(args.n_values),
        n_graphs=int(args.graphs),
        calibration_graphs=int(args.calibration_graphs),
        training_seeds=tuple(int(seed) for seed in args.training_seeds),
        mqlib_runtime=float(args.mqlib_runtime),
        calibration_steps=int(args.calibration_steps),
        final_steps=int(args.final_steps),
        keep_arrays=keep_arrays,
    )


def choices(context: Context) -> list[Any]:
    return [context.base.choose_minimum_qubit_encoding(n) for n in context.n_values]


def choice_table(context: Context) -> pd.DataFrame:
    rows = []
    for choice in choices(context):
        rows.append(
            {
                "n_vertices": int(choice.n_vertices),
                "q_qubits": int(choice.q_qubits),
                "k_order": int(choice.k_order),
                "capacity": int(choice.capacity),
                "full_basis_size": int(choice.full_basis_size),
            }
        )
    return pd.DataFrame(rows)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def safe_float_label(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def collect_json_rows(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        if path.name.startswith("manifest") or path.name.startswith("config"):
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as error:
            raise RuntimeError(f"JSON inválido em {path}: {error}") from error
    return pd.DataFrame(rows)


def update_json(base: ModuleType, path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    row = base.read_json(path)
    row.update(updates)
    base.write_json_atomic(path, row)
    return row


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_Sem resultados._"
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_pipeline_config(context: Context) -> None:
    context.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline_version": cfg.PIPELINE_VERSION,
        "project_root": str(context.project_root),
        "output_root": str(context.output_root),
        "n_values": list(context.n_values),
        "graphs_per_size": context.n_graphs,
        "calibration_graphs_per_size": context.calibration_graphs,
        "training_seeds": list(context.training_seeds),
        "calibration_steps": context.calibration_steps,
        "final_steps": context.final_steps,
        "learning_rate": cfg.LEARNING_RATE,
        "eval_every": cfg.EVAL_EVERY,
        "alpha_q_multipliers": cfg.ALPHA_Q_MULTIPLIERS,
        "beta_values": cfg.BETA_VALUES,
        "mqlib_heuristics": cfg.MQLIB_HEURISTICS,
        "mqlib_seeds": cfg.MQLIB_SEEDS,
        "mqlib_runtime_seconds": context.mqlib_runtime,
        "target_average_degree": cfg.TARGET_AVERAGE_DEGREE,
        "connectivity_mode": cfg.CONNECTIVITY_MODE,
        "encodings": choice_table(context).to_dict(orient="records"),
    }
    context.base.write_json_atomic(context.output_root / "pipeline_config.json", payload)


# =============================================================================
# CAMINHOS E BANCO DE GRAFOS
# =============================================================================

def bank_dir(context: Context) -> Path:
    return context.output_root / "01_graph_bank"


def calibration_dir(context: Context) -> Path:
    return context.output_root / f"02_calibration_{context.calibration_steps}_epochs"


def final_dir(context: Context) -> Path:
    return context.output_root / f"03_final_{context.final_steps}_epochs"


def graph_dir(context: Context, n: int) -> Path:
    return bank_dir(context) / "graphs" / f"n{int(n)}"


def meta_dir(context: Context, n: int) -> Path:
    return bank_dir(context) / "metadata" / f"n{int(n)}"


def graph_path(context: Context, n: int, index: int) -> Path:
    return graph_dir(context, n) / f"random_n{int(n)}_g{int(index):04d}.txt"


def meta_path(context: Context, n: int, index: int) -> Path:
    return meta_dir(context, n) / f"random_n{int(n)}_g{int(index):04d}.json"


def graph_bank_signature(context: Context) -> dict[str, Any]:
    return {
        "graph_bank_version": cfg.GRAPH_BANK_VERSION,
        "n_values": list(context.n_values),
        "graphs_per_size": context.n_graphs,
        "target_average_degree": cfg.TARGET_AVERAGE_DEGREE,
        "minimum_average_degree": cfg.MIN_AVERAGE_DEGREE,
        "connectivity_mode": cfg.CONNECTIVITY_MODE,
        "random_cut_ar_max": cfg.RANDOM_CUT_AR_MAX,
        "mqlib_heuristics": cfg.MQLIB_HEURISTICS,
        "mqlib_seeds": cfg.MQLIB_SEEDS,
        "mqlib_runtime_seconds": context.mqlib_runtime,
        "master_seed": cfg.MASTER_SEED,
    }


def ensure_bank_manifest(context: Context) -> None:
    directory = bank_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bank_manifest.json"
    signature = graph_bank_signature(context)
    expected = fingerprint(signature)
    if path.is_file():
        current = context.base.read_json(path)
        if current.get("fingerprint") != expected:
            raise RuntimeError(
                "O banco existente foi criado com outra configuração. Para impedir "
                "sobrescrita acidental, use outro --output-root ou restaure a "
                f"configuração original. Banco: {directory}"
            )
        return
    context.base.write_json_atomic(
        path,
        {"fingerprint": expected, "signature": signature, "created_utc": context.base.utc_now()},
    )


def resolve_graph_from_metadata(context: Context, row: dict[str, Any]) -> Path:
    relative = row.get("graph_relpath")
    if relative:
        path = (context.output_root / str(relative)).resolve()
        if path.is_file():
            return path
    expected = graph_path(context, int(row["n_vertices"]), int(row["graph_index"]))
    if expected.is_file():
        return expected.resolve()
    absolute = row.get("graph_file")
    if absolute and Path(str(absolute)).is_file():
        return Path(str(absolute)).resolve()
    raise FileNotFoundError(
        f"Arquivo do grafo ausente para {row.get('graph_id')}. Esperado: {expected}"
    )


def load_graph_metadata(context: Context, n: int, expected: int | None = None) -> pd.DataFrame:
    directory = meta_dir(context, n)
    directory.mkdir(parents=True, exist_ok=True)
    graph_dir(context, n).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"random_n{int(n)}_g*.json")):
        row = context.base.read_json(path)
        resolved = resolve_graph_from_metadata(context, row)
        correct_relative = str(resolved.relative_to(context.output_root))
        if row.get("graph_relpath") != correct_relative or row.get("graph_file") != str(resolved):
            row["graph_relpath"] = correct_relative
            row["graph_file"] = str(resolved)
            context.base.write_json_atomic(path, row)
        rows.append(row)

    rows.sort(key=lambda item: int(item["graph_index"]))
    for expected_index, row in enumerate(rows):
        if int(row["graph_index"]) != expected_index:
            raise RuntimeError(
                f"Banco n={n} possui lacuna no índice {expected_index}. "
                "O código não sobrescreverá arquivos automaticamente."
            )

    # Arquivo de grafo sem metadado é tratado como inconsistência, não como
    # autorização para começar do zero.
    graph_files = sorted(graph_dir(context, n).glob(f"random_n{int(n)}_g*.txt"))
    if len(graph_files) != len(rows):
        raise RuntimeError(
            f"Banco inconsistente para n={n}: {len(graph_files)} arquivos de grafo "
            f"e {len(rows)} metadados. Corrija o banco antes de continuar."
        )

    frame = pd.DataFrame(rows)
    if expected is not None and len(frame) < int(expected):
        raise RuntimeError(
            f"Banco incompleto para n={n}: {len(frame)}/{expected}. "
            "Rode 01_prepare_graphs_v2.py."
        )
    if expected is not None:
        frame = frame.iloc[: int(expected)].copy()
    return frame


def write_all_graphs_csv(context: Context) -> pd.DataFrame:
    frames = []
    for n in context.n_values:
        frame = load_graph_metadata(context, n)
        if not frame.empty:
            frames.append(frame)
    all_graphs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_graphs.empty:
        context.base.write_csv_atomic(bank_dir(context) / "graphs.csv", all_graphs)
    return all_graphs


def connect_components_minimally(base: ModuleType, graph: nx.Graph, n: int, attempt: int) -> tuple[int, list[tuple[int, int]]]:
    components = [sorted(component) for component in nx.connected_components(graph)]
    count = len(components)
    if count <= 1:
        return count, []
    rng = np.random.default_rng(base.stable_seed(cfg.MASTER_SEED, "connect_components", n, attempt))
    order = np.arange(count)
    rng.shuffle(order)
    bridges: list[tuple[int, int]] = []
    for left_index, right_index in zip(order[:-1], order[1:]):
        left = components[int(left_index)]
        right = components[int(right_index)]
        u = int(left[int(rng.integers(0, len(left)))])
        v = int(right[int(rng.integers(0, len(right)))])
        graph.add_edge(u, v, weight=1.0)
        bridges.append((u, v))
    if not nx.is_connected(graph):
        raise RuntimeError("Falha interna ao conectar componentes.")
    return count, bridges


def prepare_graphs(context: Context) -> None:
    write_pipeline_config(context)
    ensure_bank_manifest(context)
    base = context.base
    executable, mqlib_root = base.find_or_build_mqlib(context.project_root)
    bank = bank_dir(context)
    tmp_dir = bank / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mqlib_csv = bank / "mqlib_runs.csv"
    if mqlib_csv.is_file():
        mqlib_rows = pd.read_csv(mqlib_csv).to_dict(orient="records")
    else:
        mqlib_rows = []

    with base.ExperimentLock(bank / ".prepare.lock"):
        for choice in choices(context):
            n = int(choice.n_vertices)
            graph_dir(context, n).mkdir(parents=True, exist_ok=True)
            meta_dir(context, n).mkdir(parents=True, exist_ok=True)
            existing = load_graph_metadata(context, n)
            accepted = len(existing)
            if accepted >= context.n_graphs:
                print(f"n={n}: {context.n_graphs}/{context.n_graphs} grafos já válidos; reutilizando.")
                continue

            state_path = bank / f"generation_state_n{n}.json"
            if state_path.is_file():
                state = base.read_json(state_path)
                attempt = int(state.get("last_attempt", 0))
            else:
                attempt = 0

            probability = min(1.0, cfg.TARGET_AVERAGE_DEGREE / max(1, n - 1))
            print("\n" + "=" * 94)
            print(
                f"FASE 1 — n={n}, q={choice.q_qubits}, k={choice.k_order}, "
                f"2^q={choice.full_basis_size}; retomando {accepted}/{context.n_graphs}"
            )
            print("=" * 94)

            while accepted < context.n_graphs:
                attempt += 1
                if attempt > cfg.MAX_GRAPH_ATTEMPTS_PER_SIZE:
                    raise RuntimeError(
                        f"Limite de tentativas atingido em n={n}: {accepted}/{context.n_graphs}."
                    )
                graph_seed = base.stable_seed(cfg.MASTER_SEED, "random_graph_v2", n, attempt)
                graph = nx.fast_gnp_random_graph(n, probability, seed=graph_seed, directed=False)
                nx.set_edge_attributes(graph, 1.0, "weight")
                if graph.number_of_edges() == 0:
                    continue

                average_degree_before = 2.0 * graph.number_of_edges() / n
                if average_degree_before < cfg.MIN_AVERAGE_DEGREE:
                    continue
                components_before, bridges = connect_components_minimally(base, graph, n, attempt)
                average_degree_after = 2.0 * graph.number_of_edges() / n

                candidate = tmp_dir / f"candidate_n{n}_a{attempt:07d}.txt"
                candidate.unlink(missing_ok=True)
                base.save_graph_mqlib(graph, candidate)

                best_reference = -math.inf
                reference_source = ""
                candidate_mqlib_rows: list[dict[str, Any]] = []
                for heuristic in cfg.MQLIB_HEURISTICS:
                    for mqlib_seed in cfg.MQLIB_SEEDS:
                        result = base.run_mqlib(
                            executable=executable,
                            graph_file=candidate,
                            heuristic=heuristic,
                            seed=mqlib_seed,
                            runtime_seconds=context.mqlib_runtime,
                        )
                        result.update(
                            {
                                "candidate_n": n,
                                "candidate_attempt": attempt,
                                "graph_seed": graph_seed,
                                "mqlib_root": str(mqlib_root),
                            }
                        )
                        candidate_mqlib_rows.append(result)
                        if result.get("status") == "ok":
                            objective = float(result["reported_objective"])
                            if objective > best_reference:
                                best_reference = objective
                                reference_source = f"MQLib:{heuristic}:seed{mqlib_seed}"

                mqlib_rows.extend(candidate_mqlib_rows)
                base.write_csv_atomic(mqlib_csv, pd.DataFrame(mqlib_rows))
                if not math.isfinite(best_reference) or best_reference <= 0:
                    candidate.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Todas as execuções MQLib falharam para n={n}, tentativa={attempt}."
                    )

                rng = np.random.default_rng(base.stable_seed(cfg.MASTER_SEED, "random_filter", n, attempt))
                random_bits = rng.integers(0, 2, size=n, dtype=np.uint8)
                random_cut = base.cut_value(random_bits, graph)
                random_ar = float(random_cut / best_reference)
                diagnostic_cuts = [
                    base.cut_value(rng.integers(0, 2, size=n, dtype=np.uint8), graph)
                    for _ in range(cfg.RANDOM_CUT_DIAGNOSTIC_SAMPLES)
                ]
                if random_ar > cfg.RANDOM_CUT_AR_MAX:
                    candidate.unlink(missing_ok=True)
                    continue

                index = accepted
                final_graph = graph_path(context, n, index)
                final_meta = meta_path(context, n, index)
                if final_graph.exists() or final_meta.exists():
                    raise RuntimeError(
                        f"Recusa de sobrescrita: {final_graph} ou {final_meta} já existe."
                    )
                final_graph.parent.mkdir(parents=True, exist_ok=True)
                final_meta.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, final_graph)
                graph_id = f"random_n{n}_g{index:04d}"
                row = {
                    "created_utc": base.utc_now(),
                    "graph_bank_version": cfg.GRAPH_BANK_VERSION,
                    "graph_id": graph_id,
                    "graph_index": index,
                    "n_vertices": n,
                    "q_qubits": int(choice.q_qubits),
                    "k_order": int(choice.k_order),
                    "capacity": int(choice.capacity),
                    "full_basis_size": int(choice.full_basis_size),
                    "attempt": attempt,
                    "graph_seed": int(graph_seed),
                    "n_edges": int(graph.number_of_edges()),
                    "average_degree_before_repair": float(average_degree_before),
                    "average_degree": float(average_degree_after),
                    "components_before_repair": int(components_before),
                    "bridges_added": int(len(bridges)),
                    "connectivity_repaired": bool(len(bridges) > 0),
                    "connected": True,
                    "initial_reference_cut": float(best_reference),
                    "reference_source": reference_source,
                    "mqlib_runtime_per_run_s": float(context.mqlib_runtime),
                    "mqlib_heuristics": list(cfg.MQLIB_HEURISTICS),
                    "mqlib_seeds": list(cfg.MQLIB_SEEDS),
                    "random_filter_cut": float(random_cut),
                    "random_filter_ar": float(random_ar),
                    "random_cut_mean": float(np.mean(diagnostic_cuts)),
                    "random_cut_std": float(np.std(diagnostic_cuts)),
                    "random_cut_best": float(np.max(diagnostic_cuts)),
                    "graph_relpath": str(final_graph.relative_to(context.output_root)),
                    "graph_file": str(final_graph.resolve()),
                    "bridges": bridges,
                }
                base.write_json_atomic(final_meta, row)
                accepted += 1
                base.write_json_atomic(
                    state_path,
                    {
                        "n_vertices": n,
                        "last_attempt": attempt,
                        "accepted": accepted,
                        "updated_utc": base.utc_now(),
                    },
                )
                write_all_graphs_csv(context)
                print(
                    f"  aceito {accepted}/{context.n_graphs}: {graph_id}, "
                    f"ref={best_reference:.3f}, random AR={random_ar:.3f}, "
                    f"componentes={components_before}, pontes={len(bridges)}"
                )

    all_graphs = write_all_graphs_csv(context)
    base.write_json_atomic(
        bank / "manifest_completed.json",
        {
            "completed_utc": base.utc_now(),
            "n_graphs_total": int(len(all_graphs)),
            "expected": int(len(context.n_values) * context.n_graphs),
            "mqlib_executable": str(executable),
        },
    )
    print(f"\nBanco concluído em: {bank}")


# =============================================================================
# HIPERPARÂMETROS E TREINAMENTO
# =============================================================================

def choices_by_k(context: Context) -> dict[int, list[Any]]:
    groups: dict[int, list[Any]] = {}
    for choice in choices(context):
        groups.setdefault(int(choice.k_order), []).append(choice)
    for group in groups.values():
        group.sort(key=lambda item: int(item.n_vertices))
    return dict(sorted(groups.items()))


def hyperparameter_grid(context: Context) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, group in choices_by_k(context).items():
        q_reference = max(int(choice.q_qubits) for choice in group)
        for multiplier in cfg.ALPHA_Q_MULTIPLIERS:
            alpha = float(multiplier) * float(q_reference)
            for beta in cfg.BETA_VALUES:
                rows.append(
                    {
                        "k_order": k,
                        "q_reference": q_reference,
                        "alpha_multiplier": float(multiplier),
                        "alpha": alpha,
                        "beta": float(beta),
                    }
                )
    return pd.DataFrame(rows).sort_values(["k_order", "alpha", "beta"]).reset_index(drop=True)


def deterministic_calibration_selection(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    count = min(int(count), len(frame))
    positions = np.unique(np.linspace(0, len(frame) - 1, count, dtype=int))
    selected = frame.iloc[positions].copy().reset_index(drop=True)
    selected["selection_position"] = np.arange(len(selected), dtype=int)
    return selected


def calibration_selection(context: Context) -> pd.DataFrame:
    directory = calibration_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "calibration_selection.csv"
    if path.is_file():
        selection = pd.read_csv(path)
        expected_rows = len(context.n_values) * min(context.calibration_graphs, context.n_graphs)
        if len(selection) != expected_rows:
            raise RuntimeError(
                f"Seleção de calibração existente possui {len(selection)} linhas; "
                f"esperado {expected_rows}. Use outro output-root para outro protocolo."
            )
        return selection

    frames = []
    for n in context.n_values:
        bank = load_graph_metadata(context, n, expected=context.n_graphs)
        selected = deterministic_calibration_selection(bank, context.calibration_graphs)
        frames.append(selected)
    selection = pd.concat(frames, ignore_index=True)
    context.base.write_csv_atomic(path, selection)
    return selection


def ensure_decision_masks(context: Context, choice: Any, directory: Path) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"decision_masks_n{choice.n_vertices}_q{choice.q_qubits}_k{choice.k_order}.npy"
    )
    if path.is_file():
        masks = np.load(path)
    else:
        masks = context.base.choose_decision_masks(choice)
        np.save(path, masks)
    return np.asarray(masks, dtype=np.int64)


def runtime_config(context: Context, phase_dir: Path, choice: Any, steps: int) -> Any:
    return context.base.RuntimeConfig(
        project_dir=context.project_root,
        output_dir=phase_dir,
        n_values=[int(choice.n_vertices)],
        n_graphs=context.n_graphs,
        training_seeds=list(context.training_seeds),
        steps=int(steps),
        learning_rate=float(cfg.LEARNING_RATE),
        eval_every=int(cfg.EVAL_EVERY),
        mqlib_runtime_seconds=float(context.mqlib_runtime),
        device=context.base.DEVICE_NAME,
        dtype_name=context.base.DTYPE_NAME,
    )


def graph_from_row(context: Context, row: Any) -> nx.Graph:
    data = row._asdict() if hasattr(row, "_asdict") else dict(row)
    path = resolve_graph_from_metadata(context, data)
    return context.base.load_graph_mqlib(path)


def calibration_ranking(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    keys = ["k_order", "alpha", "beta"]
    per_n = (
        runs.groupby(keys + ["n_vertices"], as_index=False)
        .agg(
            mean_ar_n=("final_raw_ar_initial_reference", "mean"),
            std_ar_n=("final_raw_ar_initial_reference", "std"),
        )
    )
    macro = per_n.groupby(keys, as_index=False).agg(
        macro_mean_ar=("mean_ar_n", "mean"),
        macro_std_across_n=("mean_ar_n", "std"),
    )
    pooled = runs.groupby(keys, as_index=False).agg(
        n_runs=("graph_id", "size"),
        n_graphs=("graph_id", "nunique"),
        pooled_mean_ar=("final_raw_ar_initial_reference", "mean"),
        pooled_std_ar=("final_raw_ar_initial_reference", "std"),
        pooled_min_ar=("final_raw_ar_initial_reference", "min"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    p10 = (
        runs.groupby(keys)["final_raw_ar_initial_reference"]
        .quantile(0.10)
        .rename("p10_ar")
        .reset_index()
    )
    best_seed = (
        runs.sort_values(
            ["graph_id", "k_order", "alpha", "beta", "final_raw_cut", "runtime_s"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(["graph_id"] + keys, as_index=False)
        .first()
    )
    best_summary = best_seed.groupby(keys, as_index=False).agg(
        mean_best_of_10_ar=("final_raw_ar_initial_reference", "mean"),
        min_best_of_10_ar=("final_raw_ar_initial_reference", "min"),
    )
    ranking = macro.merge(pooled, on=keys).merge(p10, on=keys).merge(best_summary, on=keys)
    ranking = ranking.sort_values(
        ["k_order", "macro_mean_ar", "p10_ar", "pooled_std_ar", "mean_runtime_s"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    ranking["rank_within_k"] = ranking.groupby("k_order").cumcount() + 1
    return ranking


def write_calibration_report(context: Context, grid: pd.DataFrame, ranking: pd.DataFrame, selected: pd.DataFrame, runs: pd.DataFrame) -> None:
    directory = calibration_dir(context)
    lines = [
        "# Relatório de calibração — Etapa 1 Walsh/PCE",
        "",
        f"- Épocas por execução: {context.calibration_steps}",
        f"- Grafos por tamanho: {context.calibration_graphs}",
        f"- Seeds por configuração: {len(context.training_seeds)} ({list(context.training_seeds)})",
        f"- Execuções concluídas: {len(runs)}",
        "- Bitflip: não",
        "- Critério primário: maior média macro de AR entre os tamanhos que compartilham k.",
        "",
        "## Codificações",
        "",
        markdown_table(choice_table(context), ["n_vertices", "q_qubits", "k_order", "capacity", "full_basis_size"]),
        "",
        "## Grade",
        "",
        markdown_table(grid, ["k_order", "q_reference", "alpha_multiplier", "alpha", "beta"]),
        "",
        "## Par selecionado automaticamente por k",
        "",
        markdown_table(selected, ["k_order", "selected_alpha", "selected_beta", "macro_mean_ar", "p10_ar", "mean_best_of_10_ar"]),
    ]
    for k in sorted(ranking["k_order"].unique()):
        lines.extend(
            [
                "",
                f"## Top 5 para k={int(k)}",
                "",
                markdown_table(
                    ranking[ranking["k_order"] == k].head(5),
                    ["rank_within_k", "alpha", "beta", "macro_mean_ar", "p10_ar", "pooled_std_ar", "mean_best_of_10_ar", "mean_runtime_s"],
                ),
            ]
        )
    (directory / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")

    bundle = directory / "calibration_report_bundle.zip"
    bundle.unlink(missing_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [
            "calibration_report.md",
            "calibration_selection.csv",
            "hyperparameter_grid.csv",
            "calibration_runs.csv",
            "calibration_ranking.csv",
            "selected_hyperparameters.csv",
        ]:
            path = directory / name
            if path.is_file():
                archive.write(path, arcname=name)


def calibrate(context: Context) -> None:
    write_pipeline_config(context)
    for n in context.n_values:
        load_graph_metadata(context, n, expected=context.n_graphs)

    base = context.base
    directory = calibration_dir(context)
    run_json_dir = directory / "run_results"
    array_dir = directory / "run_arrays"
    mask_dir = directory / "decision_masks"
    for path in [directory, run_json_dir, array_dir, mask_dir]:
        path.mkdir(parents=True, exist_ok=True)

    selection = calibration_selection(context)
    grid = hyperparameter_grid(context)
    base.write_csv_atomic(directory / "hyperparameter_grid.csv", grid)
    base.SAVE_ALL_ARRAYS = False
    base.UPDATE_PLOT_AFTER_EACH_GRAPH = False
    base.CREATE_FINAL_ZIP = False

    expected = len(selection) * len(context.training_seeds) * len(cfg.ALPHA_Q_MULTIPLIERS) * len(cfg.BETA_VALUES)
    print(f"Calibração esperada: {expected} execuções de {context.calibration_steps} épocas.")
    completed_since_save = 0

    with base.ExperimentLock(directory / ".calibration.lock"):
        choice_map = {int(item.n_vertices): item for item in choices(context)}
        for grid_row in grid.itertuples(index=False):
            k = int(grid_row.k_order)
            alpha = float(grid_row.alpha)
            beta = float(grid_row.beta)
            print(f"\n--- k={k}, alpha={alpha:g}, beta={beta:g} ---")
            relevant_choices = [item for item in choices(context) if int(item.k_order) == k]
            for choice in relevant_choices:
                n = int(choice.n_vertices)
                masks = ensure_decision_masks(context, choice, mask_dir)
                runtime = runtime_config(context, directory, choice, context.calibration_steps)
                selected_n = selection[pd.to_numeric(selection["n_vertices"]).astype(int) == n]
                for graph_row in selected_n.itertuples(index=False):
                    graph_id = str(graph_row.graph_id)
                    graph = graph_from_row(context, graph_row)
                    reference = float(graph_row.initial_reference_cut)
                    hp_dir = (
                        run_json_dir
                        / f"k{k}"
                        / f"alpha_{safe_float_label(alpha)}"
                        / f"beta_{safe_float_label(beta)}"
                    )
                    hp_array_dir = (
                        array_dir
                        / f"k{k}"
                        / f"alpha_{safe_float_label(alpha)}"
                        / f"beta_{safe_float_label(beta)}"
                    )
                    hp_dir.mkdir(parents=True, exist_ok=True)
                    hp_array_dir.mkdir(parents=True, exist_ok=True)
                    for seed in context.training_seeds:
                        identifier = base.run_identifier(
                            choice=choice,
                            graph_id=graph_id,
                            init_seed=int(seed),
                            alpha=alpha,
                            beta=beta,
                            config=runtime,
                        )
                        result_json = hp_dir / f"{identifier}.json"
                        array_npz = hp_array_dir / f"{identifier}.npz"
                        row = base.train_full_basis(
                            choice=choice,
                            graph_id=graph_id,
                            graph=graph,
                            initial_reference_cut=reference,
                            decision_masks_np=masks,
                            alpha=alpha,
                            beta=beta,
                            init_seed=int(seed),
                            config=runtime,
                            result_json=result_json,
                            array_npz=array_npz,
                        )
                        updates = {
                            "phase": "calibration",
                            "alpha_multiplier": float(grid_row.alpha_multiplier),
                            "q_reference": int(grid_row.q_reference),
                        }
                        if any(row.get(key) != value for key, value in updates.items()):
                            update_json(base, result_json, updates)
                        completed_since_save += 1
                        if completed_since_save >= cfg.SAVE_PROGRESS_EVERY_RUNS:
                            runs = collect_json_rows(run_json_dir)
                            base.write_csv_atomic(directory / "calibration_runs_partial.csv", runs)
                            print(f"  progresso: {len(runs)}/{expected}")
                            completed_since_save = 0

        runs = collect_json_rows(run_json_dir)
        base.write_csv_atomic(directory / "calibration_runs.csv", runs)
        ranking = calibration_ranking(runs)
        base.write_csv_atomic(directory / "calibration_ranking.csv", ranking)
        selected = (
            ranking[ranking["rank_within_k"] == 1]
            .rename(columns={"alpha": "selected_alpha", "beta": "selected_beta"})
            .reset_index(drop=True)
        )
        base.write_csv_atomic(directory / "selected_hyperparameters.csv", selected)
        write_calibration_report(context, grid, ranking, selected, runs)
        base.write_json_atomic(
            directory / "manifest_completed.json",
            {
                "completed_utc": base.utc_now(),
                "completed_runs": int(len(runs)),
                "expected_runs": int(expected),
                "selected_hyperparameters": selected.to_dict(orient="records"),
            },
        )
    print(f"\nCalibração concluída. Envie: {directory / 'calibration_report_bundle.zip'}")


# =============================================================================
# RODADA FINAL
# =============================================================================

def load_selected_hyperparameters(context: Context, csv_path: str | Path | None) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve() if csv_path else calibration_dir(context) / "selected_hyperparameters.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Hiperparâmetros não encontrados: {path}")
    frame = pd.read_csv(path)
    required = {"k_order", "selected_alpha", "selected_beta"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV precisa conter {sorted(required)}")
    frame = frame.drop_duplicates("k_order", keep="last").copy()
    frame.attrs["source_path"] = str(path)
    return frame


def prune_arrays(base: ModuleType, graph_rows: list[tuple[Path, dict[str, Any]]], keep: str) -> int | None:
    if not graph_rows:
        return None
    best_index = max(
        range(len(graph_rows)),
        key=lambda index: (
            float(graph_rows[index][1]["final_raw_cut"]),
            float(graph_rows[index][1]["final_raw_ar_initial_reference"]),
            -float(graph_rows[index][1]["runtime_s"]),
        ),
    )
    for index, (json_path, row) in enumerate(graph_rows):
        array_path = Path(str(row["arrays_path"]))
        should_keep = keep == "all" or (keep == "best" and index == best_index)
        if should_keep:
            update_json(base, json_path, {"arrays_saved": bool(array_path.is_file() and array_path.stat().st_size > 0)})
            continue
        array_path.unlink(missing_ok=True)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        array_path.touch()
        update_json(base, json_path, {"arrays_saved": False, "arrays_placeholder": str(array_path)})
    return best_index


def final_summaries(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if runs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    all_summary = (
        runs.groupby(["n_vertices", "q_qubits", "k_order", "alpha", "beta"], as_index=False)
        .agg(
            n_graphs=("graph_id", "nunique"),
            n_runs=("graph_id", "size"),
            mean_ar_all_seeds=("final_raw_ar_initial_reference", "mean"),
            std_ar_all_seeds=("final_raw_ar_initial_reference", "std"),
            min_ar_all_seeds=("final_raw_ar_initial_reference", "min"),
            max_ar_all_seeds=("final_raw_ar_initial_reference", "max"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_min_abs_correlator=("min_abs_correlator", "mean"),
            mean_saturation_099=("saturation_fraction_099", "mean"),
        )
    )
    best = (
        runs.sort_values(["graph_id", "final_raw_cut", "runtime_s"], ascending=[True, False, True])
        .groupby("graph_id", as_index=False)
        .first()
    )
    best["walsh_beats_mqlib_reference"] = best["final_raw_cut"] > best["initial_reference_cut"] + 1e-12
    best_summary = (
        best.groupby(["n_vertices", "q_qubits", "k_order", "alpha", "beta"], as_index=False)
        .agg(
            n_graphs=("graph_id", "size"),
            mean_best_of_10_ar=("final_raw_ar_initial_reference", "mean"),
            std_best_of_10_ar=("final_raw_ar_initial_reference", "std"),
            min_best_of_10_ar=("final_raw_ar_initial_reference", "min"),
            max_best_of_10_ar=("final_raw_ar_initial_reference", "max"),
            fraction_beating_mqlib=("walsh_beats_mqlib_reference", "mean"),
            mean_best_runtime_s=("runtime_s", "mean"),
        )
    )
    return all_summary, best, best_summary


def write_final_report(context: Context, hyperparameters: pd.DataFrame, runs: pd.DataFrame, all_summary: pd.DataFrame, best_summary: pd.DataFrame) -> None:
    directory = final_dir(context)
    lines = [
        "# Relatório final — Etapa 1 Walsh/PCE",
        "",
        f"- Épocas: {context.final_steps}",
        f"- Grafos por tamanho: {context.n_graphs}",
        f"- Seeds: {len(context.training_seeds)} ({list(context.training_seeds)})",
        f"- Execuções concluídas: {len(runs)}",
        "- Bitflip: não",
        "- O denominador do AR permanece a referência fixa do MQLib salva na Fase 1.",
        "",
        "## Hiperparâmetros",
        "",
        markdown_table(hyperparameters, ["k_order", "selected_alpha", "selected_beta", "macro_mean_ar", "p10_ar"]),
        "",
        "## Todas as dez seeds",
        "",
        markdown_table(
            all_summary,
            ["n_vertices", "q_qubits", "k_order", "alpha", "beta", "mean_ar_all_seeds", "std_ar_all_seeds", "min_ar_all_seeds", "max_ar_all_seeds", "mean_runtime_s"],
        ),
        "",
        "## Melhor seed entre dez por grafo",
        "",
        markdown_table(
            best_summary,
            ["n_vertices", "q_qubits", "k_order", "alpha", "beta", "mean_best_of_10_ar", "std_best_of_10_ar", "min_best_of_10_ar", "max_best_of_10_ar", "fraction_beating_mqlib", "mean_best_runtime_s"],
        ),
    ]
    (directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_final(context: Context, hyperparameters_csv: str | Path | None = None) -> None:
    write_pipeline_config(context)
    for n in context.n_values:
        load_graph_metadata(context, n, expected=context.n_graphs)

    base = context.base
    directory = final_dir(context)
    run_json_dir = directory / "run_results"
    array_dir = directory / "run_arrays"
    mask_dir = directory / "decision_masks"
    for path in [directory, run_json_dir, array_dir, mask_dir]:
        path.mkdir(parents=True, exist_ok=True)

    hp = load_selected_hyperparameters(context, hyperparameters_csv)
    required_k = {int(item.k_order) for item in choices(context)}
    available_k = set(pd.to_numeric(hp["k_order"]).astype(int))
    if required_k - available_k:
        raise RuntimeError(f"Faltam hiperparâmetros para k={sorted(required_k - available_k)}")

    base.SAVE_ALL_ARRAYS = context.keep_arrays != "none"
    base.UPDATE_PLOT_AFTER_EACH_GRAPH = False
    base.CREATE_FINAL_ZIP = False
    expected = len(context.n_values) * context.n_graphs * len(context.training_seeds)
    print(f"Rodada final esperada: {expected} execuções de {context.final_steps} épocas.")

    with base.ExperimentLock(directory / ".final.lock"):
        for choice in choices(context):
            n = int(choice.n_vertices)
            hp_row = hp[pd.to_numeric(hp["k_order"]).astype(int) == int(choice.k_order)].iloc[-1]
            alpha = float(hp_row.selected_alpha)
            beta = float(hp_row.selected_beta)
            masks = ensure_decision_masks(context, choice, mask_dir)
            runtime = runtime_config(context, directory, choice, context.final_steps)
            graphs = load_graph_metadata(context, n, expected=context.n_graphs)
            print("\n" + "=" * 94)
            print(f"FINAL n={n}, q={choice.q_qubits}, k={choice.k_order}, alpha={alpha:g}, beta={beta:g}")
            print("=" * 94)

            for position, graph_row in enumerate(graphs.itertuples(index=False), start=1):
                graph_id = str(graph_row.graph_id)
                graph = graph_from_row(context, graph_row)
                reference = float(graph_row.initial_reference_cut)
                graph_rows: list[tuple[Path, dict[str, Any]]] = []
                n_dir = run_json_dir / f"n{n}"
                n_array_dir = array_dir / f"n{n}"
                n_dir.mkdir(parents=True, exist_ok=True)
                n_array_dir.mkdir(parents=True, exist_ok=True)

                for seed in context.training_seeds:
                    identifier = base.run_identifier(
                        choice=choice,
                        graph_id=graph_id,
                        init_seed=int(seed),
                        alpha=alpha,
                        beta=beta,
                        config=runtime,
                    )
                    result_json = n_dir / f"{identifier}.json"
                    array_npz = n_array_dir / f"{identifier}.npz"
                    row = base.train_full_basis(
                        choice=choice,
                        graph_id=graph_id,
                        graph=graph,
                        initial_reference_cut=reference,
                        decision_masks_np=masks,
                        alpha=alpha,
                        beta=beta,
                        init_seed=int(seed),
                        config=runtime,
                        result_json=result_json,
                        array_npz=array_npz,
                    )
                    updates = {
                        "phase": "final",
                        "hyperparameter_source": hp.attrs.get("source_path", "unknown"),
                    }
                    if any(row.get(key) != value for key, value in updates.items()):
                        row = update_json(base, result_json, updates)
                    graph_rows.append((result_json, row))

                best_index = prune_arrays(base, graph_rows, context.keep_arrays)
                if context.keep_arrays == "best" and best_index is not None:
                    best_json, best_row = graph_rows[best_index]
                    best_array = Path(str(best_row["arrays_path"]))
                    if not best_array.is_file() or best_array.stat().st_size == 0:
                        # Recuperação rara: o melhor atual era um placeholder de
                        # uma execução anterior incompleta. Reexecuta somente ele.
                        best_json.unlink(missing_ok=True)
                        best_array.unlink(missing_ok=True)
                        seed = int(best_row["init_seed"])
                        recovered = base.train_full_basis(
                            choice=choice,
                            graph_id=graph_id,
                            graph=graph,
                            initial_reference_cut=reference,
                            decision_masks_np=masks,
                            alpha=alpha,
                            beta=beta,
                            init_seed=seed,
                            config=runtime,
                            result_json=best_json,
                            array_npz=best_array,
                        )
                        update_json(base, best_json, {"phase": "final", "arrays_saved": True})

                if position % 5 == 0 or position == len(graphs):
                    runs = collect_json_rows(run_json_dir)
                    base.write_csv_atomic(directory / "final_runs_partial.csv", runs)
                    print(f"  n={n}: {position}/{len(graphs)} grafos; total {len(runs)}/{expected} runs")

        runs = collect_json_rows(run_json_dir)
        base.write_csv_atomic(directory / "final_runs.csv", runs)
        all_summary, best, best_summary = final_summaries(runs)
        base.write_csv_atomic(directory / "final_summary_all_seeds.csv", all_summary)
        base.write_csv_atomic(directory / "final_best_seed_per_graph.csv", best)
        base.write_csv_atomic(directory / "final_summary_best_of_10.csv", best_summary)
        write_final_report(context, hp, runs, all_summary, best_summary)
        base.write_json_atomic(
            directory / "manifest_completed.json",
            {
                "completed_utc": base.utc_now(),
                "completed_runs": int(len(runs)),
                "expected_runs": int(expected),
                "keep_arrays": context.keep_arrays,
            },
        )
    print(f"\nRodada final concluída em: {directory}")


# =============================================================================
# CLI
# =============================================================================

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=cfg.N_VALUES)
    parser.add_argument("--graphs", type=int, default=cfg.N_GRAPHS_PER_SIZE)
    parser.add_argument("--calibration-graphs", type=int, default=cfg.CALIBRATION_GRAPHS_PER_SIZE)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=cfg.TRAINING_SEEDS)
    parser.add_argument("--mqlib-runtime", type=float, default=cfg.MQLIB_RUNTIME_SECONDS)
    parser.add_argument("--calibration-steps", type=int, default=cfg.CALIBRATION_STEPS)
    parser.add_argument("--final-steps", type=int, default=cfg.FINAL_STEPS)
    parser.add_argument("--keep-arrays", choices=["all", "best", "none"], default=cfg.KEEP_FINAL_ARRAYS)


def parser_for_phase(phase: str) -> argparse.ArgumentParser:
    descriptions = {
        "prepare": "Fase 1: gerar e validar o banco de grafos.",
        "calibrate": "Fase 2: calibrar alpha e beta por k.",
        "final": "Fase 3: rodar 500 épocas nos grafos completos.",
        "all": "Executar as três fases em sequência.",
    }
    parser = argparse.ArgumentParser(description=descriptions[phase])
    add_common_arguments(parser)
    if phase == "final":
        parser.add_argument("--hyperparameters-csv", default=None)
    if phase == "all":
        parser.add_argument("--stop-after", choices=["graphs", "calibration", "final"], default="final")
        parser.add_argument("--hyperparameters-csv", default=None)
    return parser


def cli(phase: str, argv: Sequence[str] | None = None) -> None:
    args = parser_for_phase(phase).parse_args(argv)
    context = build_context(args)
    print(f"Projeto: {context.project_root}")
    print(f"Saída:   {context.output_root}")
    print(f"Seeds:   {list(context.training_seeds)}")
    print(choice_table(context).to_string(index=False))

    if phase == "prepare":
        prepare_graphs(context)
    elif phase == "calibrate":
        calibrate(context)
    elif phase == "final":
        run_final(context, args.hyperparameters_csv)
    elif phase == "all":
        prepare_graphs(context)
        if args.stop_after == "graphs":
            return
        calibrate(context)
        if args.stop_after == "calibration":
            return
        run_final(context, args.hyperparameters_csv)
    else:
        raise ValueError(phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Walsh/PCE v2")
    parser.add_argument("phase", choices=["prepare", "calibrate", "final", "all"])
    known, remaining = parser.parse_known_args()
    cli(known.phase, remaining)
