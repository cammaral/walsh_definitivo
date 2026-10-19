#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTO 1 — REFERÊNCIA WALSH COMPLETA EM GRAFOS ALEATÓRIOS
================================================================

Objetivo
--------
Treinar a base Walsh completa em grafos MaxCut aleatórios e salvar a
referência expressiva que será usada posteriormente para:

1. projetar a solução completa nos suportes estruturados

       A_B = B_B union {b xor S_i : b in B_B, S_i in S};

2. medir preservação da distribuição, dos correlatores, dos sinais e do corte;

3. comparar treinamento independente e crescimento progressivo dos suportes.

Protocolo padrão
----------------
- tamanhos n = 100, 800 e 2000;
- para cada n, escolhe automaticamente o menor q e o k central que maximizam
  C(q, k), sob C(q, k) >= n;
- 250 grafos aleatórios por tamanho;
- 5 inicializações por grafo;
- Adam por 300 passos;
- learning rate = 1e-3;
- base Walsh completa, com 2**q coeficientes;
- correlatores calculados exatamente por FWHT;
- sem bitflip;
- referência clássica calculada pelo MQLib;
- filtros inspirados no PCE:
    grau médio >= 3;
    grafo conectado;
    AR de um corte aleatório <= 0.82;
- alpha e beta:
    primeiro tenta ler os valores selecionados pelo estudo anterior em
    results_full_basis_alpha_beta/selected_hyperparameters.csv;
    caso não encontre, usa alpha = 1.5*q e beta = 0.5.

Localização
-----------
Coloque este arquivo na RAIZ do projeto. Exemplo:

    projeto/
    ├── experiment_1_full_walsh_random.py
    ├── MQLib/
    ├── src/
    └── results/

As saídas serão criadas em:

    results/experiment_1_full_walsh_random/

Execução
--------
Execução final padrão:

    python experiment_1_full_walsh_random.py

Somente um tamanho:

    python experiment_1_full_walsh_random.py --n-values 800

Retomar:
    basta executar o mesmo comando; grafos e treinamentos concluídos são
    detectados e reutilizados.

Indicar MQLib explicitamente:

    export MQLIB_EXECUTABLE=/caminho/MQLib/bin/MQLib
    python experiment_1_full_walsh_random.py

Dependências
------------
numpy, pandas, networkx, matplotlib, torch

Observação
----------
O MQLib fornece a melhor referência encontrada dentro do orçamento escolhido,
não uma prova de ótimo global para grafos grandes. Ao final, a referência de
cada grafo é atualizada caso alguma execução Walsh encontre um corte maior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


# =============================================================================
# CONFIGURAÇÃO FINAL DO EXPERIMENTO
# =============================================================================

DEFAULT_N_VALUES = [100, 200, 400, 500, 800, 2000, 7000]

N_GRAPHS_PER_SIZE = 250
TRAINING_SEEDS = [11, 22, 33, 44, 55]

STEPS = 300
LEARNING_RATE = 1.0e-3
EVAL_EVERY = 10

DEFAULT_ALPHA_MULTIPLIER = 1.5
DEFAULT_BETA = 0.5

TARGET_AVERAGE_DEGREE = 6.0
MIN_AVERAGE_DEGREE = 3.0
REQUIRE_CONNECTED = True
RANDOM_CUT_AR_MAX = 0.82
RANDOM_CUT_DIAGNOSTIC_SAMPLES = 200
MAX_GRAPH_ATTEMPTS = 100_000

MQLIB_HEURISTICS = [
    "BURER2002",
    "FESTA2002G",
    "FESTA2002VNS",
    "LAGUNA2009HCE",
]
MQLIB_SEEDS = [101, 202]
MQLIB_RUNTIME_SECONDS = 1.0
MQLIB_TIMEOUT_FACTOR = 5.0
AUTO_BUILD_MQLIB = True
MQLIB_BUILD_JOBS = max(1, min(8, os.cpu_count() or 1))
EXACT_MAXCUT_MAX_VERTICES = 24

MASTER_SEED = 20260616
DECISION_MASK_SEED = 271828

DTYPE_NAME = "float64"
DEVICE_NAME = "cpu"
TORCH_NUM_THREADS = max(1, min(8, os.cpu_count() or 1))

MAX_FULL_WALSH_COEFFICIENTS = 262_144  # até q=18
SAVE_ALL_ARRAYS = True
UPDATE_PLOT_AFTER_EACH_GRAPH = True
CREATE_FINAL_ZIP = True

OUTPUT_RELATIVE_PATH = Path("results") / "experiment_1_full_walsh_random"
CACHE_VERSION = 1


# =============================================================================
# ESTRUTURAS DE CONFIGURAÇÃO
# =============================================================================

@dataclass(frozen=True)
class EncodingChoice:
    n_vertices: int
    q_qubits: int
    k_order: int
    capacity: int
    symmetric_k: int
    full_basis_size: int


@dataclass
class RuntimeConfig:
    project_dir: Path
    output_dir: Path
    n_values: list[int]
    n_graphs: int
    training_seeds: list[int]
    steps: int
    learning_rate: float
    eval_every: int
    mqlib_runtime_seconds: float
    device: str
    dtype_name: str


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: Any, modulus: int = 2**32 - 1) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(int(hashlib.sha256(raw).hexdigest()[:16], 16) % modulus)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def unique_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    return Path(name)


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = unique_temp_path(path)
    try:
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    tmp = unique_temp_path(path)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as file:
            frame.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz.tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp = Path(name)
    try:
        # np.savez adiciona .npz se receber string sem essa extensão.
        with tmp.open("wb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def environment_info() -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "networkx": nx.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


class ExperimentLock:
    """Evita duas execuções escrevendo na mesma pasta de resultados."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    @staticmethod
    def process_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def __enter__(self) -> "ExperimentLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                old = read_json(self.path)
                old_pid = int(old.get("pid", -1))
            except Exception:
                old_pid = -1

            if self.process_exists(old_pid):
                raise RuntimeError(
                    "Já existe uma execução ativa usando esta pasta de "
                    f"resultados (PID {old_pid})."
                )
            self.path.unlink(missing_ok=True)

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(self.path, flags)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(
                {"pid": os.getpid(), "created_utc": utc_now()},
                file,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


# =============================================================================
# ESCOLHA DE q, k E MÁSCARAS
# =============================================================================

def choose_minimum_qubit_encoding(n_vertices: int) -> EncodingChoice:
    """
    Menor q tal que max_k C(q,k) >= n.

    No q mínimo, escolhe o menor dos k centrais que atingem a capacidade
    máxima. Isso fornece a maior compressão possível em número de qubits.
    """
    if n_vertices < 2:
        raise ValueError("n_vertices deve ser >= 2.")

    q = 2
    while True:
        capacities = [(k, math.comb(q, k)) for k in range(1, q)]
        maximum = max(capacity for _, capacity in capacities)
        if maximum >= n_vertices:
            best_k = min(k for k, capacity in capacities if capacity == maximum)
            return EncodingChoice(
                n_vertices=n_vertices,
                q_qubits=q,
                k_order=best_k,
                capacity=maximum,
                symmetric_k=q - best_k,
                full_basis_size=1 << q,
            )
        q += 1


def masks_of_weight(q: int, k: int) -> np.ndarray:
    values: list[int] = []
    for positions in itertools.combinations(range(q), k):
        mask = 0
        for position in positions:
            mask |= 1 << position
        values.append(mask)
    return np.asarray(values, dtype=np.int64)


def choose_decision_masks(choice: EncodingChoice) -> np.ndarray:
    """
    Seleção determinística e reproduzível de n máscaras de peso k.

    A permutação evita escolher somente as primeiras combinações lexicográficas,
    mantendo as mesmas máscaras para todos os grafos do mesmo tamanho.
    """
    candidates = masks_of_weight(choice.q_qubits, choice.k_order)
    if len(candidates) < choice.n_vertices:
        raise RuntimeError("Capacidade insuficiente para as máscaras.")

    rng = np.random.default_rng(
        stable_seed(
            MASTER_SEED,
            DECISION_MASK_SEED,
            choice.n_vertices,
            choice.q_qubits,
            choice.k_order,
        )
    )
    chosen = rng.choice(
        candidates,
        size=choice.n_vertices,
        replace=False,
    )
    return np.sort(chosen.astype(np.int64))


# =============================================================================
# GRAFOS E MAXCUT
# =============================================================================

def graph_edge_arrays(
    graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = sorted(graph.edges(data=True))
    edge_u = np.asarray([int(u) for u, _, _ in edges], dtype=np.int64)
    edge_v = np.asarray([int(v) for _, v, _ in edges], dtype=np.int64)
    edge_w = np.asarray(
        [float(data.get("weight", 1.0)) for _, _, data in edges],
        dtype=np.float64,
    )
    return edge_u, edge_v, edge_w


def cut_value_arrays(
    bits: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
) -> float:
    bits = np.asarray(bits, dtype=np.uint8)
    return float(np.sum(edge_w * (bits[edge_u] != bits[edge_v])))


def cut_value(bits: np.ndarray, graph: nx.Graph) -> float:
    edge_u, edge_v, edge_w = graph_edge_arrays(graph)
    return cut_value_arrays(bits, edge_u, edge_v, edge_w)


def build_adjacency(graph: nx.Graph) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(graph.number_of_nodes())
    ]
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        adjacency[int(u)].append((int(v), weight))
        adjacency[int(v)].append((int(u), weight))
    return adjacency


def exact_maxcut_gray(graph: nx.Graph) -> tuple[float, np.ndarray]:
    n = graph.number_of_nodes()
    if n > EXACT_MAXCUT_MAX_VERTICES:
        raise ValueError("Grafo maior que o limite exato configurado.")

    adjacency = build_adjacency(graph)
    bits = np.zeros(n, dtype=np.uint8)
    current_cut = 0.0
    best_cut = 0.0
    best_bits = bits.copy()

    # Fixa o primeiro bit para remover a simetria global.
    for value in range(1, 1 << (n - 1)):
        low_bit = value & -value
        local_position = low_bit.bit_length() - 1
        vertex = local_position + 1
        old_bit = int(bits[vertex])

        gain = 0.0
        for neighbor, weight in adjacency[vertex]:
            gain += (
                weight
                if old_bit == int(bits[neighbor])
                else -weight
            )

        bits[vertex] ^= 1
        current_cut += gain
        if current_cut > best_cut:
            best_cut = float(current_cut)
            best_bits = bits.copy()

    return best_cut, best_bits


def estimate_nu_unweighted(graph: nx.Graph) -> float:
    """
    Escala de Edwards–Erdős usada na regularização PCE para MaxCut
    não ponderado.
    """
    return (
        0.5 * graph.number_of_edges()
        + 0.25 * (graph.number_of_nodes() - 1)
    )


def save_graph_mqlib(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = sorted(
        (
            min(int(u), int(v)) + 1,
            max(int(u), int(v)) + 1,
            float(data.get("weight", 1.0)),
        )
        for u, v, data in graph.edges(data=True)
        if int(u) != int(v)
    )

    with path.open("w", encoding="utf-8") as file:
        file.write(f"{graph.number_of_nodes()} {len(edges)}\n")
        for u, v, weight in edges:
            if abs(weight - round(weight)) < 1e-12:
                file.write(f"{u} {v} {int(round(weight))}\n")
            else:
                file.write(f"{u} {v} {weight:.17g}\n")


def load_graph_mqlib(path: Path) -> nx.Graph:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "%", "c"))
    ]
    n_vertices, _ = map(int, lines[0].split()[:2])
    graph = nx.Graph()
    graph.add_nodes_from(range(n_vertices))

    for line in lines[1:]:
        parts = line.split()
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        weight = float(parts[2]) if len(parts) >= 3 else 1.0
        graph.add_edge(u, v, weight=weight)

    return graph


# =============================================================================
# MQLIB
# =============================================================================

def candidate_mqlib_roots(project_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for base in [
        project_dir,
        project_dir.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]:
        for name in ["MQLib", "mqlib", "MQLIB"]:
            candidate = (base / name).resolve()
            if candidate.exists() and candidate not in roots:
                roots.append(candidate)
    return roots


def find_or_build_mqlib(project_dir: Path) -> tuple[Path, Path]:
    environment_path = os.environ.get("MQLIB_EXECUTABLE")
    if environment_path:
        executable = Path(environment_path).expanduser().resolve()
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable, executable.parent.parent
        raise FileNotFoundError(
            "MQLIB_EXECUTABLE precisa apontar para um arquivo executável: "
            f"{executable}"
        )

    roots = candidate_mqlib_roots(project_dir)

    for root in roots:
        candidates = [
            root / "bin" / "MQLib",
            root / "build" / "MQLib",
            root / "MQLib",
            root / "MQLib" / "bin" / "MQLib",
        ]
        for executable in candidates:
            if executable.is_file() and os.access(executable, os.X_OK):
                return executable.resolve(), root.resolve()

    if AUTO_BUILD_MQLIB:
        for root in roots:
            if not (root / "Makefile").is_file():
                continue
            print(f"Compilando MQLib em {root} ...")
            process = subprocess.run(
                ["make", f"-j{MQLIB_BUILD_JOBS}"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            if process.returncode != 0:
                print(process.stdout)
                print(process.stderr, file=sys.stderr)
                continue

            for executable in [
                root / "bin" / "MQLib",
                root / "build" / "MQLib",
                root / "MQLib",
            ]:
                if executable.is_file():
                    executable.chmod(executable.stat().st_mode | 0o111)
                    if os.access(executable, os.X_OK):
                        return executable.resolve(), root.resolve()

    raise FileNotFoundError(
        "Executável MQLib não encontrado. Defina:\n"
        "export MQLIB_EXECUTABLE=/caminho/MQLib/bin/MQLib"
    )


def parse_mqlib_summary(stdout: str, heuristic: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if heuristic not in line:
            continue
        try:
            fields = next(csv.reader([line.strip()]))
        except Exception:
            continue

        if len(fields) >= 5 and fields[1].strip() == heuristic:
            return {
                "reported_runtime_limit": float(fields[0]),
                "reported_heuristic": fields[1].strip(),
                "reported_instance": fields[2],
                "reported_objective": float(fields[3]),
                "reported_runtime": float(fields[4]),
                "reported_history": fields[5] if len(fields) > 5 else "",
                "summary_line": line.strip(),
            }

    raise ValueError("Não foi possível interpretar a saída do MQLib.")


def run_mqlib(
    executable: Path,
    graph_file: Path,
    heuristic: str,
    seed: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(f"Executável MQLib inválido: {executable}")

    command = [
        str(executable),
        "-fM", str(graph_file),
        "-h", str(heuristic),
        "-r", str(float(runtime_seconds)),
        "-s", str(int(seed)),
        "-ps",
    ]
    timeout = max(
        30.0,
        runtime_seconds * MQLIB_TIMEOUT_FACTOR + 15.0,
    )

    started = time.perf_counter()
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    row: dict[str, Any] = {
        "heuristic": heuristic,
        "seed": int(seed),
        "command": json.dumps(command),
        "returncode": int(process.returncode),
        "wall_time_s": float(time.perf_counter() - started),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "status": "ok" if process.returncode == 0 else "error",
    }

    if process.returncode == 0:
        try:
            row.update(parse_mqlib_summary(process.stdout, heuristic))
        except Exception as error:
            row["status"] = "parse_error"
            row["parse_error"] = str(error)

    return row


# =============================================================================
# WALSH COMPLETO
# =============================================================================

def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"dtype não reconhecido: {name}")


def fwht(vector: torch.Tensor) -> torch.Tensor:
    """Fast Walsh–Hadamard Transform não normalizada e diferenciável."""
    if vector.ndim != 1:
        raise ValueError("FWHT espera um vetor unidimensional.")

    size = vector.numel()
    if size <= 0 or size & (size - 1):
        raise ValueError("O tamanho da FWHT deve ser potência de dois.")

    transformed = vector
    block = 1

    while block < size:
        reshaped = transformed.reshape(-1, 2, block)
        left = reshaped[:, 0, :]
        right = reshaped[:, 1, :]
        transformed = torch.cat(
            (left + right, left - right),
            dim=1,
        ).reshape(size)
        block *= 2

    return transformed


def full_walsh_correlators(
    coefficients: torch.Tensor,
    decision_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula r[S] = sum_l a_l a_{l xor S} por FWHT:

        r = H((H a)^2) / N.
    """
    size = coefficients.numel()
    transformed = fwht(coefficients)
    autocorrelation = fwht(transformed.square()) / float(size)
    epsilon = torch.finfo(coefficients.dtype).eps
    denominator = autocorrelation[0].clamp_min(epsilon)
    return autocorrelation[decision_masks] / denominator


def walsh_probability_distribution(
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    f(x) = sum_l a_l w_l(x) = FWHT(a);
    p(x) = f(x)^2 / sum_y f(y)^2.
    """
    amplitudes_unnormalized = fwht(coefficients)
    probabilities = amplitudes_unnormalized.square()
    return probabilities / probabilities.sum().clamp_min(
        torch.finfo(coefficients.dtype).eps
    )


def initialize_full_coefficients(
    choice: EncodingChoice,
    decision_masks: np.ndarray,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Inicialização alinhada à estrutura usada no projeto:
    termo constante dominante, máscaras de decisão com ruído pequeno
    e ruído de fundo muito pequeno.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    coefficients = torch.zeros(
        choice.full_basis_size,
        dtype=dtype,
        device="cpu",
    )
    coefficients[0] = 1.0

    decision_noise = 1.0e-2 * torch.randn(
        len(decision_masks),
        dtype=dtype,
        generator=generator,
    )
    coefficients[
        torch.as_tensor(decision_masks, dtype=torch.long)
    ] = decision_noise

    coefficients += 1.0e-4 * torch.randn(
        choice.full_basis_size,
        dtype=dtype,
        generator=generator,
    )
    coefficients /= coefficients.norm().clamp_min(torch.finfo(dtype).eps)
    return coefficients.to(device)


def bits_from_correlators(correlators: np.ndarray) -> np.ndarray:
    """
    z_i = sign(c_i), x_i=(1-z_i)/2.
    A inversão global não altera o valor do corte.
    """
    return (np.asarray(correlators) < 0.0).astype(np.uint8)


# =============================================================================
# HIPERPARÂMETROS
# =============================================================================

def find_hyperparameter_csv(project_dir: Path) -> Path | None:
    candidates = [
        project_dir
        / "results_full_basis_alpha_beta"
        / "selected_hyperparameters.csv",
        project_dir
        / "results"
        / "full_basis_alpha_beta"
        / "selected_hyperparameters.csv",
    ]

    candidates.extend(
        sorted(
            project_dir.glob(
                "results_full_basis_alpha_beta*/selected_hyperparameters.csv"
            )
        )
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_alpha_beta(
    choice: EncodingChoice,
    project_dir: Path,
) -> tuple[float, float, str]:
    csv_path = find_hyperparameter_csv(project_dir)

    if csv_path is not None:
        try:
            frame = pd.read_csv(csv_path)
            n_column = (
                "n_vertices"
                if "n_vertices" in frame.columns
                else "n"
                if "n" in frame.columns
                else None
            )

            if n_column is not None:
                matches = frame[
                    pd.to_numeric(
                        frame[n_column],
                        errors="coerce",
                    )
                    == choice.n_vertices
                ]
                if not matches.empty:
                    row = matches.iloc[-1]
                    alpha = float(row["selected_alpha"])
                    beta = float(row["selected_beta"])
                    return alpha, beta, f"arquivo:{csv_path}"
        except Exception as error:
            print(
                f"[aviso] não foi possível ler hiperparâmetros de "
                f"{csv_path}: {error}"
            )

    alpha = DEFAULT_ALPHA_MULTIPLIER * choice.q_qubits
    return alpha, DEFAULT_BETA, "fallback:Aolita_alpha_1.5q_beta_0.5"


# =============================================================================
# TREINAMENTO
# =============================================================================

def run_identifier(
    choice: EncodingChoice,
    graph_id: str,
    init_seed: int,
    alpha: float,
    beta: float,
    config: RuntimeConfig,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "n": choice.n_vertices,
        "q": choice.q_qubits,
        "k": choice.k_order,
        "graph_id": graph_id,
        "init_seed": init_seed,
        "alpha": alpha,
        "beta": beta,
        "steps": config.steps,
        "lr": config.learning_rate,
        "dtype": config.dtype_name,
        "device": config.device,
        "full_basis": True,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{graph_id}_q{choice.q_qubits}_k{choice.k_order}_"
        f"seed{init_seed}_{digest}"
    )


def train_full_basis(
    *,
    choice: EncodingChoice,
    graph_id: str,
    graph: nx.Graph,
    initial_reference_cut: float,
    decision_masks_np: np.ndarray,
    alpha: float,
    beta: float,
    init_seed: int,
    config: RuntimeConfig,
    result_json: Path,
    array_npz: Path,
) -> dict[str, Any]:
    if result_json.is_file() and array_npz.is_file():
        return read_json(result_json)

    if choice.full_basis_size > MAX_FULL_WALSH_COEFFICIENTS:
        raise MemoryError(
            f"Base completa 2^{choice.q_qubits}="
            f"{choice.full_basis_size} excede o limite "
            f"{MAX_FULL_WALSH_COEFFICIENTS}."
        )

    dtype = torch_dtype_from_name(config.dtype_name)
    device = torch.device(config.device)

    edge_u_np, edge_v_np, edge_w_np = graph_edge_arrays(graph)
    edge_u = torch.as_tensor(edge_u_np, dtype=torch.long, device=device)
    edge_v = torch.as_tensor(edge_v_np, dtype=torch.long, device=device)
    edge_w = torch.as_tensor(edge_w_np, dtype=dtype, device=device)
    decision_masks = torch.as_tensor(
        decision_masks_np,
        dtype=torch.long,
        device=device,
    )

    effective_seed = stable_seed(
        MASTER_SEED,
        "full_training",
        choice.n_vertices,
        graph_id,
        init_seed,
    )

    coefficients = torch.nn.Parameter(
        initialize_full_coefficients(
            choice=choice,
            decision_masks=decision_masks_np,
            seed=effective_seed,
            dtype=dtype,
            device=device,
        )
    )
    optimizer = torch.optim.Adam(
        [coefficients],
        lr=config.learning_rate,
    )

    nu = estimate_nu_unweighted(graph)
    loss_history = np.empty(config.steps, dtype=np.float64)
    data_loss_history = np.empty(config.steps, dtype=np.float64)
    regularization_history = np.empty(config.steps, dtype=np.float64)
    gradient_history = np.empty(config.steps, dtype=np.float64)

    evaluation_steps: list[int] = []
    evaluation_cuts: list[float] = []
    evaluation_ars: list[float] = []
    evaluation_mean_abs_corr: list[float] = []
    evaluation_saturation: list[float] = []

    started = time.perf_counter()

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        correlators = full_walsh_correlators(
            coefficients,
            decision_masks,
        )
        relaxed_spins = torch.tanh(float(alpha) * correlators)

        data_loss = torch.sum(
            edge_w * relaxed_spins[edge_u] * relaxed_spins[edge_v]
        )
        regularization = (
            float(beta)
            * float(nu)
            * torch.mean(relaxed_spins.square()).square()
        )
        total_loss = data_loss + regularization

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Loss não finita em {graph_id}, seed={init_seed}, "
                f"step={step}."
            )

        total_loss.backward()
        gradient_norm = float(coefficients.grad.norm().detach().cpu())
        optimizer.step()

        # Os correlatores são invariantes à escala global de a.
        with torch.no_grad():
            coefficients.div_(
                coefficients.norm().clamp_min(
                    torch.finfo(dtype).eps
                )
            )

        loss_history[step - 1] = float(total_loss.detach().cpu())
        data_loss_history[step - 1] = float(data_loss.detach().cpu())
        regularization_history[step - 1] = float(
            regularization.detach().cpu()
        )
        gradient_history[step - 1] = gradient_norm

        if (
            step == 1
            or step % config.eval_every == 0
            or step == config.steps
        ):
            with torch.no_grad():
                eval_corr = full_walsh_correlators(
                    coefficients,
                    decision_masks,
                )
                eval_spins = torch.tanh(float(alpha) * eval_corr)
                eval_corr_np = eval_corr.detach().cpu().numpy()
                eval_bits = bits_from_correlators(eval_corr_np)
                eval_cut = cut_value_arrays(
                    eval_bits,
                    edge_u_np,
                    edge_v_np,
                    edge_w_np,
                )

            evaluation_steps.append(step)
            evaluation_cuts.append(eval_cut)
            evaluation_ars.append(eval_cut / initial_reference_cut)
            evaluation_mean_abs_corr.append(
                float(np.mean(np.abs(eval_corr_np)))
            )
            evaluation_saturation.append(
                float(np.mean(np.abs(eval_spins.cpu().numpy()) > 0.99))
            )

    runtime_s = float(time.perf_counter() - started)

    with torch.no_grad():
        final_correlators_t = full_walsh_correlators(
            coefficients,
            decision_masks,
        )
        final_spins_t = torch.tanh(
            float(alpha) * final_correlators_t
        )
        final_probabilities_t = walsh_probability_distribution(
            coefficients
        )

        final_coefficients = coefficients.detach().cpu().numpy()
        final_correlators = final_correlators_t.detach().cpu().numpy()
        final_spins = final_spins_t.detach().cpu().numpy()
        final_probabilities = (
            final_probabilities_t.detach().cpu().numpy()
        )
        final_bits = bits_from_correlators(final_correlators)
        final_cut = cut_value_arrays(
            final_bits,
            edge_u_np,
            edge_v_np,
            edge_w_np,
        )

    row = {
        "created_utc": utc_now(),
        "cache_version": CACHE_VERSION,
        "graph_id": graph_id,
        "n_vertices": choice.n_vertices,
        "q_qubits": choice.q_qubits,
        "k_order": choice.k_order,
        "capacity": choice.capacity,
        "full_basis_size": choice.full_basis_size,
        "n_trainable_parameters": choice.full_basis_size,
        "init_seed": int(init_seed),
        "effective_seed": int(effective_seed),
        "alpha": float(alpha),
        "beta": float(beta),
        "steps": int(config.steps),
        "learning_rate": float(config.learning_rate),
        "eval_every": int(config.eval_every),
        "initial_reference_cut": float(initial_reference_cut),
        "final_raw_cut": float(final_cut),
        "final_raw_ar_initial_reference": float(
            final_cut / initial_reference_cut
        ),
        "final_total_loss": float(loss_history[-1]),
        "final_data_loss": float(data_loss_history[-1]),
        "final_regularization": float(regularization_history[-1]),
        "runtime_s": runtime_s,
        "mean_abs_correlator": float(
            np.mean(np.abs(final_correlators))
        ),
        "min_abs_correlator": float(
            np.min(np.abs(final_correlators))
        ),
        "mean_abs_spin": float(np.mean(np.abs(final_spins))),
        "saturation_fraction_099": float(
            np.mean(np.abs(final_spins) > 0.99)
        ),
        "sign_balance_fraction_one": float(np.mean(final_bits)),
        "probability_entropy": float(
            -np.sum(
                final_probabilities
                * np.log(
                    np.clip(final_probabilities, 1e-300, None)
                )
            )
        ),
        "coefficient_l1": float(np.sum(np.abs(final_coefficients))),
        "coefficient_l2": float(
            np.linalg.norm(final_coefficients)
        ),
        "coefficient_linf": float(
            np.max(np.abs(final_coefficients))
        ),
        "max_gradient_norm": float(np.max(gradient_history)),
        "best_evaluated_raw_cut": float(max(evaluation_cuts)),
        "best_evaluated_raw_ar_initial_reference": float(
            max(evaluation_cuts) / initial_reference_cut
        ),
        "bitflip_applied": False,
        "arrays_path": str(array_npz),
    }

    if SAVE_ALL_ARRAYS:
        save_npz_atomic(
            array_npz,
            coefficients=final_coefficients,
            probabilities=final_probabilities,
            correlators=final_correlators,
            relaxed_spins=final_spins,
            bits=final_bits,
            decision_masks=decision_masks_np,
            loss_history=loss_history,
            data_loss_history=data_loss_history,
            regularization_history=regularization_history,
            gradient_history=gradient_history,
            evaluation_steps=np.asarray(
                evaluation_steps,
                dtype=np.int64,
            ),
            evaluation_cuts=np.asarray(
                evaluation_cuts,
                dtype=np.float64,
            ),
            evaluation_ars_initial_reference=np.asarray(
                evaluation_ars,
                dtype=np.float64,
            ),
            evaluation_mean_abs_correlator=np.asarray(
                evaluation_mean_abs_corr,
                dtype=np.float64,
            ),
            evaluation_saturation_099=np.asarray(
                evaluation_saturation,
                dtype=np.float64,
            ),
            alpha=np.asarray(alpha),
            beta=np.asarray(beta),
            q=np.asarray(choice.q_qubits),
            k=np.asarray(choice.k_order),
            n=np.asarray(choice.n_vertices),
            initial_reference_cut=np.asarray(initial_reference_cut),
            final_raw_cut=np.asarray(final_cut),
            init_seed=np.asarray(init_seed),
        )
    else:
        array_npz.touch()

    write_json_atomic(result_json, row)
    return row


# =============================================================================
# GRÁFICOS E AGREGAÇÃO
# =============================================================================

def collect_run_rows(run_json_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_json_dir.glob("*.json")):
        try:
            rows.append(read_json(path))
        except Exception as error:
            print(f"[aviso] resultado inválido ignorado {path}: {error}")
    return pd.DataFrame(rows)


def update_progress_plot(
    *,
    n_vertices: int,
    run_frame: pd.DataFrame,
    plot_path: Path,
) -> None:
    if run_frame.empty:
        return

    current = run_frame[
        pd.to_numeric(
            run_frame["n_vertices"],
            errors="coerce",
        )
        == n_vertices
    ].copy()
    if current.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].hist(
        current["final_raw_ar_initial_reference"].astype(float),
        bins=min(30, max(8, int(math.sqrt(len(current))))),
    )
    axes[0].set_xlabel("Raw approximation ratio")
    axes[0].set_ylabel("Runs")
    axes[0].set_title("Distribution across completed runs")
    axes[0].grid(True, alpha=0.25)

    graph_summary = (
        current
        .groupby("graph_id", as_index=False)
        .agg(
            mean_ar=("final_raw_ar_initial_reference", "mean"),
            std_ar=("final_raw_ar_initial_reference", "std"),
        )
        .sort_values("graph_id")
    )
    x = np.arange(len(graph_summary))
    axes[1].errorbar(
        x,
        graph_summary["mean_ar"],
        yerr=graph_summary["std_ar"].fillna(0.0),
        marker=".",
        linewidth=1,
        capsize=2,
    )
    axes[1].set_xlabel("Completed graph index")
    axes[1].set_ylabel("Mean raw AR")
    axes[1].set_title("Mean over initialization seeds")
    axes[1].grid(True, alpha=0.25)

    arrays_to_read = [
        Path(path)
        for path in current["arrays_path"].tail(100)
        if Path(path).is_file()
    ]
    histories = []
    for array_path in arrays_to_read:
        try:
            with np.load(array_path) as data:
                histories.append(
                    np.asarray(data["loss_history"], dtype=float)
                )
        except Exception:
            pass

    if histories:
        matrix = np.vstack(histories)
        mean_loss = matrix.mean(axis=0)
        std_loss = matrix.std(axis=0)
        steps = np.arange(1, matrix.shape[1] + 1)
        axes[2].plot(steps, mean_loss)
        axes[2].fill_between(
            steps,
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.2,
        )
    axes[2].set_xlabel("Adam step")
    axes[2].set_ylabel("Total loss")
    axes[2].set_title("Mean loss of up to 100 recent runs")
    axes[2].grid(True, alpha=0.25)

    q = int(current["q_qubits"].iloc[0])
    k = int(current["k_order"].iloc[0])
    fig.suptitle(
        f"Experiment 1 progress — n={n_vertices}, q={q}, k={k}, "
        f"completed runs={len(current)}"
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_final_results(
    graph_frame: pd.DataFrame,
    run_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if graph_frame.empty or run_frame.empty:
        return run_frame, pd.DataFrame(), pd.DataFrame()

    graph_best_walsh = (
        run_frame
        .groupby("graph_id", as_index=False)
        .agg(best_walsh_cut=("final_raw_cut", "max"))
    )

    references = graph_frame[
        [
            "graph_id",
            "initial_reference_cut",
            "reference_source",
        ]
    ].merge(graph_best_walsh, on="graph_id", how="left")

    references["best_walsh_cut"] = references[
        "best_walsh_cut"
    ].fillna(-np.inf)

    references["updated_reference_cut"] = np.maximum(
        references["initial_reference_cut"].astype(float),
        references["best_walsh_cut"].astype(float),
    )
    references["reference_updated_by_walsh"] = (
        references["best_walsh_cut"]
        > references["initial_reference_cut"].astype(float) + 1e-12
    )

    enriched = run_frame.merge(
        references[
            [
                "graph_id",
                "updated_reference_cut",
                "reference_updated_by_walsh",
            ]
        ],
        on="graph_id",
        how="left",
    )
    enriched["final_raw_ar_updated_reference"] = (
        enriched["final_raw_cut"]
        / enriched["updated_reference_cut"]
    )

    summary = (
        enriched
        .groupby(
            ["n_vertices", "q_qubits", "k_order"],
            as_index=False,
        )
        .agg(
            n_graphs=("graph_id", "nunique"),
            n_runs=("graph_id", "size"),
            mean_raw_ar=("final_raw_ar_updated_reference", "mean"),
            std_raw_ar=("final_raw_ar_updated_reference", "std"),
            min_raw_ar=("final_raw_ar_updated_reference", "min"),
            max_raw_ar=("final_raw_ar_updated_reference", "max"),
            mean_raw_cut=("final_raw_cut", "mean"),
            mean_final_loss=("final_total_loss", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_abs_correlator=("mean_abs_correlator", "mean"),
            mean_min_abs_correlator=("min_abs_correlator", "mean"),
            mean_saturation_099=("saturation_fraction_099", "mean"),
            mean_probability_entropy=("probability_entropy", "mean"),
        )
    )

    return enriched, references, summary


# =============================================================================
# EXECUTOR DO EXPERIMENTO
# =============================================================================

class ExperimentRunner:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.project_dir = config.project_dir
        self.output_dir = config.output_dir

        self.graph_dir = self.output_dir / "graphs"
        self.graph_meta_dir = self.output_dir / "graph_metadata"
        self.run_json_dir = self.output_dir / "run_results"
        self.array_dir = self.output_dir / "run_arrays"
        self.plot_dir = self.output_dir / "plots"
        self.decision_mask_dir = self.output_dir / "decision_masks"

        for directory in [
            self.output_dir,
            self.graph_dir,
            self.graph_meta_dir,
            self.run_json_dir,
            self.array_dir,
            self.plot_dir,
            self.decision_mask_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.mqlib_executable, self.mqlib_root = find_or_build_mqlib(
            self.project_dir
        )

        write_json_atomic(
            self.output_dir / "environment.json",
            environment_info(),
        )
        write_json_atomic(
            self.output_dir / "config.json",
            {
                **asdict(config),
                "project_dir": str(config.project_dir),
                "output_dir": str(config.output_dir),
                "master_seed": MASTER_SEED,
                "target_average_degree": TARGET_AVERAGE_DEGREE,
                "minimum_average_degree": MIN_AVERAGE_DEGREE,
                "require_connected": REQUIRE_CONNECTED,
                "random_cut_ar_max": RANDOM_CUT_AR_MAX,
                "mqlib_executable": str(self.mqlib_executable),
                "mqlib_heuristics": MQLIB_HEURISTICS,
                "mqlib_seeds": MQLIB_SEEDS,
                "default_alpha_multiplier": DEFAULT_ALPHA_MULTIPLIER,
                "default_beta": DEFAULT_BETA,
                "bitflip": False,
                "full_walsh_basis": True,
                "loss_normalized": False,
            },
        )

    def graph_path(self, n_vertices: int, graph_index: int) -> Path:
        return (
            self.graph_dir
            / f"random_n{n_vertices}_g{graph_index:04d}.txt"
        )

    def graph_meta_path(
        self,
        n_vertices: int,
        graph_index: int,
    ) -> Path:
        return (
            self.graph_meta_dir
            / f"random_n{n_vertices}_g{graph_index:04d}.json"
        )

    def existing_graph_metadata(
        self,
        n_vertices: int,
    ) -> list[dict[str, Any]]:
        rows = []
        pattern = f"random_n{n_vertices}_g*.json"
        for path in sorted(self.graph_meta_dir.glob(pattern)):
            try:
                row = read_json(path)
                graph_file = Path(str(row["graph_file"]))
                if graph_file.is_file():
                    rows.append(row)
            except Exception as error:
                print(f"[aviso] metadado inválido {path}: {error}")

        rows.sort(key=lambda row: int(row["graph_index"]))
        return rows

    def generate_graphs(
        self,
        choice: EncodingChoice,
    ) -> pd.DataFrame:
        existing = self.existing_graph_metadata(choice.n_vertices)
        if len(existing) >= self.config.n_graphs:
            print(
                f"Grafos n={choice.n_vertices}: "
                f"{self.config.n_graphs} já disponíveis."
            )
            return pd.DataFrame(existing[: self.config.n_graphs])

        accepted = len(existing)
        known_attempts = [
            int(row.get("attempt", 0))
            for row in existing
            if int(row.get("attempt", 0)) > 0
        ]
        attempt = max(known_attempts, default=0)
        probability = min(
            1.0,
            TARGET_AVERAGE_DEGREE
            / max(1, choice.n_vertices - 1),
        )

        mqlib_rows_path = self.output_dir / "mqlib_runs.csv"
        if mqlib_rows_path.is_file():
            try:
                mqlib_rows = pd.read_csv(
                    mqlib_rows_path
                ).to_dict(orient="records")
            except Exception:
                mqlib_rows = []
        else:
            mqlib_rows = []

        print(
            f"\n=== GERAÇÃO: n={choice.n_vertices}, "
            f"q={choice.q_qubits}, k={choice.k_order} ==="
        )
        print(
            f"Retomando com {accepted}/{self.config.n_graphs} "
            "grafos prontos."
        )

        while (
            accepted < self.config.n_graphs
            and attempt < MAX_GRAPH_ATTEMPTS
        ):
            attempt += 1
            graph_seed = stable_seed(
                MASTER_SEED,
                "random_graph",
                choice.n_vertices,
                attempt,
            )
            graph = nx.fast_gnp_random_graph(
                choice.n_vertices,
                probability,
                seed=graph_seed,
                directed=False,
            )
            nx.set_edge_attributes(graph, 1.0, "weight")

            average_degree = (
                2.0 * graph.number_of_edges()
                / max(1, graph.number_of_nodes())
            )

            if graph.number_of_edges() == 0:
                continue
            if average_degree < MIN_AVERAGE_DEGREE:
                continue
            if REQUIRE_CONNECTED and not nx.is_connected(graph):
                continue

            candidate_path = (
                self.graph_dir
                / f"_candidate_n{choice.n_vertices}_a{attempt:06d}.txt"
            )
            save_graph_mqlib(graph, candidate_path)

            best_reference = -math.inf
            reference_source = None

            for heuristic in MQLIB_HEURISTICS:
                for mqlib_seed in MQLIB_SEEDS:
                    result = run_mqlib(
                        executable=self.mqlib_executable,
                        graph_file=candidate_path,
                        heuristic=heuristic,
                        seed=mqlib_seed,
                        runtime_seconds=self.config.mqlib_runtime_seconds,
                    )
                    result.update(
                        {
                            "n_vertices": choice.n_vertices,
                            "q_qubits": choice.q_qubits,
                            "k_order": choice.k_order,
                            "attempt": attempt,
                            "graph_seed": graph_seed,
                        }
                    )
                    mqlib_rows.append(result)

                    if result.get("status") == "ok":
                        objective = float(result["reported_objective"])
                        if objective > best_reference:
                            best_reference = objective
                            reference_source = (
                                f"MQLib:{heuristic}:seed{mqlib_seed}"
                            )

            write_csv_atomic(
                mqlib_rows_path,
                pd.DataFrame(mqlib_rows),
            )

            exact_reference = None
            if choice.n_vertices <= EXACT_MAXCUT_MAX_VERTICES:
                exact_reference, _ = exact_maxcut_gray(graph)
                if exact_reference > best_reference:
                    best_reference = exact_reference
                    reference_source = "exact_gray"

            if not math.isfinite(best_reference) or best_reference <= 0:
                candidate_path.unlink(missing_ok=True)
                continue

            rng = np.random.default_rng(
                stable_seed(
                    MASTER_SEED,
                    "filter_cut",
                    choice.n_vertices,
                    attempt,
                )
            )
            random_bits = rng.integers(
                0,
                2,
                size=choice.n_vertices,
                dtype=np.uint8,
            )
            random_cut = cut_value(random_bits, graph)
            random_ar = random_cut / best_reference

            diagnostic_cuts = [
                cut_value(
                    rng.integers(
                        0,
                        2,
                        size=choice.n_vertices,
                        dtype=np.uint8,
                    ),
                    graph,
                )
                for _ in range(RANDOM_CUT_DIAGNOSTIC_SAMPLES)
            ]

            if random_ar > RANDOM_CUT_AR_MAX:
                candidate_path.unlink(missing_ok=True)
                continue

            graph_index = accepted
            graph_id = (
                f"random_n{choice.n_vertices}_g{graph_index:04d}"
            )
            final_graph_path = self.graph_path(
                choice.n_vertices,
                graph_index,
            )
            candidate_path.replace(final_graph_path)

            metadata = {
                "created_utc": utc_now(),
                "graph_id": graph_id,
                "graph_index": graph_index,
                "n_vertices": choice.n_vertices,
                "q_qubits": choice.q_qubits,
                "k_order": choice.k_order,
                "capacity": choice.capacity,
                "full_basis_size": choice.full_basis_size,
                "attempt": attempt,
                "graph_seed": graph_seed,
                "n_edges": graph.number_of_edges(),
                "average_degree": average_degree,
                "connected": nx.is_connected(graph),
                "initial_reference_cut": float(best_reference),
                "reference_source": reference_source,
                "exact_reference": exact_reference,
                "random_filter_cut": float(random_cut),
                "random_filter_ar": float(random_ar),
                "random_cut_mean": float(np.mean(diagnostic_cuts)),
                "random_cut_std": float(np.std(diagnostic_cuts)),
                "random_cut_median": float(
                    np.median(diagnostic_cuts)
                ),
                "random_cut_best": float(np.max(diagnostic_cuts)),
                "graph_file": str(final_graph_path.resolve()),
            }
            write_json_atomic(
                self.graph_meta_path(
                    choice.n_vertices,
                    graph_index,
                ),
                metadata,
            )

            accepted += 1
            print(
                f"  aceito {accepted}/{self.config.n_graphs}: "
                f"{graph_id}, ref={best_reference:.3f}, "
                f"random AR={random_ar:.3f}"
            )

        rows = self.existing_graph_metadata(choice.n_vertices)
        if len(rows) < self.config.n_graphs:
            raise RuntimeError(
                f"Foram aceitos apenas {len(rows)} grafos para "
                f"n={choice.n_vertices}."
            )

        graph_frame = pd.DataFrame(rows[: self.config.n_graphs])
        self.write_all_graphs_csv()
        return graph_frame

    def write_all_graphs_csv(self) -> None:
        rows = []
        for path in sorted(self.graph_meta_dir.glob("*.json")):
            try:
                rows.append(read_json(path))
            except Exception:
                pass
        if rows:
            write_csv_atomic(
                self.output_dir / "graphs.csv",
                pd.DataFrame(rows),
            )

    def run_choice(
        self,
        choice: EncodingChoice,
    ) -> None:
        alpha, beta, hyperparameter_source = resolve_alpha_beta(
            choice,
            self.project_dir,
        )

        print("\n" + "=" * 90)
        print(
            f"n={choice.n_vertices}: q={choice.q_qubits}, "
            f"k={choice.k_order}, C(q,k)={choice.capacity}, "
            f"2^q={choice.full_basis_size}"
        )
        print(
            f"alpha={alpha:.8g}, beta={beta:.8g}, "
            f"fonte={hyperparameter_source}"
        )
        print("=" * 90)

        if choice.full_basis_size > MAX_FULL_WALSH_COEFFICIENTS:
            raise MemoryError(
                f"A base completa exige {choice.full_basis_size} "
                "coeficientes, acima do limite configurado."
            )

        decision_masks = choose_decision_masks(choice)
        np.save(
            self.decision_mask_dir
            / (
                f"decision_masks_n{choice.n_vertices}_"
                f"q{choice.q_qubits}_k{choice.k_order}.npy"
            ),
            decision_masks,
        )

        encoding_path = self.output_dir / "encoding_choices.csv"
        if encoding_path.is_file():
            encoding_frame = pd.read_csv(encoding_path)
            encoding_frame = encoding_frame[
                pd.to_numeric(
                    encoding_frame["n_vertices"],
                    errors="coerce",
                )
                != choice.n_vertices
            ]
        else:
            encoding_frame = pd.DataFrame()

        encoding_row = pd.DataFrame(
            [
                {
                    "n_vertices": choice.n_vertices,
                    "q_qubits": choice.q_qubits,
                    "k_order": choice.k_order,
                    "symmetric_k": choice.symmetric_k,
                    "capacity": choice.capacity,
                    "unused_capacity": (
                        choice.capacity - choice.n_vertices
                    ),
                    "full_basis_size": choice.full_basis_size,
                    "n_trainable_parameters": (
                        choice.full_basis_size
                    ),
                    "alpha": alpha,
                    "beta": beta,
                    "hyperparameter_source": hyperparameter_source,
                }
            ]
        )
        write_csv_atomic(
            encoding_path,
            pd.concat(
                [encoding_frame, encoding_row],
                ignore_index=True,
            ),
        )

        graph_frame = self.generate_graphs(choice)

        for graph_position, graph_row in enumerate(
            graph_frame.itertuples(index=False),
            start=1,
        ):
            graph_id = str(graph_row.graph_id)
            graph = load_graph_mqlib(
                Path(str(graph_row.graph_file))
            )
            initial_reference_cut = float(
                graph_row.initial_reference_cut
            )

            print(
                f"\n--- Treinando {graph_id} "
                f"({graph_position}/{len(graph_frame)}) ---"
            )

            for seed_position, init_seed in enumerate(
                self.config.training_seeds,
                start=1,
            ):
                identifier = run_identifier(
                    choice=choice,
                    graph_id=graph_id,
                    init_seed=init_seed,
                    alpha=alpha,
                    beta=beta,
                    config=self.config,
                )
                result_json = self.run_json_dir / f"{identifier}.json"
                array_npz = self.array_dir / f"{identifier}.npz"

                if result_json.is_file() and array_npz.is_file():
                    row = read_json(result_json)
                    print(
                        f"  seed {seed_position}/"
                        f"{len(self.config.training_seeds)} "
                        f"({init_seed}) reutilizada: "
                        f"AR={row['final_raw_ar_initial_reference']:.6f}"
                    )
                    continue

                row = train_full_basis(
                    choice=choice,
                    graph_id=graph_id,
                    graph=graph,
                    initial_reference_cut=initial_reference_cut,
                    decision_masks_np=decision_masks,
                    alpha=alpha,
                    beta=beta,
                    init_seed=init_seed,
                    config=self.config,
                    result_json=result_json,
                    array_npz=array_npz,
                )
                print(
                    f"  seed {seed_position}/"
                    f"{len(self.config.training_seeds)} "
                    f"({init_seed}): "
                    f"cut={row['final_raw_cut']:.3f}, "
                    f"AR={row['final_raw_ar_initial_reference']:.6f}, "
                    f"time={row['runtime_s']:.2f}s"
                )

            run_frame = collect_run_rows(self.run_json_dir)
            write_csv_atomic(
                self.output_dir / "training_runs_partial.csv",
                run_frame,
            )

            if UPDATE_PLOT_AFTER_EACH_GRAPH:
                update_progress_plot(
                    n_vertices=choice.n_vertices,
                    run_frame=run_frame,
                    plot_path=(
                        self.plot_dir
                        / f"progress_n{choice.n_vertices}.png"
                    ),
                )

    def finalize(self) -> None:
        self.write_all_graphs_csv()
        graph_csv = self.output_dir / "graphs.csv"
        graph_frame = (
            pd.read_csv(graph_csv)
            if graph_csv.is_file()
            else pd.DataFrame()
        )
        run_frame = collect_run_rows(self.run_json_dir)

        enriched, references, summary = aggregate_final_results(
            graph_frame,
            run_frame,
        )

        if not run_frame.empty:
            write_csv_atomic(
                self.output_dir / "training_runs.csv",
                run_frame,
            )
        if not enriched.empty:
            write_csv_atomic(
                self.output_dir
                / "training_runs_updated_reference.csv",
                enriched,
            )
        if not references.empty:
            write_csv_atomic(
                self.output_dir / "graph_references_updated.csv",
                references,
            )
        if not summary.empty:
            write_csv_atomic(
                self.output_dir / "experiment_1_summary.csv",
                summary,
            )

        for n_vertices in self.config.n_values:
            update_progress_plot(
                n_vertices=n_vertices,
                run_frame=(
                    enriched if not enriched.empty else run_frame
                ),
                plot_path=self.plot_dir / f"progress_n{n_vertices}.png",
            )

        manifest = {
            "created_utc": utc_now(),
            "status": "completed",
            "n_values": self.config.n_values,
            "expected_graphs_per_size": self.config.n_graphs,
            "training_seeds": self.config.training_seeds,
            "expected_runs": (
                len(self.config.n_values)
                * self.config.n_graphs
                * len(self.config.training_seeds)
            ),
            "completed_runs": int(len(run_frame)),
            "output_dir": str(self.output_dir),
            "important_files": [
                "config.json",
                "environment.json",
                "encoding_choices.csv",
                "graphs.csv",
                "mqlib_runs.csv",
                "training_runs.csv",
                "training_runs_updated_reference.csv",
                "graph_references_updated.csv",
                "experiment_1_summary.csv",
                "decision_masks/",
                "run_arrays/",
                "plots/",
            ],
        }
        write_json_atomic(
            self.output_dir / "manifest.json",
            manifest,
        )

        if CREATE_FINAL_ZIP:
            zip_path = (
                self.output_dir.parent
                / f"{self.output_dir.name}_bundle.zip"
            )
            zip_path.unlink(missing_ok=True)
            with zipfile.ZipFile(
                zip_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                for path in self.output_dir.rglob("*"):
                    if path.is_file() and path.name != ".experiment.lock":
                        archive.write(
                            path,
                            arcname=str(
                                path.relative_to(
                                    self.output_dir.parent
                                )
                            ),
                        )
            print(f"\nBundle final: {zip_path}")

    def run(self) -> None:
        for n_vertices in self.config.n_values:
            choice = choose_minimum_qubit_encoding(n_vertices)
            self.run_choice(choice)

        self.finalize()


# =============================================================================
# TESTES INTERNOS
# =============================================================================

def self_tests() -> None:
    print("Executando testes internos...")

    choice = choose_minimum_qubit_encoding(800)
    assert choice.q_qubits == 12
    assert choice.k_order == 6
    assert choice.capacity == 924

    q = 4
    size = 1 << q
    rng = np.random.default_rng(123)
    coefficients_np = rng.normal(size=size)
    masks_np = np.asarray([1, 3, 7, 12], dtype=np.int64)

    coefficients = torch.tensor(
        coefficients_np,
        dtype=torch.float64,
    )
    masks = torch.tensor(masks_np, dtype=torch.long)

    calculated = full_walsh_correlators(
        coefficients,
        masks,
    ).detach().numpy()

    denominator = np.sum(coefficients_np**2)
    direct = np.asarray(
        [
            sum(
                coefficients_np[index]
                * coefficients_np[index ^ int(mask)]
                for index in range(size)
            )
            / denominator
            for mask in masks_np
        ]
    )

    np.testing.assert_allclose(
        calculated,
        direct,
        rtol=1e-10,
        atol=1e-10,
    )

    probabilities = walsh_probability_distribution(
        coefficients
    ).detach().numpy()
    np.testing.assert_allclose(
        probabilities.sum(),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )

    print("Todos os testes internos passaram.")


# =============================================================================
# LINHA DE COMANDO
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimento 1: referência Walsh completa em grafos "
            "MaxCut aleatórios."
        )
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help="Tamanhos de grafo. Padrão: 100 800 2000.",
    )
    parser.add_argument(
        "--graphs",
        type=int,
        default=N_GRAPHS_PER_SIZE,
        help="Número de grafos por tamanho. Padrão: 250.",
    )
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=TRAINING_SEEDS,
        help="Seeds de treinamento. Padrão: 11 22 33 44 55.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help="Passos do Adam. Padrão: 300.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate. Padrão: 1e-3.",
    )
    parser.add_argument(
        "--mqlib-runtime",
        type=float,
        default=MQLIB_RUNTIME_SECONDS,
        help="Tempo por execução do MQLib em segundos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Pasta de saída. Padrão relativo à raiz do projeto: "
            "results/experiment_1_full_walsh_random."
        ),
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Executa apenas os testes internos.",
    )
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Gera os grafos e referências, sem treinar.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    torch.set_num_threads(TORCH_NUM_THREADS)
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)

    self_tests()
    if arguments.self_test_only:
        return

    script_path = Path(__file__).resolve()
    project_dir = script_path.parent

    if arguments.output_dir is None:
        output_dir = project_dir / OUTPUT_RELATIVE_PATH
    else:
        output_dir = Path(arguments.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = project_dir / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig(
        project_dir=project_dir,
        output_dir=output_dir,
        n_values=sorted(set(int(n) for n in arguments.n_values)),
        n_graphs=int(arguments.graphs),
        training_seeds=[
            int(seed) for seed in arguments.training_seeds
        ],
        steps=int(arguments.steps),
        learning_rate=float(arguments.lr),
        eval_every=EVAL_EVERY,
        mqlib_runtime_seconds=float(arguments.mqlib_runtime),
        device=DEVICE_NAME,
        dtype_name=DTYPE_NAME,
    )

    with ExperimentLock(output_dir / ".experiment.lock"):
        runner = ExperimentRunner(config)

        if arguments.graphs_only:
            for n_vertices in config.n_values:
                choice = choose_minimum_qubit_encoding(n_vertices)
                decision_masks = choose_decision_masks(choice)
                np.save(
                    runner.decision_mask_dir
                    / (
                        f"decision_masks_n{choice.n_vertices}_"
                        f"q{choice.q_qubits}_k{choice.k_order}.npy"
                    ),
                    decision_masks,
                )
                runner.generate_graphs(choice)
            runner.write_all_graphs_csv()
            print("\nGeração dos grafos concluída.")
            return

        runner.run()

    print("\nEXPERIMENTO 1 CONCLUÍDO.")
    print(f"Resultados em: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrompido pelo usuário. Os grafos e treinamentos "
            "concluídos foram preservados."
        )
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise
