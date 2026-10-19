#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuração oficial da Etapa 1 Walsh/PCE — versão 2 segura."""

from __future__ import annotations

from pathlib import Path

# Escalas do estudo.
N_VALUES = [100, 200, 400, 500, 800, 2000, 7000]
N_GRAPHS_PER_SIZE = 250
CALIBRATION_GRAPHS_PER_SIZE = 10

# Dez inicializações independentes, iguais na calibração e na rodada final.
TRAINING_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

# Otimização.
CALIBRATION_STEPS = 150
FINAL_STEPS = 500
LEARNING_RATE = 1.0e-3
EVAL_EVERY = 10

# Para um mesmo k, usa-se o mesmo alpha absoluto e o mesmo beta.
# O alpha é definido por multiplicador * maior q do grupo daquele k.
ALPHA_Q_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0]
BETA_VALUES = [0.0, 0.05, 0.1, 0.5]

# Grafos: G(n,p), grau médio alvo 6, seguidos de conexão mínima entre
# componentes. Isso evita o colapso em n=7000 sem tornar o grafo denso.
TARGET_AVERAGE_DEGREE = 6.0
MIN_AVERAGE_DEGREE = 3.0
CONNECTIVITY_MODE = "minimal_bridges"
RANDOM_CUT_AR_MAX = 0.82
RANDOM_CUT_DIAGNOSTIC_SAMPLES = 100
MAX_GRAPH_ATTEMPTS_PER_SIZE = 20_000

# Referência MQLib. Mantém as mesmas quatro heurísticas do projeto original,
# duas seeds e 2 s por execução. São 8 execuções por grafo, isto é, até
# 16 s de orçamento nominal de MQLib por instância.
MQLIB_HEURISTICS = [
    "BURER2002",
    "FESTA2002G",
    "FESTA2002VNS",
    "LAGUNA2009HCE",
]
MQLIB_SEEDS = [101, 202]
MQLIB_RUNTIME_SECONDS = 2.0

# Saídas. O caminho é ancorado na raiz real do projeto, nunca na pasta em que
# o pacote foi extraído.
OUTPUT_ROOT = Path("results") / "stage1_full_walsh_v2"
KEEP_FINAL_ARRAYS = "best"  # all, best ou none

# Segurança/reprodutibilidade.
PIPELINE_VERSION = 2
GRAPH_BANK_VERSION = 2
MASTER_SEED = 20260623
SAVE_PROGRESS_EVERY_RUNS = 25
