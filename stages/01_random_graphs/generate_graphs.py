from pathlib import Path
import csv
import hashlib
import random

import networkx as nx


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MASTER_SEED = 20260623

SIZES = [100, 200, 400, 500, 800, 2000, 7000]

GRAPHS_PER_SIZE = 250

MIN_AVG_DEGREE = 3.0

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data" / "graphs"
SUMMARY_FILE = OUTPUT_ROOT / "graphs.csv"


# ============================================================
# SEEDS DETERMINÍSTICAS
# ============================================================

def stable_seed(*parts):
    """
    Reproduz a derivação determinística usada no pipeline anterior:
    strings separadas por '|', SHA256, primeiros 16 hex digits,
    módulo 2^32 - 1.
    """
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


# ============================================================
# REPAIR DE CONECTIVIDADE
# ============================================================

def connect_components(G, n, attempt):
    """
    Se G possuir c componentes, adiciona exatamente c-1 arestas.

    Os componentes são embaralhados de forma determinística e
    conectados consecutivamente.
    """
    components = [
        sorted(component)
        for component in nx.connected_components(G)
    ]

    n_components = len(components)

    if n_components <= 1:
        return G, 0, n_components

    rng = random.Random(
        stable_seed(
            MASTER_SEED,
            "connect_components",
            n,
            attempt,
        )
    )

    order = list(range(n_components))
    rng.shuffle(order)

    bridges = 0

    for left_idx, right_idx in zip(order[:-1], order[1:]):
        left_component = components[left_idx]
        right_component = components[right_idx]

        u = rng.choice(left_component)
        v = rng.choice(right_component)

        G.add_edge(u, v, weight=1.0)
        bridges += 1

    return G, bridges, n_components


# ============================================================
# SAVE
# ============================================================

def save_graph(G, path):
    """
    Formato simples compatível com instâncias MaxCut/MQLib:

    primeira linha:
        number_of_vertices number_of_edges

    linhas seguintes:
        vertex_i vertex_j weight

    Os vértices são salvos em indexação 1-based.
    """
    edges = sorted(
        (min(u, v), max(u, v), data.get("weight", 1.0))
        for u, v, data in G.edges(data=True)
    )

    with path.open("w") as f:
        f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")

        for u, v, w in edges:
            f.write(f"{u + 1} {v + 1} {w:g}\n")


# ============================================================
# GERAÇÃO
# ============================================================

def generate_bank():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    total_target = len(SIZES) * GRAPHS_PER_SIZE
    total_saved = 0

    for n in SIZES:

        p = min(1.0, 6.0 / (n - 1))

        size_dir = OUTPUT_ROOT / f"n{n}"
        size_dir.mkdir(parents=True, exist_ok=True)

        accepted = 0
        attempt = 1

        print()
        print("=" * 70)
        print(f"n = {n}")
        print(f"p = {p:.12f}")
        print("=" * 70)

        while accepted < GRAPHS_PER_SIZE:

            graph_seed = stable_seed(
                MASTER_SEED,
                "random_graph_v2",
                n,
                attempt,
            )

            G = nx.fast_gnp_random_graph(
                n=n,
                p=p,
                seed=graph_seed,
                directed=False,
            )

            # Todas as arestas originais têm peso +1.
            nx.set_edge_attributes(G, 1.0, "weight")

            edges_before = G.number_of_edges()

            if edges_before == 0:
                attempt += 1
                continue

            # IMPORTANTE:
            # grau médio calculado ANTES do repair, como no pipeline antigo.
            avg_degree_before = (
                2.0 * edges_before / n
            )

            if avg_degree_before < MIN_AVG_DEGREE:
                attempt += 1
                continue

            G, bridges, components_before = connect_components(
                G,
                n=n,
                attempt=attempt,
            )

            if not nx.is_connected(G):
                raise RuntimeError(
                    f"Connectivity repair failed for n={n}, attempt={attempt}"
                )

            graph_name = f"g{accepted:04d}"
            relative_path = Path(f"n{n}") / f"{graph_name}.txt"
            output_path = OUTPUT_ROOT / relative_path

            save_graph(G, output_path)

            rows.append(
                {
                    "graph_id": graph_name,
                    "n": n,
                    "p": p,
                    "attempt": attempt,
                    "graph_seed": graph_seed,
                    "edges_before_repair": edges_before,
                    "edges_after_repair": G.number_of_edges(),
                    "avg_degree_before_repair": avg_degree_before,
                    "components_before_repair": components_before,
                    "bridge_edges_added": bridges,
                    "path": str(relative_path),
                }
            )

            accepted += 1
            total_saved += 1

            if accepted % 25 == 0 or accepted == GRAPHS_PER_SIZE:
                print(
                    f"[{accepted:3d}/{GRAPHS_PER_SIZE}] "
                    f"global {total_saved}/{total_target}"
                )

            attempt += 1

    fieldnames = [
        "graph_id",
        "n",
        "p",
        "attempt",
        "graph_seed",
        "edges_before_repair",
        "edges_after_repair",
        "avg_degree_before_repair",
        "components_before_repair",
        "bridge_edges_added",
        "path",
    ]

    with SUMMARY_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("GRAPH BANK COMPLETE")
    print("=" * 70)
    print(f"Graphs: {len(rows)}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    generate_bank()
