from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from walsh_pce.initialization import initialize_complete


DTYPE = torch.float64


# m -> (q, k)
SIZE_CONFIG = {
    100:  (9, 4),
    200:  (10, 5),
    400:  (11, 5),
    500:  (12, 6),
    800:  (12, 6),
    2000: (14, 7),
    7000: (16, 8),
}


@dataclass(frozen=True)
class GraphData:
    n: int
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor

    @property
    def num_edges(self):
        return len(self.u)


@lru_cache(maxsize=None)
def load_graph(path_str):
    path = Path(path_str)

    with path.open("r") as f:
        first = f.readline().split()
        n = int(first[0])
        m = int(first[1])

        data = np.loadtxt(f)

    if data.ndim == 1:
        data = data[None, :]

    if len(data) != m:
        raise RuntimeError(
            f"{path}: header says {m} edges, found {len(data)}"
        )

    u = torch.tensor(
        data[:, 0].astype(np.int64) - 1,
        dtype=torch.long,
    )

    v = torch.tensor(
        data[:, 1].astype(np.int64) - 1,
        dtype=torch.long,
    )

    w = torch.tensor(
        data[:, 2],
        dtype=DTYPE,
    )

    # Stage 02 is intentionally restricted to the
    # unit-weight random-graph bank.
    if not torch.all(w == 1.0):
        raise RuntimeError(
            f"Stage 02 expects unit-weight graphs: {path}"
        )

    return GraphData(
        n=n,
        u=u,
        v=v,
        w=w,
    )


@lru_cache(maxsize=None)
def decision_masks(q, k, m):
    """
    First m Hamming-weight-k masks in lexicographic
    order of their qubit supports.
    """
    masks = []

    for support in combinations(range(q), k):
        mask = 0

        for bit in support:
            mask |= 1 << bit

        masks.append(mask)

        if len(masks) == m:
            break

    if len(masks) != m:
        raise RuntimeError(
            f"Not enough masks: q={q}, k={k}, m={m}"
        )

    return torch.tensor(masks, dtype=torch.long)


def fwht(x):
    """
    Unnormalised Fast Walsh-Hadamard Transform.

    H(H(x)) = N x
    """
    n = x.numel()

    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("FWHT length must be a power of two")

    y = x
    h = 1

    while h < n:
        y2 = y.reshape(-1, 2 * h)

        left = y2[:, :h]
        right = y2[:, h:]

        y = torch.cat(
            (left + right, left - right),
            dim=1,
        ).reshape(n)

        h *= 2

    return y


def complete_correlators(a, masks):
    """
    XOR autocorrelation

        c_S = sum_l a_l a_{l xor S} / sum_l a_l^2

    evaluated for the complete Walsh basis using FWHT.
    """
    n = a.numel()

    spectrum = fwht(a)

    autocorr = fwht(
        spectrum * spectrum
    ) / n

    denominator = torch.sum(a * a)

    autocorr = autocorr / denominator

    return autocorr[masks]


def pce_loss(
    a,
    graph,
    masks,
    alpha,
    beta,
):
    c = complete_correlators(a, masks)

    s = torch.tanh(alpha * c)

    energy = torch.sum(
        graph.w
        * s[graph.u]
        * s[graph.v]
    )

    # PCE scale for connected, unit-weight graphs:
    #
    # nu = |E|/2 + (|V|-1)/4
    nu = (
        0.5 * graph.num_edges
        + 0.25 * (graph.n - 1)
    )

    regularizer = (
        beta
        * nu
        * torch.mean(s * s) ** 2
    )

    loss = energy + regularizer

    return loss, c


def cut_from_correlators(graph, c):
    """
    Only relative signs matter for MaxCut.
    """
    side = c >= 0

    crossed = (
        side[graph.u]
        != side[graph.v]
    )

    return float(
        torch.sum(graph.w[crossed]).item()
    )


def train_complete(
    graph,
    q,
    k,
    alpha,
    beta,
    seed,
    epochs=150,
    lr=1e-3,
):
    masks = decision_masks(
        q=q,
        k=k,
        m=graph.n,
    )

    a = initialize_complete(
        n_coeffs=2 ** q,
        variable_masks=masks,
        seed=seed,
        dtype=DTYPE,
    )

    a.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [a],
        lr=lr,
    )

    for _ in range(epochs):

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, _ = pce_loss(
            a=a,
            graph=graph,
            masks=masks,
            alpha=alpha,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        # Projection back onto the unit sphere.
        with torch.no_grad():
            a /= torch.linalg.vector_norm(a)

    with torch.no_grad():

        final_loss, c = pce_loss(
            a=a,
            graph=graph,
            masks=masks,
            alpha=alpha,
            beta=beta,
        )

        cut = cut_from_correlators(
            graph,
            c,
        )

        norm = float(
            torch.linalg.vector_norm(a).item()
        )

    return {
        "cut": cut,
        "loss": float(final_loss.item()),
        "norm": norm,
    }
