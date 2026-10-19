from functools import lru_cache
import hashlib

import numpy as np
import torch

from walsh_pce.initialization import initialize_sparse

from walsh_pce.core import (
    DTYPE,
    decision_masks,
    pce_loss,
    cut_from_correlators,
)


MASTER_SUPPORT_SEED = 20260909


def stable_seed(*parts):
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


@lru_cache(maxsize=None)
def base_masks_for_support(n, q, k, B):

    N = 2**q

    rng = np.random.default_rng(
        stable_seed(
            MASTER_SUPPORT_SEED,
            "base_masks",
            n,
            q,
            k,
        )
    )

    others = np.arange(
        1,
        N,
        dtype=np.int64,
    )

    rng.shuffle(others)

    base_order = np.concatenate(
        ([0], others)
    )

    return torch.tensor(
        base_order[:B],
        dtype=torch.long,
    )


@lru_cache(maxsize=None)
def support_indices(n, q, k, B):
    """
    Nested paired support:

        A(B,S) = B U {b xor S_i}.
    """

    masks = decision_masks(
        q=q,
        k=k,
        m=n,
    ).numpy()

    base_masks = base_masks_for_support(
        n=n,
        q=q,
        k=k,
        B=B,
    ).numpy()

    active = set()

    for b in base_masks:

        b = int(b)

        active.add(b)

        for S in masks:
            active.add(
                b ^ int(S)
            )

    return torch.tensor(
        sorted(active),
        dtype=torch.long,
    )



def train_sparse(
    graph,
    q,
    k,
    B,
    alpha,
    beta,
    seed,
    epochs=500,
    lr=1e-3,
):
    masks = decision_masks(
        q=q,
        k=k,
        m=graph.n,
    )

    active_idx = support_indices(
        n=graph.n,
        q=q,
        k=k,
        B=B,
    )

    base_masks = base_masks_for_support(
        n=graph.n,
        q=q,
        k=k,
        B=B,
    )

    # Only active coefficients are trainable.
    a = initialize_sparse(
        active_idx=active_idx,
        base_masks=base_masks,
        seed=seed,
        n_coeffs=2**q,
        dtype=DTYPE,
    )

    a.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [a],
        lr=lr,
    )

    N = 2**q

    for _ in range(epochs):

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

        loss, _ = pce_loss(
            a=full,
            graph=graph,
            masks=masks,
            alpha=alpha,
            beta=beta,
        )

        loss.backward()

        optimizer.step()

        # Keep the Walsh state normalised.
        with torch.no_grad():
            a /= torch.linalg.vector_norm(a)

    with torch.no_grad():

        full = torch.zeros(
            N,
            dtype=DTYPE,
        ).scatter(
            0,
            active_idx,
            a,
        )

        final_loss, c = pce_loss(
            a=full,
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
        "active_size": len(active_idx),
    }
