import torch


def _full_initial_vector(
    n_coeffs,
    seed,
    dtype=torch.float64,
    noise_scale=1e-2,
):
    """
    Deterministic global Walsh initialization:

        a_0 = 1
        a_j ~ N(0, noise_scale^2),  j != 0

    Random numbers are generated in global Walsh-index order.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    a = noise_scale * torch.randn(
        int(n_coeffs),
        dtype=dtype,
        generator=generator,
    )

    a[0] = 1.0

    return a


def initialize_complete(
    n_coeffs,
    variable_masks,
    seed,
    dtype=torch.float64,
    noise_scale=1e-2,
    background_scale=None,
):
    """
    Complete Walsh basis.

    variable_masks and background_scale are retained only
    for API compatibility.
    """

    a = _full_initial_vector(
        n_coeffs=n_coeffs,
        seed=seed,
        dtype=dtype,
        noise_scale=noise_scale,
    )

    return (
        a
        / torch.linalg.vector_norm(a)
    )


def initialize_sparse(
    active_idx,
    base_masks,
    seed,
    n_coeffs,
    dtype=torch.float64,
    noise_scale=1e-2,
):
    """
    Sparse Walsh support.

    The same global Walsh mask receives the same pre-normalisation
    random value for a fixed seed, independently of support B.

    base_masks is retained only for API compatibility.
    """

    n_coeffs = int(n_coeffs)

    if active_idx.ndim != 1:
        raise RuntimeError(
            "active_idx must be one-dimensional."
        )

    if len(active_idx) == 0:
        raise RuntimeError(
            "active_idx cannot be empty."
        )

    if int(active_idx.min()) < 0:
        raise RuntimeError(
            "Negative Walsh index."
        )

    if int(active_idx.max()) >= n_coeffs:
        raise RuntimeError(
            "Walsh index exceeds n_coeffs."
        )

    zero_count = int(
        torch.sum(active_idx == 0).item()
    )

    if zero_count != 1:
        raise RuntimeError(
            "Expected exactly one Walsh mask 0 "
            "in the active support."
        )

    full = _full_initial_vector(
        n_coeffs=n_coeffs,
        seed=seed,
        dtype=dtype,
        noise_scale=noise_scale,
    )

    a = full[active_idx].clone()

    return (
        a
        / torch.linalg.vector_norm(a)
    )
