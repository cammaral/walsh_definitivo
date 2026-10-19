"""Extensões de medição para o Walsh Series Loader de ``quantum_sub.py``.

O módulo preserva o circuito original e acrescenta:
- suporte direto aos coeficientes esparsos salvos na Etapa 5;
- correção explícita da ordem de bits PennyLane -> máscaras LSB do projeto;
- circuito equivalente com cancelamento de CNOTs em ordem Gray;
- statevector, shots, pós-seleção e ruído por trajetórias de Pauli;
- probabilidades, correlatores e contagem lógica de recursos.

Nenhuma função deste módulo treina parâmetros.
"""

from __future__ import annotations

from tqdm import tqdm

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml

from quantum_sub import run_subcircuit


@dataclass(frozen=True)
class LoaderResources:
    n_data_qubits: int
    n_active_coeffs: int
    n_nonzero_walsh_terms: int
    original_toffoli_count: int
    original_controlled_rz_count: int
    optimized_cnot_count: int
    optimized_controlled_rz_count: int
    hadamard_count: int
    phase_shift_count: int
    optimized_serial_gate_count: int


def fwht_numpy(values: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard não normalizada, na mesma convenção de ``pce.py``."""
    x = np.asarray(values, dtype=np.float64).copy()
    if x.ndim != 1:
        raise ValueError("FWHT espera vetor unidimensional.")
    n = int(x.size)
    if n <= 0 or (n & (n - 1)):
        raise ValueError("O comprimento deve ser potência de dois.")

    h = 1
    while h < n:
        view = x.reshape(-1, 2 * h)
        left = view[:, :h].copy()
        right = view[:, h : 2 * h].copy()
        view[:, :h] = left + right
        view[:, h : 2 * h] = left - right
        h *= 2
    return x


def dense_coefficients(
    coeffs: Sequence[float], coeff_masks: Sequence[int], n: int
) -> np.ndarray:
    """Espalha os coeficientes esparsos em um vetor denso de tamanho ``2**n``."""
    coeffs_array = np.asarray(coeffs, dtype=np.float64)
    masks_array = np.asarray(coeff_masks, dtype=np.int64)
    if coeffs_array.ndim != 1 or masks_array.ndim != 1:
        raise ValueError("coeffs e coeff_masks devem ser vetores 1D.")
    if coeffs_array.size != masks_array.size:
        raise ValueError("coeffs e coeff_masks devem ter o mesmo tamanho.")
    dimension = 1 << int(n)
    if np.any(masks_array < 0) or np.any(masks_array >= dimension):
        raise ValueError("Há máscara fora do espaço de n qubits.")
    if np.unique(masks_array).size != masks_array.size:
        raise ValueError("coeff_masks contém índices repetidos.")

    dense = np.zeros(dimension, dtype=np.float64)
    dense[masks_array] = coeffs_array
    return dense


def bit_reverse(value: int, n: int) -> int:
    result = 0
    for _ in range(int(n)):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def bit_reverse_permutation(n: int) -> np.ndarray:
    """Permutação entre a ordem do statevector e a ordem LSB das máscaras PCE."""
    dimension = 1 << int(n)
    return np.fromiter(
        (bit_reverse(index, n) for index in range(dimension)),
        dtype=np.int64,
        count=dimension,
    )


def reorder_register_state_to_project_order(raw_register_state: np.ndarray, n: int) -> np.ndarray:
    """Converte ``|wire 1 ... wire n>`` para o inteiro LSB usado em ``pce.py``.

    No circuito enviado, o bit 0 da máscara Walsh atua no wire 1. Já o vetor de
    estado PennyLane enumera o wire 1 como o bit mais significativo do índice.
    Portanto, a comparação com a FWHT clássica exige reversão dos n bits.
    """
    state = np.asarray(raw_register_state, dtype=np.complex128)
    if state.shape != (1 << int(n),):
        raise ValueError("Tamanho incompatível com n.")
    return state[bit_reverse_permutation(n)]


def project_order_indices_from_samples(samples: np.ndarray, n: int) -> np.ndarray:
    """Converte colunas dos wires 1..n em inteiros com wire 1 = bit 0."""
    array = np.asarray(samples, dtype=np.int8)
    if array.ndim != 2 or array.shape[1] != n + 1:
        raise ValueError("samples deve ter shape (shots, n+1).")
    data = array[:, 1:]
    powers = (1 << np.arange(n, dtype=np.int64))[None, :]
    return np.sum(data.astype(np.int64) * powers, axis=1, dtype=np.int64)


def target_walsh_outputs(dense_coeffs: np.ndarray) -> dict[str, np.ndarray | float]:
    """Estado e distribuição ideais definidos pelos coeficientes Walsh."""
    dense = np.asarray(dense_coeffs, dtype=np.float64)
    function_values = fwht_numpy(dense)
    norm = float(np.linalg.norm(function_values))
    if norm <= 0.0:
        raise FloatingPointError("A função Walsh é nula.")
    state = function_values.astype(np.complex128) / norm
    probabilities = np.square(function_values) / (norm * norm)
    return {
        "function_values": function_values,
        "state": state,
        "probabilities": probabilities,
        "max_abs_function": float(np.max(np.abs(function_values))),
    }


def analytical_loader_outputs(dense_coeffs: np.ndarray, er: float) -> dict[str, np.ndarray | float]:
    """Saída exata do loader controlado, antes de qualquer erro de amostragem."""
    dense = np.asarray(dense_coeffs, dtype=np.float64)
    n = int(round(np.log2(dense.size)))
    if (1 << n) != dense.size:
        raise ValueError("dense_coeffs deve ter tamanho potência de dois.")

    function_values = fwht_numpy(dense)
    phases = float(er) * function_values
    # Amplitude do ramo ancilla=1 depois de H e PhaseShift(-pi/2).
    unnormalized = (-1j) * (1.0 - np.exp(-1j * phases)) / (2.0 * np.sqrt(dense.size))
    success_probability = float(np.vdot(unnormalized, unnormalized).real)
    if success_probability <= 0.0:
        raise FloatingPointError("Probabilidade de sucesso numericamente nula.")
    state = unnormalized / np.sqrt(success_probability)
    probabilities = np.abs(state) ** 2
    return {
        "function_values": function_values,
        "success_probability": success_probability,
        "state": state,
        "probabilities": probabilities,
    }


def _inverse_gray(gray_value: int) -> int:
    value = 0
    gray_value = int(gray_value)
    while gray_value:
        value ^= gray_value
        gray_value >>= 1
    return value


def build_gray_schedule(coeff_masks: Iterable[int], n: int) -> list[tuple[int, list[tuple[int, int]]]]:
    """Agrupa por target/MSB e ordena os controles pela posição no Gray code."""
    groups: dict[int, list[tuple[int, int]]] = {}
    dimension = 1 << int(n)
    for raw_mask in coeff_masks:
        mask = int(raw_mask)
        if not (0 <= mask < dimension):
            raise ValueError(f"Máscara {mask} fora do intervalo.")
        if mask == 0:
            continue
        target_bit = mask.bit_length() - 1
        control_mask = mask ^ (1 << target_bit)
        groups.setdefault(target_bit, []).append((mask, control_mask))

    schedule: list[tuple[int, list[tuple[int, int]]]] = []
    for target_bit in sorted(groups):
        ordered = sorted(
            groups[target_bit],
            key=lambda item: (_inverse_gray(item[1]), item[1], item[0]),
        )
        schedule.append((target_bit, ordered))
    return schedule


def _set_bits(mask: int) -> list[int]:
    bits: list[int] = []
    value = int(mask)
    while value:
        lsb = value & -value
        bits.append(lsb.bit_length() - 1)
        value ^= lsb
    return bits


def _apply_one_qubit_pauli_error(rng: np.random.Generator | None, probability: float, wire: int) -> None:
    if rng is None or probability <= 0.0 or rng.random() >= probability:
        return
    choice = int(rng.integers(0, 3))
    (qml.PauliX, qml.PauliY, qml.PauliZ)[choice](wires=wire)


def _apply_two_qubit_pauli_error(
    rng: np.random.Generator | None,
    probability: float,
    wire_a: int,
    wire_b: int,
) -> None:
    """Uma trajetória do canal despolarizante de dois qubits.

    Com probabilidade ``probability``, aplica uniformemente um dos 15 produtos
    de Pauli não identidade.
    """
    if rng is None or probability <= 0.0 or rng.random() >= probability:
        return
    # 0=I, 1=X, 2=Y, 3=Z; exclui II.
    choice = int(rng.integers(1, 16))
    pauli_a, pauli_b = divmod(choice, 4)
    operations = {1: qml.PauliX, 2: qml.PauliY, 3: qml.PauliZ}
    if pauli_a:
        operations[pauli_a](wires=wire_a)
    if pauli_b:
        operations[pauli_b](wires=wire_b)


def apply_loader_optimized(
    dense_coeffs: np.ndarray,
    coeff_masks: Sequence[int],
    n: int,
    er: float,
    *,
    gate_noise_rng: np.random.Generator | None = None,
    p1: float = 0.0,
    p2: float = 0.0,
) -> None:
    """Aplica o mesmo WSL do arquivo original com cancelamento de CNOTs.

    A rede de paridade é aplicada sem controle da ancilla e somente a rotação
    RZ é controlada. Isso é exatamente equivalente a controlar o bloco inteiro:
    no ramo ancilla=0, a rede e sua inversa se cancelam; no ramo ancilla=1,
    obtém-se o bloco Walsh desejado. A ordem Gray cancela CNOTs entre termos
    consecutivos com o mesmo target.
    """
    dense = np.asarray(dense_coeffs, dtype=np.float64)
    masks = [int(mask) for mask in coeff_masks]
    if dense.shape != (1 << int(n),):
        raise ValueError("dense_coeffs incompatível com n.")

    ancilla = 0
    for wire in range(n + 1):
        qml.Hadamard(wires=wire)
        _apply_one_qubit_pauli_error(gate_noise_rng, p1, wire)

    if 0 in masks:
        qml.PhaseShift(-float(er) * float(dense[0]), wires=ancilla)
        _apply_one_qubit_pauli_error(gate_noise_rng, p1, ancilla)

    for target_bit, terms in build_gray_schedule(masks, n):
        target_wire = 1 + target_bit
        previous_controls = 0

        for walsh_mask, control_mask in terms:
            transition = previous_controls ^ control_mask
            for control_bit in _set_bits(transition):
                control_wire = 1 + control_bit
                qml.CNOT(wires=[control_wire, target_wire])
                _apply_two_qubit_pauli_error(
                    gate_noise_rng, p2, control_wire, target_wire
                )

            angle = 2.0 * float(er) * float(dense[walsh_mask])
            qml.ctrl(qml.RZ, control=ancilla)(angle, wires=target_wire)
            _apply_two_qubit_pauli_error(
                gate_noise_rng, p2, ancilla, target_wire
            )
            previous_controls = control_mask

        # Retorna a paridade ao estado original antes de mudar de target.
        for control_bit in _set_bits(previous_controls):
            control_wire = 1 + control_bit
            qml.CNOT(wires=[control_wire, target_wire])
            _apply_two_qubit_pauli_error(
                gate_noise_rng, p2, control_wire, target_wire
            )

    qml.Hadamard(wires=ancilla)
    _apply_one_qubit_pauli_error(gate_noise_rng, p1, ancilla)
    qml.PhaseShift(-np.pi / 2.0, wires=ancilla)
    _apply_one_qubit_pauli_error(gate_noise_rng, p1, ancilla)


def choose_device(
    n_wires: int,
    *,
    shots: int | None,
    seed: int | None,
    preference: Sequence[str] = ("lightning.qubit", "default.qubit"),
):
    errors: list[str] = []
    for device_name in preference:
        try:
            device = qml.device(device_name, wires=n_wires, seed=seed)
            return device_name, device
        except Exception as error:  # pragma: no cover - depende da instalação local
            errors.append(f"{device_name}: {type(error).__name__}: {error}")
    raise RuntimeError("Nenhum simulador PennyLane disponível:\n" + "\n".join(errors))


def _postselect_and_reorder(full_state: np.ndarray, n: int, tol: float = 1e-18):
    state = np.asarray(full_state, dtype=np.complex128).reshape(2, 1 << n)
    raw_register = state[1]
    success_probability = float(np.vdot(raw_register, raw_register).real)
    if success_probability <= tol:
        raise FloatingPointError("Probabilidade de pós-seleção numericamente nula.")
    raw_postselected = raw_register / np.sqrt(success_probability)
    project_state = reorder_register_state_to_project_order(raw_postselected, n)
    return success_probability, project_state, np.abs(project_state) ** 2


def run_loader_statevector(
    dense_coeffs: np.ndarray,
    coeff_masks: Sequence[int],
    n: int,
    er: float,
    *,
    device_preference: Sequence[str] = ("lightning.qubit", "default.qubit"),
) -> dict[str, object]:
    device_name, device = choose_device(
        n + 1, shots=None, seed=None, preference=device_preference
    )

    @qml.qnode(device)
    def circuit():
        apply_loader_optimized(dense_coeffs, coeff_masks, n, er)
        return qml.state()

    full_state = np.asarray(circuit(), dtype=np.complex128)
    success_probability, state, probabilities = _postselect_and_reorder(full_state, n)
    return {
        "device": device_name,
        "success_probability_ancilla_1": success_probability,
        "postselected_state": state,
        "postselected_probabilities": probabilities,
        "full_state": full_state,
    }


def run_original_loader_statevector(
    dense_coeffs: np.ndarray,
    coeff_masks: Sequence[int],
    n: int,
    er: float,
) -> dict[str, object]:
    """Executa diretamente o circuito enviado e corrige apenas a ordem de bits."""
    result = run_subcircuit(
        avec=np.asarray(dense_coeffs, dtype=np.float64),
        n=n,
        list_j=[int(mask) for mask in coeff_masks],
        er=float(er),
        return_full_state=True,
    )
    raw_state = np.asarray(result["postselected_state"], dtype=np.complex128)
    state = reorder_register_state_to_project_order(raw_state, n)
    return {
        "success_probability_ancilla_1": float(result["success_probability_ancilla_1"]),
        "postselected_state": state,
        "postselected_probabilities": np.abs(state) ** 2,
        "full_state": np.asarray(result["full_state"], dtype=np.complex128),
    }


def run_loader_samples(
    dense_coeffs: np.ndarray,
    coeff_masks: Sequence[int],
    n: int,
    er: float,
    shots: int,
    seed: int,
    *,
    device_preference: Sequence[str] = ("lightning.qubit", "default.qubit"),
    gate_noise_seed: int | None = None,
    p1: float = 0.0,
    p2: float = 0.0,
) -> dict[str, object]:
    if shots < 1:
        raise ValueError("shots deve ser positivo.")
    device_name, device = choose_device(
        n + 1, shots=int(shots), seed=int(seed), preference=device_preference
    )
    noise_rng = (
        np.random.default_rng(int(gate_noise_seed))
        if gate_noise_seed is not None and (p1 > 0.0 or p2 > 0.0)
        else None
    )

    @qml.qnode(device)
    def circuit():
        apply_loader_optimized(
            dense_coeffs,
            coeff_masks,
            n,
            er,
            gate_noise_rng=noise_rng,
            p1=float(p1),
            p2=float(p2),
        )
        return qml.sample(wires=list(range(n + 1)))

    sampled_circuit = qml.set_shots(circuit, shots=int(shots))
    samples = np.asarray(sampled_circuit(), dtype=np.int8)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return {"device": device_name, "samples": samples}


def apply_readout_error(samples: np.ndarray, probability: float, seed: int) -> np.ndarray:
    array = np.asarray(samples, dtype=np.int8).copy()
    if not (0.0 <= probability <= 1.0):
        raise ValueError("A probabilidade de readout deve estar em [0,1].")
    if probability == 0.0:
        return array
    rng = np.random.default_rng(int(seed))
    flips = rng.random(array.shape) < float(probability)
    array ^= flips.astype(np.int8)
    return array


def postselected_histogram(samples: np.ndarray, n: int) -> dict[str, np.ndarray | int | float]:
    array = np.asarray(samples, dtype=np.int8)
    if array.ndim != 2 or array.shape[1] != n + 1:
        raise ValueError("samples deve ter shape (shots, n+1).")
    total_shots = int(array.shape[0])
    accepted = array[array[:, 0] == 1]
    accepted_shots = int(accepted.shape[0])
    dimension = 1 << int(n)
    if accepted_shots == 0:
        counts = np.zeros(dimension, dtype=np.int64)
        probabilities = np.zeros(dimension, dtype=np.float64)
    else:
        indices = project_order_indices_from_samples(accepted, n)
        counts = np.bincount(indices, minlength=dimension).astype(np.int64)
        probabilities = counts.astype(np.float64) / accepted_shots
    return {
        "total_shots": total_shots,
        "accepted_shots": accepted_shots,
        "observed_success_probability": accepted_shots / max(total_shots, 1),
        "counts": counts,
        "probabilities": probabilities,
    }


def correlators_from_probabilities(
    probabilities: np.ndarray, variable_masks: Sequence[int]
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    transformed = fwht_numpy(probs)
    masks = np.asarray(variable_masks, dtype=np.int64)
    return transformed[masks]


def state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    a = np.asarray(state_a, dtype=np.complex128)
    b = np.asarray(state_b, dtype=np.complex128)
    return float(abs(np.vdot(a, b)) ** 2)


def total_variation(prob_a: np.ndarray, prob_b: np.ndarray) -> float:
    a = np.asarray(prob_a, dtype=np.float64)
    b = np.asarray(prob_b, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(a - b)))


def resource_counts(coeff_masks: Sequence[int], n: int) -> LoaderResources:
    masks = [int(mask) for mask in coeff_masks]
    nonzero = [mask for mask in masks if mask != 0]
    original_toffoli = sum(2 * max(mask.bit_count() - 1, 0) for mask in nonzero)

    optimized_cnot = 0
    for _target_bit, terms in build_gray_schedule(masks, n):
        previous = 0
        for _walsh_mask, controls in terms:
            optimized_cnot += (previous ^ controls).bit_count()
            previous = controls
        optimized_cnot += previous.bit_count()

    hadamards = n + 2
    phase_shifts = 1 + int(0 in masks)
    controlled_rz = len(nonzero)
    serial = optimized_cnot + controlled_rz + hadamards + phase_shifts
    return LoaderResources(
        n_data_qubits=int(n),
        n_active_coeffs=len(masks),
        n_nonzero_walsh_terms=len(nonzero),
        original_toffoli_count=int(original_toffoli),
        original_controlled_rz_count=int(controlled_rz),
        optimized_cnot_count=int(optimized_cnot),
        optimized_controlled_rz_count=int(controlled_rz),
        hadamard_count=int(hadamards),
        phase_shift_count=int(phase_shifts),
        optimized_serial_gate_count=int(serial),
    )


def run_pauli_trajectory_samples(
    dense_coeffs: np.ndarray,
    coeff_masks: Sequence[int],
    n: int,
    er: float,
    *,
    trajectories: int,
    shots_per_trajectory: int,
    base_seed: int,
    p1: float,
    p2: float,
    p_readout: float,
    device_preference: Sequence[str] = ("lightning.qubit", "default.qubit"),
) -> dict[str, object]:
    """Aproxima canais despolarizantes por trajetórias puras de erros de Pauli."""
    if trajectories < 1 or shots_per_trajectory < 1:
        raise ValueError("trajectories e shots_per_trajectory devem ser positivos.")
    batches: list[np.ndarray] = []
    devices: list[str] = []
    for trajectory in tqdm(
        range(trajectories),
        total=trajectories,
        desc="noise trajectories",
        unit="traj",
    ):
        trajectory_seed = int(base_seed) + 100_003 * trajectory
        result = run_loader_samples(
            dense_coeffs,
            coeff_masks,
            n,
            er,
            shots=shots_per_trajectory,
            seed=trajectory_seed + 17,
            device_preference=device_preference,
            gate_noise_seed=trajectory_seed + 31,
            p1=p1,
            p2=p2,
        )
        noisy = apply_readout_error(
            result["samples"], p_readout, seed=trajectory_seed + 47
        )
        batches.append(noisy)
        devices.append(str(result["device"]))
    return {
        "device": ",".join(sorted(set(devices))),
        "samples": np.concatenate(batches, axis=0),
        "trajectories": int(trajectories),
        "shots_per_trajectory": int(shots_per_trajectory),
    }
