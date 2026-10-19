import pennylane as qml
import pennylane.numpy as np


# ============================================================
# Expansões binárias
# ============================================================
def binary_expansion(j, n):
    return [int(bit) for bit in bin(j)[2:].zfill(n)]

def dyadic_expansion(x, n):
    if 0 <= x < 1:
        expansion = []
        for _ in range(n):
            bit = int(x * 2)
            expansion.append(bit)
            x = x * 2 - bit
        return expansion
    raise ValueError("x deve estar em [0,1).")


# ============================================================
# Utilitários
# ============================================================
def walsh_index_info(j, n):
    """
    Retorna a estrutura do índice j no formato usado no seu circuito.

    Observação:
    - wires do registrador vão de 1 até n
    - wire 0 é a ancilla
    - usamos bitstring invertida para casar com a convenção do seu código
    """
    if not (0 <= j < 2**n):
        raise ValueError(f"j deve estar entre 0 e {2**n - 1}.")

    jbin = bin(j)[2:].zfill(n)[::-1]  # mesma convenção do seu código
    posicoes_de_1 = [i + 1 for i, bit in enumerate(jbin) if bit == "1"]

    if len(posicoes_de_1) == 0:
        return {
            "jbin": jbin,
            "ones": [],
            "controls": [],
            "target": None,
        }

    target = posicoes_de_1[-1]
    controls = posicoes_de_1[:-1]

    return {
        "jbin": jbin,
        "ones": posicoes_de_1,
        "controls": controls,
        "target": target,
    }


def postselect(state, n, tol=1e-14):
    state = qml.math.reshape(state, (2, 2**n))
    reg_state = state[1]

    norm2 = qml.math.sum(qml.math.conj(reg_state) * reg_state)
    norm2_real = qml.math.real(norm2)

    if float(norm2_real) < tol:
        return reg_state

    norm = qml.math.sqrt(norm2_real)
    return reg_state / norm


def prob(state, tol=1e-14):
    norm2 = qml.math.sum(qml.math.conj(state) * state)
    norm2_real = qml.math.real(norm2)

    if float(norm2_real) < tol:
        return qml.math.zeros_like(qml.math.abs(state), dtype=float)

    norm = qml.math.sqrt(norm2_real)
    return qml.math.abs(state / norm) ** 2


def ancilla_success_probability(full_state, n):
    full_state = qml.math.reshape(full_state, (2, 2**n))
    reg_state = full_state[1]
    p = qml.math.sum(qml.math.conj(reg_state) * reg_state)
    return float(qml.math.real(p))


def _postselected_outputs_from_full_state(full_state, n, eps=1e-14):
    full_state = qml.math.reshape(full_state, (2, 2**n))
    reg_state = full_state[1]

    success_p = qml.math.real(qml.math.sum(qml.math.conj(reg_state) * reg_state))
    norm = qml.math.sqrt(success_p + eps)
    postselected_state = reg_state / norm
    postselected_probabilities = qml.math.abs(postselected_state) ** 2

    return success_p, postselected_state, postselected_probabilities


# ============================================================
# Bloco Walsh individual U_j
# ============================================================
def Rj_single_circuit(avec, n, j):
    """
    Implementa apenas o bloco correspondente ao índice j > 0.

    Se j = 0, este termo não entra aqui porque no seu código original
    ele aparece como uma fase na ancilla.
    """
    if j == 0:
        return

    info = walsh_index_info(j, n)
    controls = info["controls"]
    target = info["target"]

    theta = 2 * avec[j]

    if len(info["ones"]) == 1:
        qml.RZ(theta, wires=target)
    else:
        for c in controls:
            qml.CNOT(wires=[c, target])

        qml.RZ(theta, wires=target)

        for c in controls[::-1]:
            qml.CNOT(wires=[c, target])


def selected_walsh_blocks(avec, n, list_j):
    """
    Aplica vários blocos escolhidos.
    """
    for j in list_j:
        if j == 0:
            continue
        Rj_single_circuit(avec, n, j)


# ============================================================
# Execução do subcircuito escolhido
# ============================================================
def run_subcircuit(avec, n, list_j, er=1e-6, return_full_state=False):
    """
    Executa apenas os índices de Walsh em list_j.

    Importante:
    isso NÃO gera o |f> completo, a menos que você inclua todos os índices
    relevantes. Ele gera o estado do loader truncado ao subconjunto list_j.
    """
    list_j = list(list_j)

    for j in list_j:
        if not (0 <= j < len(avec)):
            raise ValueError(f"Índice j={j} fora do tamanho de avec.")
        if j >= 2**n:
            raise ValueError(f"Para n={n}, o maior índice permitido é {2**n - 1}.")

    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev)
    def circuit(avec, er=1e-6):
        avec = er * np.array(avec)

        # ancilla + registrador em superposição
        for w in range(n + 1):
            qml.Hadamard(wires=w)

        # termo j = 0 -> fase relativa na ancilla
        if 0 in list_j:
            qml.PhaseShift(-avec[0], wires=0)

        # termos j > 0 -> blocos controlados
        list_j_nonzero = [j for j in list_j if j != 0]
        if len(list_j_nonzero) > 0:
            qml.ctrl(selected_walsh_blocks, control=0)(avec, n, list_j_nonzero)

        # bloco final da ancilla
        qml.Hadamard(wires=0)
        qml.PhaseShift(-np.pi / 2, wires=0)

        return qml.state()

    full_state = circuit(avec, er=er)
    reg_state = postselect(full_state, n)

    out = {
        "indices": list_j,
        "success_probability_ancilla_1": ancilla_success_probability(full_state, n),
        "postselected_state": reg_state,
        "postselected_probabilities": prob(reg_state),
    }

    if return_full_state:
        out["full_state"] = full_state

    return out


def run_subcircuit_differentiable(avec, n, list_j, er=1e-6, return_full_state=False):
    """
    Versão diferenciável do subcircuito para uso com backprop.
    Mantém a mesma semântica de saída, mas evita conversões para float
    no caminho da loss.
    """
    list_j = list(list_j)

    for j in list_j:
        if not (0 <= j < len(avec)):
            raise ValueError(f"Índice j={j} fora do tamanho de avec.")
        if j >= 2**n:
            raise ValueError(f"Para n={n}, o maior índice permitido é {2**n - 1}.")

    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(avec, er=1e-6):
        avec_eff = er * avec

        for w in range(n + 1):
            qml.Hadamard(wires=w)

        if 0 in list_j:
            qml.PhaseShift(-avec_eff[0], wires=0)

        list_j_nonzero = [j for j in list_j if j != 0]
        if len(list_j_nonzero) > 0:
            qml.ctrl(selected_walsh_blocks, control=0)(avec_eff, n, list_j_nonzero)

        qml.Hadamard(wires=0)
        qml.PhaseShift(-np.pi / 2, wires=0)

        return qml.state()

    full_state = circuit(avec, er=er)
    success_p, postselected_state, postselected_probabilities = _postselected_outputs_from_full_state(
        full_state, n
    )

    out = {
        "indices": list_j,
        "success_probability_ancilla_1": success_p,
        "postselected_state": postselected_state,
        "postselected_probabilities": postselected_probabilities,
    }

    if return_full_state:
        out["full_state"] = full_state

    return out


def run_single_unitary(avec, n, j, er=1e-6, return_full_state=False):
    """
    Executa apenas uma unitária/bloco Walsh de índice j.
    """
    return run_subcircuit(
        avec=avec,
        n=n,
        list_j=[j],
        er=er,
        return_full_state=return_full_state,
    )


# ============================================================
# Desenho do circuito
# ============================================================
def draw_subcircuit(avec, n, list_j, er=1e-6):
    list_j = list(list_j)

    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev)
    def circuit(avec, er=1e-6):
        avec = er * np.array(avec)

        for w in range(n + 1):
            qml.Hadamard(wires=w)

        if 0 in list_j:
            qml.PhaseShift(-avec[0], wires=0)

        list_j_nonzero = [j for j in list_j if j != 0]
        if len(list_j_nonzero) > 0:
            qml.ctrl(selected_walsh_blocks, control=0)(avec, n, list_j_nonzero)

        qml.Hadamard(wires=0)
        qml.PhaseShift(-np.pi / 2, wires=0)

        return qml.state()

    drawer = qml.draw_mpl(circuit, show_all_wires=True)
    return drawer(avec, er=er)


def draw_single_unitary(avec, n, j, er=1e-6):
    return draw_subcircuit(avec, n, [j], er=er)


# ============================================================
# Seu helper original
# ============================================================
def bits_to_state(n, bits):
    if len(bits) != n:
        raise ValueError("O tamanho de bits deve ser igual a n.")

    if any(b not in [0, 1] for b in bits):
        raise ValueError("bits deve conter apenas 0 e 1.")

    bitstring = "".join(map(str, bits))
    labels = [f"{i:0{n}b}" for i in range(2**n)]

    vec = np.zeros(2**n, dtype=int)
    vec[int(bitstring, 2)] = 1

    return bitstring, labels, vec
