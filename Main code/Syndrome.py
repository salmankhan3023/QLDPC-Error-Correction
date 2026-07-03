"""
Build a Stim syndrome-extraction circuit for a CSS code given its
H_X and H_Z check matrices.

Approach: treat this as scheduling a bipartite interaction graph.
Each check (ancilla) needs a CNOT with every data qubit in its support.
We greedily schedule these CNOTs into time-slots such that no qubit
(ancilla or data) is used twice in the same slot -- this guarantees a
valid, collision-free circuit (not necessarily the minimal-depth
schedule from the paper, but structurally correct and simulatable).
"""

import stim
import numpy as np
from bb_code import build_bb_code


def _schedule_single_type(edges):
    """Greedily schedule a list of (ancilla, data, ctype) edges -- all of
    the SAME check type -- into collision-free timesteps."""
    timesteps = []
    for anc, data, ctype in edges:
        placed = False
        for slot in timesteps:
            used_qubits = {q for (a, d, _) in slot for q in (a, d)}
            if anc not in used_qubits and data not in used_qubits:
                slot.append((anc, data, ctype))
                placed = True
                break
        if not placed:
            timesteps.append([(anc, data, ctype)])
    return timesteps


def schedule_edges(HX, HZ):
    """
    Returns a list of timesteps. Each timestep is a list of
    (ancilla_qubit, data_qubit, check_type) tuples that can be
    applied simultaneously.

    IMPORTANT: X-check and Z-check interactions are scheduled in two
    SEPARATE phases (all X-checks first, then all Z-checks), never
    interleaved. A data qubit acts as a CNOT *target* for X-checks and
    as a CNOT *control* for Z-checks -- these two roles don't commute
    in arbitrary order, so freely interleaving them (as a naive combined
    greedy schedule would) breaks the determinism of the syndrome
    measurement, even in the ideal noiseless circuit.
    """
    n_x_checks, n = HX.shape
    n_z_checks, _ = HZ.shape

    x_ancilla = lambda i: n + i
    z_ancilla = lambda i: n + n_x_checks + i

    x_edges = [(x_ancilla(i), int(j), 'X')
               for i in range(n_x_checks) for j in np.nonzero(HX[i])[0]]
    z_edges = [(z_ancilla(i), int(j), 'Z')
               for i in range(n_z_checks) for j in np.nonzero(HZ[i])[0]]

    x_timesteps = _schedule_single_type(x_edges)
    z_timesteps = _schedule_single_type(z_edges)

    timesteps = x_timesteps + z_timesteps
    return timesteps, n, n_x_checks, n_z_checks


def build_memory_circuit(HX, HZ, rounds: int, p: float, basis: str = 'Z'):
    """
    Build a `rounds`-round quantum memory experiment circuit for the given
    CSS code, with uniform circuit-level depolarizing-style noise at rate p.
    `basis` = 'Z' means we're protecting against Z errors detected by X
    checks being flipped (standard Z-memory experiment convention varies;
    we keep it simple and symmetric here since both check types are present).
    """
    timesteps, n, n_x, n_z = schedule_edges(HX, HZ)
    x_ancillas = list(range(n, n + n_x))
    z_ancillas = list(range(n + n_x, n + n_x + n_z))
    all_ancillas = x_ancillas + z_ancillas
    total_qubits = n + n_x + n_z

    circuit = stim.Circuit()

    def syndrome_round(first_round: bool):
        # Reset ancillas
        circuit.append("R", x_ancillas)
        circuit.append("RX", []) if False else None  # placeholder no-op
        circuit.append("H", x_ancillas)               # X-ancillas start in |+>
        circuit.append("R", z_ancillas)                # Z-ancillas start in |0>
        circuit.append("DEPOLARIZE1", x_ancillas + z_ancillas, p)

        for slot in timesteps:
            pairs = []
            for anc, data, ctype in slot:
                if ctype == 'X':
                    pairs += [anc, data]   # control=ancilla, target=data
                else:
                    pairs += [data, anc]   # control=data, target=ancilla
            circuit.append("CX", pairs)
            circuit.append("DEPOLARIZE2", pairs, p)

        circuit.append("H", x_ancillas)
        circuit.append("DEPOLARIZE1", x_ancillas + z_ancillas, p)
        circuit.append("MR", x_ancillas + z_ancillas)

    # First round: data qubits start in |0>, which is a +1 eigenstate of
    # every Z-type check automatically -- so ONLY the Z-check outcomes are
    # deterministic on their own in round 1. X-check outcomes are NOT
    # predictable from nothing (|0> is not an eigenstate of X-type
    # checks); they only become deterministic from round 2 onward, by
    # comparing against round 1. So round 1 gets detectors for Z-ancillas
    # only.
    syndrome_round(first_round=True)
    n_anc = len(all_ancillas)
    n_x = len(x_ancillas)
    # measurement record order was x_ancillas + z_ancillas, so within the
    # last n_anc records: indices [0:n_x] are X-ancillas, [n_x:n_anc] are Z
    for i in range(n_x, n_anc):
        circuit.append("DETECTOR", [stim.target_rec(-n_anc + i)])

    for r in range(rounds - 1):
        syndrome_round(first_round=False)
        for i in range(n_anc):
            circuit.append(
                "DETECTOR",
                [stim.target_rec(-n_anc + i), stim.target_rec(-2 * n_anc + i)],
            )

    circuit.append("M", list(range(n)))  # final data qubit readout

    return circuit


if __name__ == "__main__":
    l, m = 12, 6
    A_terms = [('x', 3), ('y', 1), ('y', 2)]
    B_terms = [('y', 3), ('x', 1), ('x', 2)]
    code = build_bb_code(l, m, A_terms, B_terms)

    timesteps, n, n_x, n_z = schedule_edges(code['HX'], code['HZ'])
    print(f"Data qubits: {n}, X-ancillas: {n_x}, Z-ancillas: {n_z}")
    print(f"Greedy schedule depth: {len(timesteps)} timesteps "
          f"(paper's optimized schedule achieves 6 -- ours is a first-pass, "
          f"unoptimized version)")

    circuit = build_memory_circuit(code['HX'], code['HZ'], rounds=3, p=0.001)
    print(f"\nCircuit built: {circuit.num_qubits} qubits, "
          f"{len(circuit)} instructions")

    # Sanity check: does Stim accept this as a valid detector circuit?
    dem = circuit.detector_error_model()
    print(f"Valid detector error model built successfully: "
          f"{dem.num_detectors} detectors, {dem.num_errors} error mechanisms")

    # Noiseless sanity check: with p=0, all detectors should always fire "0"
    clean_circuit = build_memory_circuit(code['HX'], code['HZ'], rounds=3, p=0.0)
    sampler = clean_circuit.compile_detector_sampler()
    samples = sampler.sample(shots=10)
    print(f"\nNoiseless sanity check -- all detectors should read 0: "
          f"{'PASS' if not samples.any() else 'FAIL'}")
