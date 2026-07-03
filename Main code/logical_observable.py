"""
Find a logical Z operator for the code, and wire it into the Stim circuit
as an OBSERVABLE_INCLUDE, so we can actually measure logical error rate.

A Z-type logical operator is a binary vector v (over the n data qubits)
such that:
  1. It commutes with every X-check:  H_X . v = 0  (mod 2)
     -> v lies in the null space (kernel) of H_X
  2. It is NOT itself a product of Z-stabilizers:
     -> v is not in the row space of H_Z

Any vector satisfying both is a valid representative of a nontrivial
logical Z operator. We don't need the full basis of all 12 logical
qubits for a first working pipeline -- just one, to track as our
observable.
"""

import numpy as np
import galois
import stim

from bb_code import build_bb_code
from build_circuit import build_memory_circuit, schedule_edges

GF2 = galois.GF(2)


def find_logical_z(HX: np.ndarray, HZ: np.ndarray) -> np.ndarray:
    HX_gf = GF2(HX.astype(np.uint8))
    HZ_gf = GF2(HZ.astype(np.uint8))

    null_space = HX_gf.null_space()   # rows span ker(H_X)
    rank_HZ = int(np.linalg.matrix_rank(HZ_gf))

    for i in range(null_space.shape[0]):
        v = null_space[i]
        # Check if v is in rowspan(H_Z): does adding v to H_Z's rows
        # increase the rank? If not, v is already in the span.
        stacked = np.vstack([HZ_gf, v.reshape(1, -1)])
        rank_stacked = int(np.linalg.matrix_rank(GF2(stacked)))
        if rank_stacked > rank_HZ:
            return np.array(v, dtype=int)

    raise RuntimeError("No nontrivial logical Z operator found -- "
                        "something is wrong with the code construction.")


def build_memory_circuit_with_observable(HX, HZ, rounds: int, p: float):
    circuit = build_memory_circuit(HX, HZ, rounds=rounds, p=p)
    logical_z = find_logical_z(HX, HZ)
    support = np.nonzero(logical_z)[0]

    n = HX.shape[1]
    # The final "M" instruction measured qubits [0..n-1] in order, so the
    # measurement record for data qubit j is at offset -(n - j) from the end.
    targets = [stim.target_rec(-(n - j)) for j in support]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit, logical_z


if __name__ == "__main__":
    l, m = 12, 6
    A_terms = [('x', 3), ('y', 1), ('y', 2)]
    B_terms = [('y', 3), ('x', 1), ('x', 2)]
    code = build_bb_code(l, m, A_terms, B_terms)

    logical_z = find_logical_z(code['HX'], code['HZ'])
    print(f"Found logical Z operator with weight {logical_z.sum()} "
          f"(support size out of {len(logical_z)} data qubits)")

    circuit, _ = build_memory_circuit_with_observable(
        code['HX'], code['HZ'], rounds=3, p=0.001
    )
    dem = circuit.detector_error_model()
    print(f"Valid detector error model with observable: "
          f"{dem.num_detectors} detectors, {dem.num_observables} observable(s)")

    # Noiseless sanity check: with p=0, the logical observable should
    # never flip (always reads 0)
    clean_circuit, _ = build_memory_circuit_with_observable(
        code['HX'], code['HZ'], rounds=3, p=0.0
    )
    sampler = clean_circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=20, separate_observables=True)
    print(f"Noiseless observable check -- should never flip: "
          f"{'PASS' if not obs.any() else 'FAIL'}")
