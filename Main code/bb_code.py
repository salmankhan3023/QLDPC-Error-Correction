"""
Bivariate bicycle (BB) code construction, following Bravyi et al. 2024
("High-threshold and low-overhead fault-tolerant quantum memory", Nature).

We work in the group algebra of Z_l x Z_m over GF(2). Elements x, y are the
generators of two cyclic groups of order l and m respectively, represented
as l x l and m x m cyclic shift matrices, lifted to (l*m) x (l*m) matrices:

    x = S_l  (x)  I_m
    y = I_l  (x)  S_m

A "polynomial" like x^3 + y + y^2 is then just the sum (mod 2) of the
corresponding matrix powers/products.

Given two polynomials A(x,y) and B(x,y), the BB code's parity-check
matrices are:

    H_X = [ A | B ]
    H_Z = [ B^T | A^T ]

each of shape (l*m) x (2*l*m). This gives n = 2*l*m physical qubits.
"""

import numpy as np
import galois

GF2 = galois.GF(2)


def cyclic_shift(n: int) -> np.ndarray:
    """n x n cyclic shift matrix S where S[i, (i+1) mod n] = 1."""
    S = np.zeros((n, n), dtype=int)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def make_generators(l: int, m: int):
    """Return the lifted x, y generator matrices, each (l*m) x (l*m)."""
    Sl = cyclic_shift(l)
    Sm = cyclic_shift(m)
    Il = np.eye(l, dtype=int)
    Im = np.eye(m, dtype=int)
    x = np.kron(Sl, Im)
    y = np.kron(Il, Sm)
    return x, y


def matrix_power(mat: np.ndarray, power: int) -> np.ndarray:
    result = np.eye(mat.shape[0], dtype=int)
    for _ in range(power):
        result = (result @ mat) % 2
    return result


def poly_xy(x: np.ndarray, y: np.ndarray, terms):
    """
    Evaluate a polynomial given as a list of (gen, power) monomials, e.g.
    terms = [('x', 3), ('y', 1), ('y', 2)]  ->  x^3 + y + y^2  (mod 2)
    """
    dim = x.shape[0]
    total = np.zeros((dim, dim), dtype=int)
    for gen, power in terms:
        base = x if gen == 'x' else y
        total = (total + matrix_power(base, power)) % 2
    return total


def gf2_rank(mat: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(GF2(mat.astype(np.uint8))))


def build_bb_code(l: int, m: int, A_terms, B_terms):
    x, y = make_generators(l, m)
    A = poly_xy(x, y, A_terms)
    B = poly_xy(x, y, B_terms)

    HX = np.concatenate([A, B], axis=1)          # (lm) x (2lm)
    HZ = np.concatenate([B.T, A.T], axis=1)       # (lm) x (2lm)

    n = 2 * l * m

    # CSS commutation check: H_X H_Z^T = 0 (mod 2)
    commute = (HX @ HZ.T) % 2
    assert not commute.any(), "H_X and H_Z do not commute -- invalid CSS code"

    rank_X = gf2_rank(HX)
    rank_Z = gf2_rank(HZ)
    k = n - rank_X - rank_Z

    return {
        "HX": HX,
        "HZ": HZ,
        "n": n,
        "k": k,
        "rank_X": rank_X,
        "rank_Z": rank_Z,
    }


if __name__ == "__main__":
    # The "Gross code": l=12, m=6, A = x^3 + y + y^2, B = y^3 + x + x^2
    # Published parameters: [[144, 12, 12]]
    l, m = 12, 6
    A_terms = [('x', 3), ('y', 1), ('y', 2)]
    B_terms = [('y', 3), ('x', 1), ('x', 2)]

    code = build_bb_code(l, m, A_terms, B_terms)

    print(f"n (physical qubits) = {code['n']}")
    print(f"rank(H_X) = {code['rank_X']}, rank(H_Z) = {code['rank_Z']}")
    print(f"k (logical qubits) = {code['k']}")
    print(f"Expected: [[144, 12, 12]]  ->  n=144, k=12 match: {code['n'] == 144 and code['k'] == 12}")
