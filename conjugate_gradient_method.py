import numpy as np
from scipy.sparse import csr_matrix


def solve(A, b, x=None, eps=1e-3):
    n = len(b)
    if not x:
        x = np.ones(n)
    rows = [row for row, col in A.A]
    cols = [col for row, col in A.A]
    A = csr_matrix((list(A.A.values()), (rows, cols)))
    r = A.dot(x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = A.dot(p)
        alpha = r_k_norm / np.dot(p, Ap)

        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)

        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < eps:
            break
        p = beta * p - r
    print('iter:', i)
    return x