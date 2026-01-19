import logging
import numpy as np

from temfpy import slater, setup_logging
from temfpy.utils import HT

setup_logging(logging.DEBUG)


def hoppingH(L, t=-1):
    M = np.diag(t * np.ones(L - 1), 1)
    return M + M.T


def randomH(L, range=3):
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    scale = np.exp(-abs(x - y) / range)
    M = np.random.normal(size=(2, L, L), scale=scale)
    M = M[0] + 1j * M[1]
    return M + HT(M)


chi = 200

L = 32
H = randomH(L)

mps = slater.H_to_MPS(H, {"chi_max": chi})

# Verify with correlation matrix
C, _ = slater.correlation_matrix(H)

CdC = mps.correlation_function("Cd", "C").T
dev = CdC - C
print(np.max(np.abs(dev)), np.linalg.norm(dev))
print(np.linalg.norm(CdC.imag))
