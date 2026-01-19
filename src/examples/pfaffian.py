import logging
import numpy as np
from temfpy import pfaffian, setup_logging

setup_logging(logging.DEBUG)


def majorana_nn_Hamiltonian(L, t=1j):
    M = np.diag(t * np.ones(2 * L - 1), 1)
    return M + M.T.conj()


def majorana_random_Hamiltonian(L, range=3):
    x, y = np.meshgrid(np.arange(2 * L), np.arange(2 * L), indexing="ij")
    scale = np.exp(-abs(x - y) / range)
    M = np.random.normal(scale=scale)
    return 1j * (M - M.T)


L = 20

H = majorana_random_Hamiltonian(L)

chi = 200

psi = pfaffian.H_to_MPS(H, {"chi_max": chi}, basis="M")

# Verify with correlation matrix
C = pfaffian.correlation_matrix(H, basis="M->C")

CdC = psi.correlation_function("Cd", "C").T
dev = CdC - C[::2, ::2]
print(np.max(np.abs(dev)), np.linalg.norm(dev))
print(np.linalg.norm(CdC.imag))

CC = psi.correlation_function("C", "C").T
dev = CC - C[::2, 1::2]
print(np.max(np.abs(dev)), np.linalg.norm(dev))
print(np.linalg.norm(CC.imag))
