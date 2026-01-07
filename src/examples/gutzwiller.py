import logging
import numpy as np
import matplotlib.pyplot as plt

from temfpy import setup_logging, slater, gutzwiller

setup_logging(logging.DEBUG)


def hoppingH(L, t=-1):
    M = np.diag(t * np.ones(L - 1), 1)
    return M + M.T


L = 32

H = hoppingH(L)
C, _ = slater.correlation_matrix(H)

chi = 200

mps_ferm = slater.C_to_MPS(C, {"chi_max": chi}, spinful="PH")

mps_spin = gutzwiller.abrikosov_ph(mps_ferm, inplace=False, return_canonical=True)

spectrum = mps_spin.entanglement_spectrum(by_charge=True)
bond = L // 2
for (q,), s in spectrum[bond]:
    plt.plot(q * np.ones(len(s)), s, "kx")
plt.show()
