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

chi = 200

mps_ferm = slater.H_to_MPS(H, {"chi_max": chi}, spinful="PH")

mps_spin = gutzwiller.abrikosov_ph(mps_ferm, inplace=False, return_canonical=True)

print(mps_spin.sites[0])

spectrum = mps_spin.entanglement_spectrum(by_charge=True)
bond = L // 2
logging.disable()  # suppress matplotlib debug logs
for (q,), s in spectrum[bond]:
    plt.plot(q * np.ones(len(s)), s, "kx")
plt.show()
