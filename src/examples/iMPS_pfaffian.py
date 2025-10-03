import numpy as np
from tenpy.networks import MPS
from temfpy import pfaffian


# Majorana Hamiltonian of a gapped Kitaev chain
def H(L, t1=1.5j, t2=1j):
    M = t1 * np.ones(2 * L - 1)
    M[1::2] = t2
    M = np.diag(M, 1)
    return M + M.T.conj()


# We use default cutoff and degeneracy_tol = 1e-12
trunc_par = dict(chi_max=100)
diag_tol = 1e-6

L_short = 64
cell = 1
cut = L_short // 2

C_short = pfaffian.correlation_matrix(H(L_short), "M->M")
C_long = pfaffian.correlation_matrix(H(L_short + cell), "M->M")

iMPS, val_metric = pfaffian.C_to_iMPS(C_short, C_long, trunc_par, cell, cut, basis="M")
print("Error metric:", val_metric)

# check overlap after inserting more unit cells
n_cell = 8

mps_short = pfaffian.C_to_MPS(C_short, trunc_par, basis="M")

C_vlong = pfaffian.correlation_matrix(H(L_short + n_cell * cell), "M->M")
mps_vlong = pfaffian.C_to_MPS(C_vlong, trunc_par, basis="M")
# reconstruction from mps_short and iMPS
s_vlong = mps_short.sites[:cut] + iMPS.sites * n_cell + mps_short.sites[cut:]
B_vlong = mps_short._B[:cut] + iMPS._B * n_cell + mps_short._B[cut:]
S_vlong = mps_short._S[:cut] + iMPS._S[:-1] * n_cell + mps_short._S[cut:]
f_vlong = mps_short.form[:cut] + iMPS.form * n_cell + mps_short.form[cut:]
mps_rec = MPS(sites=s_vlong, Bs=B_vlong, SVs=S_vlong, form=f_vlong)
print("Reconstruction overlap:", mps_vlong.overlap(mps_rec))
