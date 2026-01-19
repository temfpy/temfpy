import numpy as np
from tenpy.networks import MPS
from temfpy import slater


def H(L, t1=-1, t2=-1.5):
    M = t1 * np.ones(L - 1)
    M[1::2] = t2
    M = np.diag(M, 1)
    return M + M.T


# We use default cutoff and degeneracy_tol = 1e-12
trunc_par = dict(chi_max=100)
diag_tol = 1e-6

L_short = 128
cell = 2
cut = L_short // 2

H_short = H(L_short)
H_long = H(L_short + cell)
iMPS, val_metric = slater.H_to_iMPS(H_short, H_long, trunc_par, 2, cut)
print("Error metric:", val_metric)

# check overlap after inserting more unit cells
n_cell = 8

mps_short = slater.H_to_MPS(H(L_short), trunc_par)

H_vlong = H(L_short + n_cell * cell)
mps_vlong = slater.H_to_MPS(H_vlong, trunc_par)
# reconstruction from mps_short and iMPS
s_vlong = mps_short.sites[:cut] + iMPS.sites * n_cell + mps_short.sites[cut:]
B_vlong = mps_short._B[:cut] + iMPS._B * n_cell + mps_short._B[cut:]
S_vlong = mps_short._S[:cut] + iMPS._S * n_cell + mps_short._S[cut:]
f_vlong = mps_short.form[:cut] + iMPS.form * n_cell + mps_short.form[cut:]
mps_rec = MPS(sites=s_vlong, Bs=B_vlong, SVs=S_vlong, form=f_vlong)
print("Reconstruction overlap:", mps_vlong.overlap(mps_rec))
