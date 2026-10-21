"""Link-budget calculator (per-voxel functional SNR of transcranial power Doppler at the temporal window).
d' = delta / sqrt(1/(K*SNR0^2) + lam^2);  SNR0 = sqrt(N_ind)*s/(1+s);  N_ind = M*T/tau_c;
tau_c = (c/f)/(2*v);  1/s = 1/s_n + 1/s_tc + 1/s_rev.  All dB are 10log10 for echo SNR, 20log10 for d'.
Constants: linkbudget/params.yaml (via params.py). The functions are the ones that produced the paper's numbers;
depth_curves.py extends model() to a continuous depth.
Usage: python snr_model.py   (self-check: reproduces the worked-example d' at 2 MHz, 30-mm aperture, delta 10 %)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as PR

db = lambda x: 10 ** (x / 10)
C = PR.C
# worked-example rows: (f MHz, z cm) -> (s_n dB, cells per 27 mm^3, d' with clutter+reverb at delta=0.10, 30 mm aperture)
TABLE = PR.TABLE
S_REV_2MHZ = PR.S_REV_2MHZ               # assumed reverberation leakage, temporal bone, 2 MHz
S_TC_2MHZ = PR.S_TC_2MHZ                 # assumed tissue-clutter leakage, 2 MHz
AGENT = PR.AGENT                         # worked-example inputs

def tau_c(f, v): return (C / (f * 1e6)) * 1e3 / (2 * v)        # s, with lambda in mm and v in mm/s
def dprime(s_lin, M, f, delta, lam, K, T, v):
    n_ind = M * T / tau_c(f, v); snr0 = np.sqrt(n_ind) * s_lin / (1 + s_lin)
    return delta / np.sqrt(1 / (K * snr0 ** 2) + lam ** 2)

def leak_from_table(f, z):
    """Back out the worked example's leakage term L = 1/s_tc + 1/s_rev from its own d'_with (used for 1 and 3 MHz)."""
    sn, M, dw = TABLE[(f, z)]; a = AGENT
    snr0 = 1 / np.sqrt(a["K"] * ((a["delta"] / dw) ** 2 - a["lam"] ** 2))
    q = snr0 / np.sqrt(M * a["T"] / tau_c(f, a["v"])); s = q / (1 - q)
    return max(1 / s - 1 / db(sn), 0.0)

def model(f=PR.F_MHZ, z=4, delta=PR.DEF["delta"], lam=PR.DEF["lam"], K=PR.DEF["K"], T=PR.T_IMAGE, v=PR.DEF["v_mm_per_s"],
          aperture_mm=PR.APERTURE_MM, cal_db=0.0, bone_mm=PR.BONE_REF_MM, leak_db=0.0, voxel_mm3=PR.V_LB):
    sn, M, _ = TABLE[(f, z)]
    sn = sn + cal_db
    if aperture_mm == 20: sn -= PR.AP20_SN_LOSS_DB; M = M / PR.AP20_CELLS_DIV   # 20 mm costs ~2 dB in s_n and 1.5x in sqrt(M)
    M = M * voxel_mm3 / PR.V_LB
    if f == 2:
        dbone = 2 * PR.BONE_COEF * 2 ** PR.BONE_EXP * (bone_mm - PR.BONE_REF_MM)   # two-way compact-bone loss change at 2 MHz
        sn -= dbone
        L = 1 / db(S_TC_2MHZ + leak_db) + 1 / db(S_REV_2MHZ[z] - dbone + leak_db)
    else:
        L = leak_from_table(f, z) / db(leak_db)
    s = 1 / (1 / db(sn) + L)
    return dprime(s, M, f, delta, lam, K, T, v)

d2dB = lambda d: 20 * np.log10(d)
if __name__ == "__main__":
    chk = {z: round(float(model(2, z, delta=0.10, aperture_mm=30)), 3) for z in (2, 3, 4, 5)}
    ref = {z: TABLE[(2, z)][2] for z in (2, 3, 4, 5)}
    print("worked-example check at 2 MHz (model vs table d'):", chk, ref)
