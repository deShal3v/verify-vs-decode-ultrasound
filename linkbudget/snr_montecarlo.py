"""Monte Carlo over the uncertain inputs of the temple link budget (snr_model.py), all varied jointly, at depths 3-5 cm
(the original form; depth_curves.py evaluates every depth on the same draws). Writes $WORK_DIR/linkbudget/snr_montecarlo.json.
Usage: python snr_montecarlo.py"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as PR
from snr_model import model


def run():
    rng = np.random.default_rng(PR.MC_SEED); N = PR.MC_N
    logu = lambda a, b, n: np.exp(rng.uniform(np.log(a), np.log(b), n))
    R = PR.MC_RANGES
    S = dict(cal_db=rng.uniform(*R["cal_db"][1], N),            # absolute calibration of echo SNR
             delta=logu(*R["delta"][1], N),                      # brain response, blood-volume change
             bone_mm=rng.uniform(*R["bone_mm"][1], N),           # temporal bone, good to marginal window
             z=rng.choice(PR.MC_Z_CHOICE, N),                    # depth of target, cm
             leak_db=rng.uniform(*R["leak_db"][1], N),           # clutter + ringing leakage
             lam=logu(*R["lam"][1], N),                          # physiological noise
             v=logu(*R["v"][1], N))                              # blood speed in the voxel, mm/s
    d = np.array([model(f=2, z=int(S["z"][i]), delta=S["delta"][i], lam=S["lam"][i], v=S["v"][i], cal_db=S["cal_db"][i],
                        bone_mm=S["bone_mm"][i], leak_db=S["leak_db"][i]) for i in range(N)])
    dB = 20 * np.log10(d)
    q = np.percentile(dB, [10, 25, 50, 75, 90])
    out = {"N": N, "pct10_25_50_75_90_dB": q.tolist(), "frac_above_-10dB": float((dB > -10).mean()),
           "frac_above_-20dB": float((dB > -20).mean()), "frac_below_-30dB": float((dB < -30).mean())}
    from scipy.stats import spearmanr
    out["spearman_with_output"] = {k: float(spearmanr(v, dB)[0]) for k, v in S.items()}
    return out


if __name__ == "__main__":
    import vvd_paths as VP
    out = run(); od = os.path.join(VP.WORK, "linkbudget"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "snr_montecarlo.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
