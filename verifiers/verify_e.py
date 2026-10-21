"""Phase E: verify_us.py + --preset <npy of full-grid hf5 columns>: the brain matrix holds exactly the preset voxels (after the
noise is applied and the condition's set taken, as in tang_fit_e.py). Without --preset: identical to Phase C.
Original docstring: Phase C contrastive verifier C (DESIGN_C.md 5.2): Phase A/B verify.py unchanged (PCA q x lambda grid, InfoNCE,
early stopping and calibration on the calibration story, LOSO over the rung's stories, full-rung model for every
other block), run on a cell's voxel set and noise. Changes: (1) cells are <cond>_<rung> with ultrasound conditions
(noise_us.USNoise: per-voxel noise on the full grid, re-z-score, restrict to the set) and story rungs n1/n2/n3;
(2) a 1-story rung has no LOSO model (its own story is not scored; all other stories are scored by the full model).
Output: verify.py's npz format (results_C_us/<S>_<cell>.npz) + one json per cell (<S>_<cell>_cell.json = one entry of verify.py's cells.json).
Usage: python verify_us.py --subject S1 --cells usL50_r15,usL90_r15,native_n1"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify as Vf
from verify import Data, Text, pca_fit, project, build_xy, train, Scorer, fit_T_theta, summarise, flatten, seed_of, ranges_for, sync, QS, LAMS, OFFS, DEV
from noise_us import USNoise, parse_cell, rung_stories, is_us, pseudo_db
CALIB = "fromboyhoodtofatherhood"
PRESET = None                                                              # set in __main__ (module global read by DataUS.load)


class DataUS(Data):
    """verify.Data whose response matrix holds one condition's (noisy, voxel-restricted) blocks"""
    def __init__(self, subj):
        super().__init__(subj); self.V_full = self.V; del self.R; torch.cuda.empty_cache()
        self.R = None; self.noise = USNoise(subj)
    def load(self, cond):
        t = time.time(); cols = self.noise.cols(cond); base = np.arange(self.V_full) if cols is None else cols
        pos = np.sort(np.where(np.isin(base, PRESET))[0]) if PRESET is not None else np.arange(len(base)); pos_t = torch.from_numpy(pos).to(DEV)
        V = len(pos); self.preset_cols = base[pos]
        self.R = None; torch.cuda.empty_cache(); self.R = torch.empty((self.N, V), dtype=torch.float16, device=DEV); self.V = V
        for key, (s0, n) in self.blocks.items():
            x = torch.from_numpy(np.ascontiguousarray(self.mm[s0:s0 + n])).to(DEV)
            y = self.noise.apply(x, cond, key, restrict=True) if cond != "native" else x
            self.R[s0:s0 + n] = y[:, pos_t].half(); del y
        sync(); return time.time() - t


def run_subject(subj, cells, out_dir, s2s):
    D = DataUS(subj); T = Text(); t_all = time.time()
    cjf = lambda c: f"{out_dir}/{subj}_{c}_cell.json"                      # one json per cell (parallel jobs never share a file)
    by_cond = {}
    for c in cells: by_cond.setdefault(parse_cell(c)[0], []).append(c)
    for cond, cl in by_cond.items():
        todo = [c for c in cl if not os.path.exists(cjf(c))]
        if not todo: continue
        tl = D.load(cond)
        for cell in todo:
            _, rung = parse_cell(cell); t_cell = time.time(); gpu = {"pca": 0.0, "train": 0.0, "eval": 0.0, "tune": 0.0}
            rs = rung_stories(rung, s2s); rung_st = [s for s in D.train_stories if s in rs]; assert len(rung_st) == len(rs), (rung, rs)
            hours = sum(D.blocks[s][1] for s in rung_st) * 2 / 3600
            trials = []
            for q in QS:
                qq = min(q, D.V - 1)
                t = time.time(); mu, comps, _ = pca_fit(D, ranges_for(D, rung_st), qq, seed=seed_of(subj, cond, rung, "pca")); F = project(D, mu, comps); sync(); gpu["tune"] += time.time() - t
                X, E, S, A = build_xy(F, D, T, rung_st); Xv, Ev, Sv, Av = build_xy(F, D, T, [CALIB])
                for lam in LAMS:
                    t = time.time(); m, norm, vl, bep, nep = train(X, E, S, A, Xv, Ev, Sv, Av, lam, seed=seed_of(subj, cond, rung, q, lam)); sync(); gpu["tune"] += time.time() - t
                    trials.append({"q": qq, "lam": lam, "val_loss": vl, "best_step": bep, "steps": nep})
                    print(f"[{subj} {cell}] tune q={qq} lam={lam:g} val={vl:.4f} step={bep}/{nep}", flush=True)
                del F, X
            bt = min(trials, key=lambda x: x["val_loss"]); q, lam = bt["q"], bt["lam"]
            models = [("FULL", None)] + ([(s, s) for s in rung_st] if len(rung_st) > 1 else [])
            cell_out = {"6": {}, "20": {}}; calib = {}
            for name, held in models:
                tr = [s for s in rung_st if s != held]
                t = time.time(); mu, comps, _ = pca_fit(D, ranges_for(D, tr), q, seed=seed_of(subj, cond, rung, "pca", name)); F = project(D, mu, comps); sync(); gpu["pca"] += time.time() - t
                X, E, S, A = build_xy(F, D, T, tr); Xv, Ev, Sv, Av = build_xy(F, D, T, [CALIB])
                t = time.time(); m, norm, vl, bep, nep = train(X, E, S, A, Xv, Ev, Sv, Av, lam, seed=seed_of(subj, cond, rung, name)); sync(); gpu["train"] += time.time() - t
                del X
                t = time.time(); sc = Scorer(D, T, F, m, norm)
                o6c = sc.score6(CALIB, CALIB, seed_of(subj, cond, rung, name, "c6")); o20c = sc.score20(CALIB, CALIB, seed_of(subj, cond, rung, name, "c20"))
                T6, th6, ba6, tht6, bat6 = fit_T_theta(o6c, T.w6["foil_valid"][o6c["sel"]]); T20, th20, ba20, tht20, bat20 = fit_T_theta(o20c, T.s20["foil_valid"][o20c["sel"]])
                calib[name] = {"T6": T6, "th6": th6, "calib_ba6": ba6, "thtn6": tht6, "calib_ba_tn6": bat6, "T20": T20, "th20": th20, "calib_ba20": ba20,
                               "thtn20": tht20, "calib_ba_tn20": bat20, "val_loss": vl, "best_step": bep, "steps": nep}
                evals = [held] if held is not None else [s for s in D.train_stories if s not in rung_st] + [f"test_rep{r}" for r in range(10)]
                for ev in evals:
                    story = "wheretheressmoke" if ev.startswith("test_rep") else ev
                    o6 = sc.score6(ev, story, seed_of(subj, cond, rung, ev, "6")); o20 = sc.score20(ev, story, seed_of(subj, cond, rung, ev, "20"))
                    cell_out["6"][ev] = {**summarise(o6, T.w6["foil_valid"][o6["sel"]], None, T6, th6), "th": th6, "thtn": tht6, "T": T6, "model": name}
                    cell_out["20"][ev] = {**summarise(o20, T.s20["foil_valid"][o20["sel"]], None, T20, th20), "th": th20, "thtn": tht20, "T": T20, "model": name}
                sync(); gpu["eval"] += time.time() - t
                del F, sc, m
                print(f"[{subj} {cell}] model {name} val={vl:.3f} T20={T20:.3f} ba20c={ba20:.3f}/{bat20:.3f} t={time.time()-t_cell:.0f}s", flush=True)
            res = {"cond": cond, "rung": rung, "dB": pseudo_db(cond), "ultrasound": is_us(cond), "V_set": int(D.V), "rung_stories": rung_st, "hours": hours,
                         "n_train_stories": len(rung_st), "q": q, "lam": lam, "trials": trials, "calib": calib, "n_params_trainable": len(OFFS) * q * 768 + 768 + 1,
                         "n_params_pca": q * D.V, "preset": (None if PRESET is None else "anat"), "n_preset_in_set": int(D.V), "n_models": len(models), "loso": len(rung_st) > 1, "gpu_s": gpu, "load_s": tl, "wall_s": time.time() - t_cell}
            np.savez_compressed(f"{out_dir}/{subj}_{cell}.npz.tmp.npz", **flatten(cell_out)); os.replace(f"{out_dir}/{subj}_{cell}.npz.tmp.npz", f"{out_dir}/{subj}_{cell}.npz")
            json.dump({cell: res}, open(cjf(cell) + ".tmp", "w"), indent=1); os.replace(cjf(cell) + ".tmp", cjf(cell))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--cells", required=True)
    ap.add_argument("--out", default=f"{VP.WORK}/results_C_us"); ap.add_argument("--preset", default=None); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    PRESET = np.load(a.preset).astype(np.int64) if a.preset else None
    s2s = json.load(open(f"{VP.TANG_BOX}/data_train/sess_to_story.json"))
    run_subject(a.subject, a.cells.split(","), a.out, s2s)
