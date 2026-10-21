"""Phase E4 within-story control (2026-10-21, Omer): the contrastive verifier exactly as verify_e.py (same voxel preset, noise, PCA q x lambda
tuning on the calibration story, seeds, InfoNCE training, one leave-one-story-out model per training story), scored on the 20-s segments of
each held-out story twice:
  (1) the paper's pool, every evaluable 20-s segment of the OTHER stories (reproduction check against the paper's loso_20 rank bits);
  (2) a same-story pool: every 20-s window of the SAME story at 2-s steps (ws_text.py, >= 3 words) whose start is at least 20 s from the true
      window's start (|anchor difference| >= 10).
Both use the same time-shifted control (same shift, same pool) and the same rank bits per window, log2((P+1)/rank) - log2((P+1)/rank_shifted),
P = that window's pool size. Every model is reseeded per call (pca_fit, train, noise), so skipping the full-rung model, the calibration T/theta fit
and the 6-s readout (none of which enters the LOSO rank bits) leaves the LOSO models unchanged.
Output: $WORK_DIR/results_C_us/<S>_ws_<cell>.npz (per held-out story: sel, anchor, rank2, rank2_sh, npool, rank2_ws, rank2_ws_sh, npool_ws)
+ <S>_ws_<cell>_cell.json. Usage: python verify_ws.py --subject S1 --cells native_n82,m12c_n82 --preset $WORK_DIR/inputs/S1/anat_cols.npy"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e as VE
from verify import Text, pca_fit, project, build_xy, train, Scorer, seed_of, ranges_for, sync, QS, LAMS, DEV
from noise_us import parse_cell, rung_stories
TXT = f"{VP.WORK}/text"


class WS:
    """same-story candidate windows (2-s steps) of one story, on the GPU"""
    def __init__(self):
        z = np.load(f"{TXT}/ws20.npz"); self.story = z["story"]; self.anchor = z["anchor"]; self.ok = z["nw"] >= 3; self.mask = z["sub_mask"]
        self.sub = np.load(f"{TXT}/ws20_sub.npy", mmap_mode="r")
    def get(self, sid):
        ix = np.where((self.story == sid) & self.ok)[0]
        return (torch.tensor(self.anchor[ix], device=DEV), torch.tensor(np.ascontiguousarray(self.sub[ix])).to(DEV).float(),
                torch.tensor(self.mask[ix], device=DEV).float())


class ScorerWS(Scorer):
    @torch.no_grad()
    def score20ws(self, key, story, shift_seed, ws):
        """verify.Scorer.score20 (same windows, same shift, same pool, same rank) plus the same-story pool"""
        T = self.T; sid = T.sid[story]; Z = self.z_block(key)
        sel = np.where((T.s20["story"] == sid) & T.ev20)[0]
        a = torch.tensor(T.s20["anchor"][sel], device=DEV)
        nA = Z.shape[0]; rng = np.random.default_rng(shift_seed + 1); sh = int(rng.integers(15, max(16, nA - 15)))
        sub = torch.tensor(np.ascontiguousarray(T.sub[sel])).to(DEV).float()
        mask = T.s20_mask[sel].float()
        pool_m = torch.tensor(T.ev20 & (T.s20["story"] != sid), device=DEV)
        Pe = T.s20_true_sub[pool_m].float(); Pm = T.s20_mask[pool_m].float()
        wa, We, Wm = ws.get(sid); keep = (wa[None, :] - a[:, None]).abs() >= 10                  # (n, Pw): start >= 20 s from the true start
        out = {}
        for tag, s in (("", 0), ("_sh", sh)):
            bidx = (a[:, None] + torch.arange(-9, 1, device=DEV)[None] + s) % nA
            Zs = Z[bidx]
            sc = (torch.einsum("nbd,nkbd->nkb", Zs, sub) * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
            st = sc[:, 0]
            ps = torch.einsum("nbd,pbd->npb", Zs, Pe); ps = (ps * Pm[None]).sum(-1) / Pm.sum(-1).clamp_min(1)[None]
            out["rank2" + tag] = (1 + (ps > st[:, None]).sum(1)).cpu().numpy().astype(np.int32)
            pw = torch.einsum("nbd,pbd->npb", Zs, We); pw = (pw * Wm[None]).sum(-1) / Wm.sum(-1).clamp_min(1)[None]
            out["rank2_ws" + tag] = (1 + ((pw > st[:, None]) & keep).sum(1)).cpu().numpy().astype(np.int32)
        out["npool"] = np.int32(pool_m.sum().item()); out["npool_ws"] = keep.sum(1).cpu().numpy().astype(np.int32)
        out["sel"] = sel.astype(np.int32); out["anchor"] = T.s20["anchor"][sel].astype(np.int32)
        return out


def run(subj, cells, out_dir, s2s):
    D = VE.DataUS(subj); T = Text(); ws = WS()
    for cell in cells:
        cond, rung = parse_cell(cell); t_cell = time.time(); tl = D.load(cond)
        rs = rung_stories(rung, s2s); rung_st = [s for s in D.train_stories if s in rs]; assert len(rung_st) == len(rs), (rung, rs)
        assert len(rung_st) > 1 and set(rung_st) == set(D.train_stories), "LOSO over all training stories only (n82)"
        trials = []
        for q in QS:                                                   # tuning: verify_e.run_subject verbatim
            qq = min(q, D.V - 1)
            mu, comps, _ = pca_fit(D, ranges_for(D, rung_st), qq, seed=seed_of(subj, cond, rung, "pca")); F = project(D, mu, comps)
            X, E, S, A = build_xy(F, D, T, rung_st); Xv, Ev, Sv, Av = build_xy(F, D, T, [VE.CALIB])
            for lam in LAMS:
                m, norm, vl, bep, nep = train(X, E, S, A, Xv, Ev, Sv, Av, lam, seed=seed_of(subj, cond, rung, q, lam))
                trials.append({"q": qq, "lam": lam, "val_loss": vl, "best_step": bep, "steps": nep})
                print(f"[{subj} {cell}] tune q={qq} lam={lam:g} val={vl:.4f} step={bep}/{nep}", flush=True)
            del F, X
        bt = min(trials, key=lambda x: x["val_loss"]); q, lam = bt["q"], bt["lam"]
        flat = {}
        for held in rung_st:                                            # the LOSO models: verify_e.run_subject verbatim (name = held)
            tr = [s for s in rung_st if s != held]
            mu, comps, _ = pca_fit(D, ranges_for(D, tr), q, seed=seed_of(subj, cond, rung, "pca", held)); F = project(D, mu, comps)
            X, E, S, A = build_xy(F, D, T, tr); Xv, Ev, Sv, Av = build_xy(F, D, T, [VE.CALIB])
            m, norm, vl, bep, nep = train(X, E, S, A, Xv, Ev, Sv, Av, lam, seed=seed_of(subj, cond, rung, held)); del X
            o = ScorerWS(D, T, F, m, norm).score20ws(held, held, seed_of(subj, cond, rung, held, "20"), ws)
            for k, v in o.items(): flat[f"{held}|{k}"] = v
            del F, m
        np.savez_compressed(f"{out_dir}/{subj}_ws_{cell}.npz", **flat)
        rb = lambda r, rs_, P: np.log2((P + 1) / r) - np.log2((P + 1) / rs_)
        b1 = np.concatenate([rb(flat[f"{s}|rank2"], flat[f"{s}|rank2_sh"], flat[f"{s}|npool"]) for s in rung_st])
        b2 = np.concatenate([rb(flat[f"{s}|rank2_ws"], flat[f"{s}|rank2_ws_sh"], flat[f"{s}|npool_ws"]) for s in rung_st])
        res = {"cond": cond, "rung": rung, "q": q, "lam": lam, "trials": trials, "V_set": int(D.V), "stories": rung_st, "n_windows": int(len(b1)),
               "bits_other_story_pool": float(b1.mean()), "bits_same_story_pool": float(b2.mean()), "load_s": tl, "wall_s": time.time() - t_cell}
        json.dump({cell: res}, open(f"{out_dir}/{subj}_ws_{cell}_cell.json", "w"), indent=1)
        print(f"[{subj} {cell}] other-story {b1.mean():.3f} bits, same-story {b2.mean():.3f} bits, {len(b1)} windows, t={time.time() - t_cell:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--cells", required=True)
    ap.add_argument("--out", default=f"{VP.WORK}/results_C_us"); ap.add_argument("--preset", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    VE.PRESET = np.load(os.path.expanduser(a.preset)).astype(np.int64)
    s2s = json.load(open(f"{VP.TANG_BOX}/data_train/sess_to_story.json"))
    run(a.subject, a.cells.split(","), a.out, s2s)
