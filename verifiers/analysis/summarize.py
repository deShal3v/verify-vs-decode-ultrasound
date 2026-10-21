"""Phase E4 copy of summ_e3.py (paths phaseE4/rescue -> phaseE4/results_e4 only). Phase E3 copy of summ_e2.py (paths phaseE3/rescue -> phaseE3/results_e3 only). Phase E2 copy (paths phaseE2/rescue -> phaseE2/results_e2 only). Phase E: Phase D summ_d.py run on phaseE/rescue (Phase D docstring follows). Phase D: Phase C summ_c.py run on phaseD/rescue (paths changed; text side falls back to Phase C's checked text metadata;
large-null decoder bits bits20_big.npz added to the windows file as T|<cell>|bitsBig / bitsGBig / bits_null_mean_big / n_null_big).
Original docstring: Phase C per-subject inputs for the final analysis, built on this box from rescue/ (CPU).
Writes results_final/<S>_summary.json + <S>_windows.npz in Phase B's summ_b.py/rebuild_b.py format, plus the add-ons.
Metric definitions: Phase B analyze.py (via cpu_prelim/code/analyze_b.py, identical except the configurable text path),
Phase B's hard set (foil GTR distance >= 0.31), per-story arrays for the story-cluster bootstrap (82 training stories in
Tang's order; n = 0 for a story a cell does not score, e.g. native_n1's own story).
Text side: each subject's verifier node's own win6/seg20 metadata (text_meta.npz), checked against Phase A/B text_side.npz.
Release note: one script for every phase (the paper ran it as summ_e.py, summ_e2.py, summ_e3.py, summ_e4.py, which differ only
in these paths). --rescue = a directory with one sub-directory per GPU machine, each holding that machine's $DATA_DIR/out
(pipeline/collect_out_e.sh), i.e. <rescue>/<machine>/<S>/work/... and <rescue>/<machine>/<S>/models/<S>/<cell>/...
Run: python summarize.py --rescue $WORK_DIR/rescue/E4 --out $WORK_DIR/results/E4"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, glob, re, hashlib, argparse, numpy as np
_ap = argparse.ArgumentParser(); _ap.add_argument("--rescue", required=True); _ap.add_argument("--out", required=True)
_ap.add_argument("--text_dir", default=None, help="text side (win6.npz, seg20.npz) when no node text metadata is found; may contain {S} (default: $WORK_DIR/text)")
_A = _ap.parse_args()
RS = os.path.abspath(_A.rescue); PC = os.path.dirname(RS); OUT = os.path.abspath(_A.out); TEXT_FALLBACK = _A.text_dir or f"{VP.WORK}/text"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_b as AB
from analyze_b import cell_windows, metrics
def seed_of(*xs): return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)
HARD = 0.31; KSUB = {1: 10, 2: 5, 5: 2, 10: 1}; NB = 1000; BLOCK = 3; SUBJ = ["S1", "S2", "S3"]
os.makedirs(OUT, exist_ok=True)


def text_for(S):
    """write the node's text metadata as win6.npz/seg20.npz into results_final/text_<S>/ and compare with text_side.npz"""
    src = sorted(glob.glob(f"{RS}/*/{S}/work/results_C_us/{S}_text_meta.npz"))
    d = f"{OUT}/text_{S}"; os.makedirs(d, exist_ok=True); chk = {"source": src[0] if src else None}
    if not src:                                                             # Phase D: use Phase C's checked text side (identical on every node)
        dc = TEXT_FALLBACK.replace("{S}", S); chk["source"] = dc + " (text side of the run)"; return (dc if os.path.isdir(dc) else None), chk
    z = np.load(src[0]); refp = f"{VP.WORK}/text_side.npz"; ref = np.load(refp) if os.path.exists(refp) else None   # optional reference text side
    for pre, fn in (("w6", "win6"), ("s20", "seg20")):
        keys = [k[len(pre) + 1:] for k in z.files if k.startswith(pre + "_")]
        np.savez(f"{d}/{fn}.npz", **{k: z[f"{pre}_{k}"] for k in keys})
        if ref is None: continue
        for k in ("story", "anchor", "nw", "foil_valid"):
            chk[f"{pre}_{k}_equal_text_side"] = bool(np.array_equal(z[f"{pre}_{k}"], ref[f"{pre}_{k}"]))
        chk[f"{pre}_foil_d_maxabsdiff_text_side"] = float(np.nanmax(np.abs(z[f"{pre}_foil_d"].astype(float) - ref[f"{pre}_foil_d"].astype(float))))
    return d, chk


def hard_ba(w, t):
    sel = w["sel"]; d = t["foil_d"][sel]; v = t["foil_valid"][sel] & (d >= HARD)
    hit = (w["s_true_tn"] > w["thtn"]).astype(float); fa = (w["s_foil_tn"].astype(np.float64) > w["thtn"][:, None])
    return {"hit_rate_tn": float(hit.mean()), "fa_tn_hard": float(fa[v].mean()) if v.any() else None,
            "yesno_ba_tn_hard": float((hit.mean() + 1 - fa[v].mean()) / 2) if v.any() else None, "n_hard_foils": int(v.sum())}


def per_story(w, nst):
    """per-story sums for the story-cluster bootstrap, full length nst (n = 0 where the story is not scored)"""
    rb = np.log2((w["npool"] + 1) / w["rank2"]) - np.log2((w["npool"] + 1) / w["rank2_sh"]); blk = w["block"]
    hit = (w["s_true_tn"] > w["thtn"]).astype(float); fa = (w["s_foil_tn"].astype(np.float64) > w["thtn"][:, None]); win = (w["s_true"][:, None].astype(np.float64) > w["s_foil"].astype(np.float64))
    o = {k: np.zeros(nst) for k in ("n", "rb_sum", "hit_sum", "fa_sum", "fa_n", "win_sum")}
    return o, (rb, hit, fa, win, blk)


def story_arrays(w, tx, nst):
    sel = w["sel"]; v = tx["foil_valid"][sel]; o, (rb, hit, fa, win, blk) = per_story(w, nst)
    for u in np.unique(blk):
        m = blk == u; o["n"][u] = m.sum(); o["rb_sum"][u] = rb[m].sum(); o["hit_sum"][u] = hit[m].sum()
        o["fa_sum"][u] = fa[m][v[m]].sum(); o["fa_n"][u] = v[m].sum(); o["win_sum"][u] = win[m][v[m]].sum()
    return {k: v.tolist() for k, v in o.items()}


def summarize(z, tx, train_st, rng, test_only=False):
    e = {}
    for rd in ("6", "20"):
        if not test_only and any(f"{rd}|{s}|sel" in z.files for s in train_st):
            w = cell_windows(z, rd, train_st); e[f"loso_{rd}"] = metrics(w, tx[rd], rng); e[f"loso_{rd}"]["hard"] = hard_ba(w, tx[rd])
            e[f"loso_{rd}_story"] = story_arrays(w, tx[rd], len(train_st))
        if f"{rd}|test_rep0|sel" in z.files:
            w0 = cell_windows(z, rd, ["test_rep0"]); e[f"test_rep0_{rd}"] = metrics(w0, tx[rd], rng, boot=False); e[f"test_rep0_{rd}"]["hard"] = hard_ba(w0, tx[rd])
            reps = [metrics(cell_windows(z, rd, [f"test_rep{r}"]), tx[rd], rng, boot=False) for r in range(10) if f"{rd}|test_rep{r}|sel" in z.files]
            e[f"test_10reps_{rd}"] = {m: [float(np.mean([x[m] for x in reps])), float(np.std([x[m] for x in reps]))] for m in ["afc2_pooled", "yesno_ba_tn_pooled", "rank_bits_p2_gain", "top1_p2"]}
    return e


def test_arrays(z, tx, W, tag):
    for rd in ("20", "6"):
        if f"{rd}|test_rep0|sel" not in z.files: continue
        w0 = cell_windows(z, rd, ["test_rep0"]); rb = np.log2((w0["npool"] + 1) / w0["rank2"]) - np.log2((w0["npool"] + 1) / w0["rank2_sh"])
        W[f"{tag}|{rd}|test_rep0|rank_bits_gain"] = rb.astype(np.float32); W[f"{tag}|{rd}|test_rep0|anchor"] = tx[rd]["anchor"][w0["sel"]].astype(np.int16)
        # per-window arrays over the 10 single repeats (for the test-story readout and its moving-block bootstrap)
        reps = [r for r in range(10) if f"{rd}|test_rep{r}|sel" in z.files]; RB, HIT, FA, FAN, HFA, HFAN, WIN = [], [], [], [], [], [], []
        for r in reps:
            w = cell_windows(z, rd, [f"test_rep{r}"]); assert np.array_equal(w["sel"], w0["sel"]); v = tx[rd]["foil_valid"][w["sel"]]; hv = v & (tx[rd]["foil_d"][w["sel"]] >= HARD)
            RB.append(np.log2((w["npool"] + 1) / w["rank2"]) - np.log2((w["npool"] + 1) / w["rank2_sh"])); HIT.append((w["s_true_tn"] > w["thtn"]).astype(float))
            fa = (w["s_foil_tn"].astype(np.float64) > w["thtn"][:, None]); FA.append((fa & v).sum(1)); FAN.append(v.sum(1)); HFA.append((fa & hv).sum(1)); HFAN.append(hv.sum(1))
            WIN.append(((w["s_true"][:, None].astype(np.float64) > w["s_foil"].astype(np.float64)) & v).sum(1))
        for nm, a in (("rb", RB), ("hit", HIT), ("fa", FA), ("fan", FAN), ("hfa", HFA), ("hfan", HFAN), ("win", WIN)):
            W[f"{tag}|{rd}|test10|{nm}"] = np.stack(a).astype(np.float32)          # (n_reps, n_windows)


# ---------------------------------------------------------------- add-ons (as cpu_prelim/code/analyze_prelim.py)
def per_window(w, tx):
    sel = w["sel"]; v = tx["foil_valid"][sel]
    rb = np.log2((w["npool"] + 1) / w["rank2"]) - np.log2((w["npool"] + 1) / w["rank2_sh"])
    hit = (w["s_true_tn"] > w["thtn"]).astype(float); fa = (w["s_foil_tn"].astype(np.float64) > w["thtn"][:, None])
    win = (w["s_true"][:, None].astype(np.float64) > w["s_foil"].astype(np.float64)); return rb, hit, fa, win, v


def agg(pw, ii):
    rb, hit, fa, win, v = pw; vv = v[ii]; return rb[ii].mean(), (hit[ii].mean() + 1 - fa[ii][vv].mean()) / 2, win[ii][vv].mean()


def summarise_k(z, rd, tx, k, rng, prefix=""):
    evs = [f"{prefix}test_rep{r}" for r in range(10)] if k == 1 else [f"{prefix}test_avg_k{k}_s{s}" for s in range(KSUB[k])]
    evs = [e for e in evs if f"{rd}|{e}|sel" in z.files]
    if not evs: return None
    W_ = [cell_windows(z, rd, [e]) for e in evs]; PW = [per_window(w, tx) for w in W_]; n = len(W_[0]["sel"])
    assert all(np.array_equal(w["sel"], W_[0]["sel"]) for w in W_)
    vals = np.array([agg(pw, np.arange(n)) for pw in PW]); out = {"n_subsets": len(evs), "n_windows": int(n)}
    nb = int(np.ceil(n / BLOCK)); bs = []
    for _ in range(NB):
        st = rng.integers(0, n - BLOCK + 1, nb); ii = (st[:, None] + np.arange(BLOCK)[None]).ravel()[:n]; bs.append(np.mean([agg(pw, ii) for pw in PW], 0))
    bs = np.array(bs)
    for j, nm in enumerate(("bits", "ba_tn", "afc2")):
        out[nm] = {"mean": float(vals[:, j].mean()), "sd": float(vals[:, j].std(ddof=1)) if len(vals) > 1 else None, "per_subset": vals[:, j].round(4).tolist(),
                   "ci95": np.percentile(bs[:, j], [2.5, 97.5]).tolist()}
    out["p_bits_ge_1_boot"] = float((bs[:, 0] >= 1).mean())
    out["hard_ba_tn"] = float(np.mean([hard_ba(w, tx)["yesno_ba_tn_hard"] for w in W_]))
    return out


def addons(S, tx, rng):
    A = {}
    for fam, sub in (("C", "results_C_us"), ("ETC", "results_etc600")):
        for cj in sorted(glob.glob(f"{RS}/*/{S}/work/{sub}/{S}_addon_*_cell.json")):
            name = os.path.basename(cj)[len(S) + 7:-len("_cell.json")]
            if name in A.get(fam, {}): continue
            info = json.load(open(cj)); z = np.load(cj.replace("_cell.json", ".npz"))
            prefixes = ["", "us_", "usN_"] if name.startswith("p90to50") else [""]
            e = {"info": {k: v for k, v in info.items() if k not in ("trials",)}, "readout": {}}
            for pre in prefixes:
                key = ("train_condition" if pre == "" else pre.rstrip("_")) if name.startswith("p90to50") else "kavg"
                e["readout"][key] = {rd: {str(k): summarise_k(z, rd, tx[rd], k, np.random.default_rng(seed_of(S, fam, name, pre, rd, k)), pre) for k in KSUB} for rd in ("20", "6")}
            A.setdefault(fam, {})[name] = e
    return A


def main():
    inv = {}
    for si, S in enumerate(SUBJ):
        d, chk = text_for(S)
        if d is None: print(S, "no text meta yet"); continue
        AB.TXT = d; tx = AB.load_text(); stories = list(tx["6"]["stories"]); train_st = stories[:82]; rng = np.random.default_rng(si)
        out = {"subject": S, "text_check": chk, "train_stories": train_st}; W = {}; I = {}
        for fam, sub in (("C", "results_C_us"), ("ETC", "results_etc600")):
            res = {}
            for cj in sorted(glob.glob(f"{RS}/*/{S}/work/{sub}/{S}_*_cell.json")):
                if "_addon_" in cj: continue
                meta = json.load(open(cj)); cell = list(meta)[0]; m = meta[cell]; p = cj.replace("_cell.json", ".npz")
                if not os.path.exists(p) or cell in res: continue          # a cell copied between nodes: first copy only
                z = np.load(p); e = {k: m.get(k) for k in ("cond", "rung", "dB", "ultrasound", "V_set", "n_voxels", "rung_stories", "hours", "n_train_stories", "q", "lam",
                                                          "n_models", "loso", "gpu_s", "wall_s", "pool_K", "kappa")}
                e["calib"] = m.get("calib"); e["source"] = cj[len(PC) + 1:]
                e.update(summarize(z, tx, train_st, rng)); test_arrays(z, tx, W, f"{fam}|{cell}"); res[cell] = e
                print(S, fam, cell, {rd: (round(e[f"loso_{rd}"]["yesno_ba_tn_pooled"], 3), round(e[f"loso_{rd}"]["rank_bits_p2_gain"], 2)) for rd in ("20",) if f"loso_{rd}" in e}, flush=True)
            out[fam] = res; I[fam] = sorted(res)
        # Tang cells (decode nodes) + every fit_info (decode and verifier nodes: duplicate refits are compared)
        T = {}; fits = {}
        for dd in sorted(glob.glob(f"{RS}/*/{S}/models/{S}/*")):
            cell = os.path.basename(dd); node = dd[len(RS) + 1:].split("/")[0]
            if os.path.exists(f"{dd}/fit_info.json"): fits.setdefault(cell, {})[node] = {k: v for k, v in json.load(open(f"{dd}/fit_info.json")).items() if k not in ("stories", "alpha_hist_sel")}
            # E4c (2026-10-21): take the first COMPLETE copy (eval.json and bits20.npz); after the 2026-10-20 disk-full incident some E4 node copies
            # have eval.json without bits20.npz, and their rescored copies come home from the E4c nodes (rescore_e4c.sh).
            if not (os.path.exists(f"{dd}/eval.json") and os.path.exists(f"{dd}/bits20.npz")) or cell in T: continue
            e = {"source_dir": dd[len(PC) + 1:]}
            for f in ("eval.json", "bits20.json", "fit_info.json", "decoded_timing.json"):
                if os.path.exists(f"{dd}/{f}"): e[f[:-5]] = json.load(open(f"{dd}/{f}"))
            if os.path.exists(f"{dd}/bits20.npz"):
                zz = np.load(f"{dd}/bits20.npz")
                for k in ("bits", "bitsG", "bert_real", "anchors", "wer"): W[f"T|{cell}|{k}"] = zz[k]
                W[f"T|{cell}|bits_null_mean"] = zz["bits_null"].mean(1); W[f"T|{cell}|bits_null"] = zz["bits_null"].astype(np.float32); W[f"T|{cell}|bert_null"] = zz["bert_null"].astype(np.float32)
            bj = sorted(glob.glob(f"{dd}/bits20_big*.npz"))
            if bj:
                zz = np.load(bj[-1]); W[f"T|{cell}|bitsBig"] = zz["bits"]; W[f"T|{cell}|bitsGBig"] = zz["bitsG"]; W[f"T|{cell}|bits_null_mean_big"] = zz["bits_null"].mean(1)
                W[f"T|{cell}|n_null_big"] = np.array(zz["bert_null"].shape[1]); W[f"T|{cell}|anchors_big"] = zz["anchors"]; e["bits20_big"] = json.load(open(bj[-1].replace(".npz", ".json")))
            if os.path.exists(f"{dd}/eval_windows.npz"):
                zz = np.load(f"{dd}/eval_windows.npz"); W[f"T|{cell}|BERT_window"] = zz["BERT_window"]; W[f"T|{cell}|BERT_null_window"] = zz["BERT_null_window"].astype(np.float32)
            if os.path.exists(f"{dd}/decoded.npz"):
                dz = np.load(f"{dd}/decoded.npz"); e["decoded_first_words"] = " ".join(list(dz["words"])[:60]); e["n_decoded_words"] = int(len(dz["words"]))
            T[cell] = e
        big_info = {}                                                           # Phase D large-null bits, run in <cell>_big/ dirs on a null node
        for bj in sorted(glob.glob(f"{RS}/*/{S}/models/{S}/*_big/bits20_big.npz")):
            cell = os.path.basename(os.path.dirname(bj))[:-4]
            if f"T|{cell}|bitsBig" in W: continue
            zz = np.load(bj); W[f"T|{cell}|bitsBig"] = zz["bits"]; W[f"T|{cell}|bitsGBig"] = zz["bitsG"]; W[f"T|{cell}|bits_null_mean_big"] = zz["bits_null"].mean(1)
            W[f"T|{cell}|n_null_big"] = np.array(zz["bert_null"].shape[1]); W[f"T|{cell}|anchors_big"] = zz["anchors"]; big_info[cell] = json.load(open(bj.replace(".npz", ".json")))
        out["bits20_big"] = big_info; I["bits20_big"] = sorted(big_info)
        out["Tang"] = T; out["fit_info"] = fits; I["Tang"] = sorted(T); I["fits"] = sorted(fits)
        out["addon"] = addons(S, tx, rng); I["addon"] = {f: sorted(v) for f, v in out["addon"].items()}
        st = {}
        for p in sorted(glob.glob(f"{RS}/*/{S}/work/{S}/phase[cde]_status*.json")):
            j = json.load(open(p)); st[p[len(RS) + 1:]] = {"counts": j["counts"], "failed": {k: v["note"] for k, v in j["jobs"].items() if v["state"] == "failed"},
                                                         "skipped": {k: v["note"] for k, v in j["jobs"].items() if v["state"] == "skipped"},
                                                         "runtime_s": {k: (v["t_end"] - v["t_start"]) for k, v in j["jobs"].items() if v.get("t_end") and v.get("t_start")}}
        out["queue_status"] = st
        json.dump(out, open(f"{OUT}/{S}_summary.json", "w"), indent=1); np.savez_compressed(f"{OUT}/{S}_windows.npz", **W); inv[S] = I
        print(S, "summary written:", {k: len(v) for k, v in I.items()}, flush=True)
    json.dump(inv, open(f"{OUT}/inventory.json", "w"), indent=1)


if __name__ == "__main__":
    main()
