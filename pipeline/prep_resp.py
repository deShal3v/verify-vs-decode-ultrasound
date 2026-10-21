"""Per subject: hf5 (float64) -> one float16 matrix on $WORK_DIR/<subj>/ plus index tables, ncsnr, alignment check.
Rows: 82 training stories in Tang's session order, then calibration story, then the 10 test repeats
(ds003020 individual_repeats; repeat 0 == the ds004510 single trial that Tang decodes).
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, numpy as np, h5py
SUBJ = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
SESS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
CALIB, TEST = "fromboyhoodtofatherhood", "wheretheressmoke"
D = f"{VP.DS}/preprocessed_data"

def main(subj):
    t0 = time.time()
    s2s = json.load(open(f"{VP.TANG_BOX}/data_train/sess_to_story.json"))
    stories = [st for s in SESS for st in s2s[str(s)]]
    sess_of = [s for s in SESS for st in s2s[str(s)]]
    out = f"{VP.WORK}/{subj}"; os.makedirs(out, exist_ok=True)
    sizes = []
    for st in stories + [CALIB]:
        with h5py.File(f"{D}/{SUBJ[subj]}/{st}.hf5", "r") as f: sizes.append(f["data"].shape)
    V = sizes[0][1]; assert all(s[1] == V for s in sizes)
    n_rows = [s[0] for s in sizes]
    N = sum(n_rows) + 10 * 291
    R = np.lib.format.open_memmap(f"{out}/resp_f16.npy", mode="w+", dtype=np.float16, shape=(N, V))
    off = 0; index = []
    for st, n, ss in zip(stories + [CALIB], n_rows, sess_of + [4]):
        with h5py.File(f"{D}/{SUBJ[subj]}/{st}.hf5", "r") as f: x = np.nan_to_num(f["data"][:]).astype(np.float32)
        R[off:off + n] = x.astype(np.float16)
        index.append({"story": st, "start": off, "n": n, "session": ss, "kind": "train" if st != CALIB else "calib",
                      "mean_abs_mean": float(np.abs(x.mean(0)).mean()), "mean_std": float(x.std(0).mean())})
        off += n
    with h5py.File(f"{D}/{SUBJ[subj]}/{TEST}.hf5", "r") as f:
        reps = np.nan_to_num(f["individual_repeats"][:]).astype(np.float32)   # (10, 291, V)
    for r in range(10):
        R[off:off + 291] = reps[r].astype(np.float16)
        index.append({"story": TEST, "start": off, "n": 291, "session": r + 2, "kind": f"test_rep{r}"}); off += 291
    R.flush()
    # ncsnr from the 10 repeats (each repeat z-scored per run): noise var = mean_t var_r; total var = mean_r var_t
    var_n = reps.var(0, ddof=1).mean(0)
    var_tot = reps.var(1).mean(0)
    var_s = np.clip(var_tot - var_n, 0, None)
    ncsnr = np.sqrt(var_s / np.maximum(var_n, 1e-6))
    np.save(f"{out}/ncsnr.npy", ncsnr.astype(np.float32)); np.save(f"{out}/var_noise.npy", var_n.astype(np.float32))
    # alignment check: word count per row vs auditory-ROI mean response at row lags
    sys.path.insert(0, VP.TANG_CODE)
    from utils_ridge.textgrid import TextGrid
    from utils_ridge.stimulus_utils import load_simulated_trfiles
    from utils_ridge.dsutils import make_word_ds
    rd = json.load(open(f"{VP.DS}/respdict.json"))
    roi = json.load(open(f"{VP.TANG_BOX}/data_train/ROIs/{subj}.json"))["auditory"]
    lagc = {L: [] for L in range(-3, 8)}
    for e in index[:82]:
        st = e["story"]
        grids = {st: TextGrid(open(f"{VP.DS}/TextGrids/{st}.TextGrid").read())}
        ws = make_word_ds(grids, load_simulated_trfiles({st: rd[st]}))[st]
        assert e["n"] == rd[st] - 20, (st, e["n"], rd[st])
        t = np.asarray(ws.data_times); rows = np.arange(e["n"])
        rate = np.array([np.sum((t >= 2 * i + 10) & (t < 2 * i + 12)) for i in rows], float)   # words heard during row i's TR
        a = R[e["start"]:e["start"] + e["n"], roi].astype(np.float32).mean(1)
        for L in lagc:
            if L >= 0: x, y = rate[:e["n"] - L], a[L:]
            else: x, y = rate[-L:], a[:e["n"] + L]
            lagc[L].append(np.corrcoef(x, y)[0, 1])
    lag = {L: float(np.nanmean(v)) for L, v in lagc.items()}
    info = {"subject": subj, "V": int(V), "N": int(N), "index": index, "n_train_rows": int(sum(n_rows[:82])),
            "ncsnr_median": float(np.median(ncsnr)), "ncsnr_p90": float(np.percentile(ncsnr, 90)),
            "frac_vox_ncsnr_gt_0.2": float(np.mean(ncsnr > 0.2)), "wordrate_auditory_corr_by_row_lag": lag,
            "prep_s": time.time() - t0}
    json.dump(info, open(f"{out}/index.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "index"}, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
