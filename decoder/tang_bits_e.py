"""Phase E: tang_bits_d.py with a cached null matrix (computed once per subject and null set, saved next to the null file as
null_matrix_<tag>.npz with the word times it was built on; later runs load it when the word times match) and cells that already
have bits20_<tag>.json are skipped, so decodes can be scored as they arrive. Phase D docstring: per-window decoding bits for several Tang cells against ONE large null set (Phase B tang_bits.py arithmetic,
unchanged: same windows, BERTScore settings, empirical bits with the leave-one-null-out null expectation, Gaussian-tail bits).
All cells must share the frozen native-r15 word times (asserted), so the null BERTScore matrix (windows x N) is computed once
and reused. Usage: python tang_bits_d.py <nulls.json> <tag> <cell_dir> [<cell_dir> ...] -> <cell_dir>/bits20_<tag>.npz/json"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tang_eval import BERTm, config, load_transcript, wer
from tang_bits import bits_emp
from scipy.stats import norm

if __name__ == "__main__":
    NP, TAG, cells = sys.argv[1], sys.argv[2], sys.argv[3:]; t0 = time.time()
    s20 = np.load(f"{VP.WORK}/text/seg20.npz"); stories = list(s20["stories"]); sid = stories.index("wheretheressmoke")
    sel = np.where((s20["story"] == sid) & (s20["nw"] >= 3))[0]; anchors = s20["anchor"][sel]; T = 2.0 * anchors + 12.0
    ref = load_transcript("perceived_speech", "wheretheressmoke")
    seg = lambda words, times, x: [w for w, t in zip(words, times) if x - 20 < t <= x]
    refs = [seg(ref["words"], ref["times"], x) for x in T]
    nulls = json.load(open(NP)); NN = len(nulls)
    bert = BERTm(np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")))
    pt0, null, bn = None, None, None
    CACHE = os.path.join(os.path.dirname(NP), f"null_matrix_{TAG}.npz")
    for cd in cells:
        if os.path.exists(f"{cd}/bits20_{TAG}.json"): print(cd, "already scored, skipped", flush=True); continue
        pred = np.load(f"{cd}/decoded.npz"); pw, pt = list(pred["words"]), pred["times"]
        if pt0 is None:
            pt0 = pt
            if os.path.exists(CACHE):
                z = np.load(CACHE)
                if len(z["pt"]) == len(pt) and np.allclose(z["pt"], pt) and z["null"].shape[1] == NN and str(z["nulls"]) == NP:
                    null = z["null"]; print(f"null matrix {null.shape} loaded from cache", flush=True)
            if null is None:
                nul = [[seg(nw, pt, x) for x in T] for nw in nulls]
                null = np.concatenate([bert.score([r for _ in range(len(part)) for r in refs], [w for nw in part for w in nw]).reshape(len(part), len(T)) for part in
                                       [nul[i:i + 5] for i in range(0, len(nul), 5)]]).T          # (n_win, N), shared by every frozen-times cell
                np.savez(CACHE + ".tmp.npz", null=null, pt=pt, nulls=np.array(NP)); os.replace(CACHE + ".tmp.npz", CACHE)
                print(f"null matrix {null.shape} computed in {time.time() - t0:.0f} s and cached", flush=True)
        assert len(pt) == len(pt0) and np.allclose(pt, pt0), f"{cd}: word times differ from the first cell (not frozen)"
        dec = [seg(pw, pt, x) for x in T]; real = bert.score(refs, dec)
        if bn is None: b, bn = bits_emp(real, null)                                  # leave-one-null-out expectation: depends on the nulls only
        else: b = np.log2((NN + 1) / (1 + (null >= real[:, None]).sum(1)))             # = bits_emp(real, null)[0]
        mu, sd = null.mean(1), null.std(1); bG = -np.log2(np.clip(norm.sf((real - mu) / sd), 1e-300, 1))
        werr = np.array([wer(" ".join(r), " ".join(d) if d else "") if r else 1.0 for r, d in zip(refs, dec)])
        np.savez(f"{cd}/bits20_{TAG}.npz", anchors=anchors, bert_real=real.astype(np.float32), bert_null=null.astype(np.float32), bits=b, bits_null=bn, bitsG=bG, wer=werr)
        out = {"n_windows": len(T), "n_null": NN, "nulls": NP, "bert_mean": float(real.mean()), "bert_null_mean": float(null.mean()), "bits_emp": float(b.mean()),
               "bits_emp_null": float(bn.mean()), "bits_gain": float(b.mean() - bn.mean()), "bitsG": float(bG.mean()), "bitsG_gain": float(bG.mean() - 1 / np.log(2)),
               "frac_at_cap": float(np.mean(b >= np.log2(NN + 1) - 1e-9)), "story_z_on_segments": float((real.mean() - null.mean(0).mean()) / null.mean(0).std()), "s": time.time() - t0}
        json.dump(out, open(f"{cd}/bits20_{TAG}.json", "w"), indent=1); print(cd, json.dumps(out), flush=True)
