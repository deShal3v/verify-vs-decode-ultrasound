"""Per-window decoding bits for one Tang cell on the verifier's 20 s test segments (DESIGN.md 3-4).
Windows: (T_a - 20, T_a], T_a = 2a+12, a = 9, 14, ... (Phase A seg20 anchors of the test story with >= 3 words).
Decoded words / null words: those whose predicted times fall in the window; truth: Tang's transcript (load_transcript).
Scores: BERTScore (Phase A BERTm), WER. Output: <cell_dir>/bits20.npz (+ .json summary)."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tang_eval import BERTm, config, load_transcript, wer
from prep_text import load_words

def bits_emp(real, null):
    N = null.shape[1]; b = np.log2((N + 1) / (1 + (null >= real[:, None]).sum(1)))
    bn = np.zeros_like(null)
    for j in range(N):
        others = np.delete(null, j, axis=1); bn[:, j] = np.log2(N / (1 + (others >= null[:, j:j + 1]).sum(1)))
    return b, bn

if __name__ == "__main__":
    cd = sys.argv[1]; NN = int(sys.argv[2]); t0 = time.time()
    s20 = np.load(f"{VP.WORK}/text/seg20.npz"); stories = list(s20["stories"]); sid = stories.index("wheretheressmoke")
    sel = np.where((s20["story"] == sid) & (s20["nw"] >= 3))[0]; anchors = s20["anchor"][sel]; T = 2.0 * anchors + 12.0
    ref = load_transcript("perceived_speech", "wheretheressmoke")
    pred = np.load(f"{cd}/decoded.npz"); pw, pt = list(pred["words"]), pred["times"]
    nulls = json.load(open(f"{cd}/nulls{NN}.json"))
    seg = lambda words, times, x: [w for w, t in zip(words, times) if x - 20 < t <= x]
    refs = [seg(ref["words"], ref["times"], x) for x in T]
    dec = [seg(pw, pt, x) for x in T]
    nul = [[seg(nw, pt, x) for x in T] for nw in nulls]
    bert = BERTm(np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")))
    real = bert.score(refs, dec)
    null = np.concatenate([bert.score([r for _ in range(len(part)) for r in refs], [w for nw in part for w in nw]).reshape(len(part), len(T)) for part in
                           [nul[i:i + 5] for i in range(0, len(nul), 5)]]).T          # (n_win, N)
    b, bn = bits_emp(real, null)
    from scipy.stats import norm
    mu, sd = null.mean(1), null.std(1); bG = -np.log2(np.clip(norm.sf((real - mu) / sd), 1e-300, 1))
    werr = np.array([wer(" ".join(r), " ".join(d) if d else "") if r else 1.0 for r, d in zip(refs, dec)])
    np.savez(f"{cd}/bits20.npz", anchors=anchors, bert_real=real.astype(np.float32), bert_null=null.astype(np.float32), bits=b, bits_null=bn, bitsG=bG, wer=werr)
    out = {"n_windows": len(T), "n_null": NN, "bert_mean": float(real.mean()), "bert_null_mean": float(null.mean()), "bits_emp": float(b.mean()), "bits_emp_null": float(bn.mean()),
           "bits_gain": float(b.mean() - bn.mean()), "bitsG": float(bG.mean()), "bitsG_gain": float(bG.mean() - 1 / np.log(2)), "frac_at_cap": float(np.mean(b >= np.log2(NN + 1) - 1e-9)),
           "story_z_on_segments": float((real.mean() - null.mean(0).mean()) / null.mean(0).std()), "s": time.time() - t0}
    json.dump(out, open(f"{cd}/bits20.json", "w"), indent=1); print(json.dumps(out))
