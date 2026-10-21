"""Phase E4 within-story control (2026-10-21, Omer): 20-s candidate windows of every story at EVERY anchor (2-s steps), each as the 10
sub-window GTR embeddings built exactly as prep_text_words.py builds a segment's true text (words in (T-20, T], T = 2a+12, binned into the
6-s windows ending at 2b+12, b = a-9..a; the same GTR call, gtr.embed(flat)).
Check: at the segment anchors of seg20 (every 5 TRs) the embeddings must equal seg20_sub[:, 0] (written to the check json).
Output: $WORK_DIR/text/ws20.npz (story, anchor, nw, sub_mask) + $WORK_DIR/text/ws20_sub.npy (N, 10, 768) fp16;
check: $WORK_DIR/results_C_us/<S>_ws_textcheck.json. Usage: python ws_text.py <S>"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_text import load_words, GTR, OUT


def main(S):
    t0 = time.time(); stories, W = load_words(); sid = {st: i for i, st in enumerate(stories)}
    rec = dict(story=[], anchor=[], nw=[], sub_mask=[]); flat = []
    for st in stories:
        words, times, n = W[st]
        for a in range(9, n - 4):
            T = 2 * a + 12.0; ii = np.where((times > T - 20) & (times <= T))[0]; t = times[ii]
            bins = [np.where((t > 2 * b + 12.0 - 6) & (t <= 2 * b + 12.0))[0] for b in range(a - 9, a + 1)]
            rec["story"].append(sid[st]); rec["anchor"].append(a); rec["nw"].append(len(ii)); rec["sub_mask"].append([len(bb) > 0 for bb in bins])
            flat += [" ".join(words[ii[j]] for j in bb) for bb in bins]
    print(f"{len(rec['story'])} windows, {len(flat)} sub-window texts, t={time.time() - t0:.0f}s", flush=True)
    gtr = GTR(); E = np.asarray(gtr.embed(flat)).reshape(len(rec["story"]), 10, 768).astype(np.float16)
    np.save(f"{OUT}/ws20_sub.npy", E); np.savez(f"{OUT}/ws20.npz", **{k: np.asarray(v) for k, v in rec.items()}, stories=np.array(stories))
    # check against the paper's segments
    s20 = np.load(f"{OUT}/seg20.npz"); sub = np.load(f"{OUT}/seg20_sub.npy", mmap_mode="r")
    pos = {(int(s), int(a)): i for i, (s, a) in enumerate(zip(rec["story"], rec["anchor"]))}
    d, nchk, miss = 0.0, 0, 0
    for j, (s, a) in enumerate(zip(s20["story"], s20["anchor"])):
        i = pos.get((int(s), int(a)))
        if i is None: miss += 1; continue
        m = np.asarray(s20["sub_mask"][j], bool)
        if m.any(): d = max(d, float(np.abs(E[i][m].astype(np.float32) - np.asarray(sub[j, 0])[m].astype(np.float32)).max())); nchk += 1
    out = {"n_windows": len(rec["story"]), "n_seg20_checked": nchk, "n_seg20_missing": miss, "max_abs_diff_vs_seg20_true_sub": d, "s": time.time() - t0}
    os.makedirs(f"{VP.WORK}/results_C_us", exist_ok=True); json.dump(out, open(f"{VP.WORK}/results_C_us/{S}_ws_textcheck.json", "w"), indent=1)
    print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
