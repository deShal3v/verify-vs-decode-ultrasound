"""Per-voxel native noise ceiling SNR of the test story (10 single-trial repeats), streamed from OpenNeuro over HTTP
(parallel HTTP range requests into RAM; nothing is written to disk except the per-voxel result). Identical definitions to Phase B prep_resp.py:
  var_n = mean_t var_r(x, ddof=1)   (noise variance; repeats were z-scored per run)
  var_tot = mean_r var_t(x)         (total variance)
  var_s = clip(var_tot - var_n, 0); ncsnr = sqrt(var_s / var_n)
Additionally: the unclipped ratio r = (var_tot - var_n)/var_n (= ncsnr^2 before clipping) and its leave-one-repeat-out
jackknife standard error (repeats are independent runs, so the jackknife is valid under temporal autocorrelation).
Usage: python ncsnr_stream.py S1 [S2 S3]  -> $WORK_DIR/geometry/ncsnr_<S>.npz"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, time, json, numpy as np, h5py, fsspec
OUT = f"{VP.WORK}/geometry"; os.makedirs(OUT, exist_ok=True)
UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
URL = "https://s3.amazonaws.com/openneuro.org/ds003020/derivatives/preprocessed_data/{u}/wheretheressmoke.hf5"

def fetch(url, off, n, parts=16 * 2**20, threads=16):
    """parallel HTTP range GETs of n bytes at offset off -> bytes"""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    buf = bytearray(n); rngs = [(a, min(a + parts, n)) for a in range(0, n, parts)]
    def get(ab):
        a, b = ab
        for k in range(5):
            try:
                r = requests.get(url, headers={"Range": f"bytes={off + a}-{off + b - 1}"}, timeout=120); r.raise_for_status()
                assert len(r.content) == b - a; buf[a:b] = r.content; return
            except Exception as e:
                if k == 4: raise
    with ThreadPoolExecutor(threads) as ex: list(ex.map(get, rngs))
    return bytes(buf)

def main(subj):
    t0 = time.time(); url = URL.format(u=UTS[subj])
    h = h5py.File(fsspec.open(url, block_size=2**20, cache_type="bytes").open(), "r"); ds = h["individual_repeats"]
    m, T, V = ds.shape; assert m == 10 and ds.dtype == np.dtype("<f8") and ds.chunks is None and ds.compression is None
    off = ds.id.get_offset(); nbytes = m * T * V * 8; assert ds.id.get_storage_size() == nbytes
    X = np.empty((m, T, V), np.float32)
    for r in range(m):                                               # one repeat at a time (189-230 MB float64)
        X[r] = np.frombuffer(fetch(url, off + r * T * V * 8, T * V * 8), dtype="<f8").reshape(T, V)
        print(f"{subj} repeat {r} fetched {time.time() - t0:.0f}s", flush=True)
    # sanity: one slab read through h5py equals the raw bytes
    assert np.allclose(np.asarray(ds[3, 100:102, :50], np.float32), X[3, 100:102, :50])
    X = X.astype(np.float64)                                          # prep_resp: float32 data, statistics in float64-ish
    S = X.sum(0); Q = (X ** 2).sum(0)                                # (T, V)
    vn_sum = ((Q - S ** 2 / m) / (m - 1)).sum(0)
    vn_loo = np.stack([(((Q - X[i] ** 2) - (S - X[i]) ** 2 / (m - 1)) / (m - 2)).sum(0) for i in range(m)])
    s1 = X.sum(1); s2 = (X ** 2).sum(1); del X, S, Q
    var_n = vn_sum / T
    var_t = s2 / T - (s1 / T) ** 2                                   # (m, V) per-repeat variance over time, ddof=0
    var_tot = var_t.mean(0)
    r = (var_tot - var_n) / np.maximum(var_n, 1e-6)
    ncsnr = np.sqrt(np.clip(var_tot - var_n, 0, None) / np.maximum(var_n, 1e-6))
    r_loo = np.empty((m, V))
    for i in range(m):
        vt_i = (var_t.sum(0) - var_t[i]) / (m - 1); vn_i = vn_loo[i] / T
        r_loo[i] = (vt_i - vn_i) / np.maximum(vn_i, 1e-6)
    se_jack = np.sqrt((m - 1) / m * ((r_loo - r_loo.mean(0)) ** 2).sum(0))
    np.savez_compressed(f"{OUT}/ncsnr_{subj}.npz", var_n=var_n.astype(np.float32), var_tot=var_tot.astype(np.float32), r=r.astype(np.float32),
                        ncsnr=ncsnr.astype(np.float32), se_jack=se_jack.astype(np.float32), r_loo=r_loo.astype(np.float32))
    info = {"subject": subj, "V": int(V), "T": int(T), "ncsnr_median": float(np.median(ncsnr)), "ncsnr_p90": float(np.percentile(ncsnr, 90)),
            "frac_r_le_0": float(np.mean(r <= 0)), "median_se_jack": float(np.median(se_jack)), "median_r": float(np.median(r)), "s": time.time() - t0}
    json.dump(info, open(f"{OUT}/ncsnr_{subj}.json", "w"), indent=1); print(json.dumps(info), flush=True)

if __name__ == "__main__":
    for s in sys.argv[1:]: main(s)
