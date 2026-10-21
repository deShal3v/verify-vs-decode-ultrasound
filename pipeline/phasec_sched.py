"""Phase C node scheduler (one subject per scheduler, N GPUs). Builds the subject's job graph (DESIGN_C.md section 6),
runs jobs on GPUs with a memory-aware packing (at most 2 jobs and 1 Tang decode per GPU), applies the adaptive rules
R1-R4 when their inputs exist, retries a failed job once (exclusive GPU if it ran out of memory), and writes
$WORK_DIR/<S>/phasec_status.json after every change. When nothing is left it writes $WORK_DIR/<S>/QUEUE_DONE and
collects the outputs with their sha256 manifest (collect_out.sh).
Several subjects may share one node: give each scheduler its own --gpu_ids; subject-independent jobs (text side,
candidate features, pool) are lock-protected and run once per node (flock + .done markers).
--part dec = only the decoder chain (fits of decode cells, decodes, scoring); --part ver = everything else (text side,
all fits for E-t-C, E-t-C, C). --only REGEX keeps matching jobs and their dependencies (split one subject's decodes
over several single-GPU nodes). --part all (default) = both. A dec node and a ver node may run the same subject.
Usage: python phasec_sched.py --subject S1 [--gpu_ids 0,1,2,3] [--gpu_mem 22] [--part all] [--dry]"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, subprocess, argparse, re
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable          # the staged code directory and this interpreter
ENV = f"export HF_HOME={VP.HF_HOME} PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd {CODE}; "
GPU_MEM = 22.0                     # GB usable per L4 (24 GB)
MEM = {"text": 8, "feats": 7, "pool": 6, "fit": 14, "fit_us": 10, "decode": 6, "score": 5, "bits": 4, "C": 8, "etc": 10, "merge": 0}

DECODE_CELLS = ["usL90_r15", "usL50_r15", "m18c_r15", "m24c_r15", "native_n3"]     # certain decodes (m24c: was R3, made certain for wall time)
ADAPTIVE_DECODES = {"usL10_r15": "R1", "usL50_r4": "R2"}
FIT_CELLS = ["usL90_r15", "usL50_r15", "usL10_r15", "usL50_r4", "usB90_r15", "usB50_r15", "m12c_r15", "m18c_r15", "m24c_r15",
             "native_r15", "m6c_r15", "native_r4", "native_r1", "native_n3", "native_n2", "native_n1"]
ETC_CELLS = list(FIT_CELLS)
C_CELLS = ["usL90_r15", "usL50_r15", "usL10_r15", "usL50_r4", "usB90_r15", "usB50_r15", "native_n3", "native_n2", "native_n1",
           "usL50lo_r15", "usL50hi_r15", "usL50raw_r15", "usL50cone_r15", "usL90cone_r15"]


def J(id, kind, cmd, deps=(), prio=50, rule=None, mem=None, gpu=True):
    return {"id": id, "kind": kind, "cmd": cmd, "deps": list(deps), "prio": prio, "rule": rule, "mem": MEM[kind] if mem is None else mem,
            "gpu": gpu, "state": "pending", "tries": 0, "gpu_id": None, "t_start": None, "t_end": None, "note": ""}


def once(tag, cmd):
    """run cmd once per node (several schedulers may share the node): flock + done marker"""
    return f"mkdir -p {VP.WORK}/locks; flock {VP.WORK}/locks/{tag}.lock bash -c '[ -f {VP.WORK}/locks/{tag}.done ] || ({cmd} && touch {VP.WORK}/locks/{tag}.done)'"


def make_plan(S, part="all", only=None):
    md = f"{VP.MODELS}/{S}"; P = []
    P.append(J("text", "text", once("text", f"{PY} prep_text_words.py && ln -sfn {VP.WORK}/text {VP.WORK}/text_b"), prio=95))
    for i in range(4): P.append(J(f"feats{i}", "feats", once(f"feats{i}", f"{PY} etc_feats.py --shard {i} --nshards 4"), ["text"], prio=80))
    P.append(J("feats_merge", "merge", once("feats_merge", f"{PY} etc_feats_merge.py"), [f"feats{i}" for i in range(4)], prio=80, gpu=False))
    P.append(J("pool600", "pool", once("pool600", f"{PY} etc_pool600.py --subject {S} --cell native_r15 --build_pool_only"), ["text"], prio=78))
    for c in FIT_CELLS:
        us = c.startswith("us"); pr = 100 if c in DECODE_CELLS else (85 if c in ADAPTIVE_DECODES else 70)
        P.append(J(f"fit_{c}", "fit_us" if us else "fit", f"{PY} tang_fit_us.py --subject {S} --cell {c} --out {md}/{c} --frozen {VP.WORK}/{S}/native_r15_decoded.npz", prio=pr))
    for c in DECODE_CELLS + list(ADAPTIVE_DECODES):
        P.append(J(f"decode_{c}", "decode", f"{PY} tang_decode_frozen.py --cell_dir {md}/{c}", [f"fit_{c}"], prio=90, rule=ADAPTIVE_DECODES.get(c)))
        P.append(J(f"score_{c}", "score", f"cp {VP.WORK}/{S}/nulls200.json {md}/{c}/nulls200.json && {PY} tang_eval2.py score --cell_dir {md}/{c} --n 200", [f"decode_{c}"], prio=92))
        P.append(J(f"bits_{c}", "bits", f"{PY} tang_bits.py {md}/{c} 200", [f"score_{c}"], prio=92))
    for c in ETC_CELLS:
        P.append(J(f"etc_{c}", "etc", f"{PY} etc_pool600.py --subject {S} --cell {c}", [f"fit_{c}", "feats_merge", "pool600"], prio=55 if c.startswith("us") else 50))
    P.append(J("fit_m30c_r15", "fit", f"{PY} tang_fit_us.py --subject {S} --cell m30c_r15 --out {md}/m30c_r15 --frozen {VP.WORK}/{S}/native_r15_decoded.npz", ["etc_m24c_r15"], prio=60, rule="R4"))
    P.append(J("etc_m30c_r15", "etc", f"{PY} etc_pool600.py --subject {S} --cell m30c_r15", ["fit_m30c_r15", "feats_merge", "pool600"], prio=60))
    for c in C_CELLS:
        P.append(J(f"C_{c}", "C", f"{PY} verify_us.py --subject {S} --cells {c}", ["text"], prio=45 if not any(x in c for x in ("lo", "hi", "raw", "cone")) else 30))
    if only:                                                                 # keep matching jobs + their dependency closure
        rx = re.compile(only); keep = {j["id"] for j in P if rx.search(j["id"])}; byid = {j["id"]: j for j in P}; todo = list(keep)
        while todo:
            for d in byid[todo.pop()]["deps"]:
                if d not in keep: keep.add(d); todo.append(d)
        P = [j for j in P if j["id"] in keep]
    if part == "all": return P
    dec = {f"fit_{c}" for c in DECODE_CELLS + list(ADAPTIVE_DECODES)} | {j["id"] for j in P if j["kind"] in ("decode", "score", "bits")}
    if part == "dec": return [j for j in P if j["id"] in dec]
    return [j for j in P if j["id"] not in dec or j["id"].startswith("fit_")]   # ver: refits again (cheap) so E-t-C never waits on another node


# ------------------------------------------------------------------ adaptive rules (DESIGN_C.md 6)
def _load(p):
    try: return json.load(open(p))
    except Exception: return None


def etc_loso20_rank_bits(npz, npool=600):
    """pooled mean over held-out training-story 20 s windows of log2((P+1)/rank) - log2((P+1)/rank_shifted)"""
    z = np.load(npz); g = []
    for k in z.files:
        m = re.match(r"20\|(.+)\|rank2$", k)
        if m and not m.group(1).startswith("test_rep") and m.group(1) != "fromboyhoodtofatherhood":
            r = z[k]; rs = z[f"20|{m.group(1)}|rank2_sh"]; g.append(np.log2((npool + 1) / r) - np.log2((npool + 1) / rs))
    return float(np.concatenate(g).mean()) if g else None


def rule(name, S, root=VP.DATA):
    """True = run, False = skip (rule says not needed), None = not yet decidable"""
    md = f"{root}/models/{S}"
    if name in ("R1", "R2"):
        e = _load(f"{md}/usL50_r15/eval.json"); return None if e is None else bool(e["BERT"]["z"] >= 2.0)
    if name == "R3":
        e = _load(f"{md}/m18c_r15/eval.json"); b = _load(f"{md}/m18c_r15/bits20.json")
        if e is None or b is None: return None
        return bool(e["BERT"]["z"] >= 3.0 or b["bits_gain"] >= 1.0 or b["bitsG_gain"] >= 1.0)
    if name == "R4":
        p = f"{VP.WORK}/results_etc600/{S}_m24c_r15.npz"
        if not os.path.exists(p): return None
        v = etc_loso20_rank_bits(p); return None if v is None else bool(v >= 1.0)
    raise KeyError(name)


# ------------------------------------------------------------------ scheduler
class Sched:
    def __init__(self, S, ngpu, plan, status_path, log_dir, rule_fn=rule, dry=False, poll=15.0, gpu_ids=None, gpu_mem=GPU_MEM):
        self.S, self.ngpu, self.jobs, self.status_path, self.log_dir = S, ngpu, {j["id"]: j for j in plan}, status_path, log_dir
        self.gpu_ids = list(range(ngpu)) if gpu_ids is None else list(gpu_ids); self.gpu_mem = gpu_mem
        self.rule_fn, self.dry, self.poll, self.procs = rule_fn, dry, poll, {}
        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(status_path):                                # resume: keep finished/skipped states
            old = json.load(open(status_path))["jobs"]
            for jid, j in old.items():
                if jid in self.jobs and j["state"] in ("done", "skipped", "failed"): self.jobs[jid].update({k: j[k] for k in ("state", "tries", "t_start", "t_end", "note", "gpu_id")})

    def save(self):
        st = {"subject": self.S, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "jobs": self.jobs,
              "counts": {s: sum(j["state"] == s for j in self.jobs.values()) for s in ("pending", "running", "done", "skipped", "failed")}}
        json.dump(st, open(self.status_path + ".tmp", "w"), indent=1); os.replace(self.status_path + ".tmp", self.status_path)

    def gpu_load(self):
        load = {g: {"mem": 0.0, "n": 0, "decode": 0} for g in range(self.ngpu)}
        for jid, (p, g) in self.procs.items():
            if g is None: continue
            j = self.jobs[jid]; load[g]["mem"] += j["mem"]; load[g]["n"] += 1; load[g]["decode"] += j["kind"] == "decode"
        return load

    def ready(self, j):
        if j["state"] != "pending": return False
        for d in j["deps"]:
            s = self.jobs[d]["state"]
            if s in ("failed", "skipped"):
                j["state"] = "skipped"; j["note"] = f"dependency {d} {s}"; return False
            if s != "done": return False
        if j["rule"]:
            r = self.rule_fn(j["rule"], self.S)
            if r is None: return False
            if r is False:
                j["state"] = "skipped"; j["note"] = f"adaptive rule {j['rule']}: not needed"; self.skip_dependents(j["id"]); return False
            j["note"] = f"adaptive rule {j['rule']}: run"
        return True

    def skip_dependents(self, jid):
        for k in self.jobs.values():
            if jid in k["deps"] and k["state"] == "pending":
                k["state"] = "skipped"; k["note"] = f"dependency {jid} skipped"; self.skip_dependents(k["id"])

    def launch(self, j, g):
        j["state"] = "running"; j["tries"] += 1; j["gpu_id"] = g; j["t_start"] = time.time()
        log = f"{self.log_dir}/{j['id']}.log"
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "" if g is None else str(self.gpu_ids[g])
        cmd = j["cmd"] if self.dry else ENV + j["cmd"]
        p = subprocess.Popen(["bash", "-c", cmd], stdout=open(log, "a"), stderr=subprocess.STDOUT, env=env)
        self.procs[j["id"]] = (p, g)

    def step(self):
        changed = False
        for jid, (p, g) in list(self.procs.items()):
            rc = p.poll()
            if rc is None: continue
            j = self.jobs[jid]; j["t_end"] = time.time(); del self.procs[jid]; changed = True
            if rc == 0: j["state"] = "done"
            else:
                txt = open(f"{self.log_dir}/{jid}.log").read()[-20000:]
                oom = "OutOfMemory" in txt or "out of memory" in txt
                if j["tries"] < 2:
                    j["state"] = "pending"; j["note"] = f"retry after rc={rc}{' (OOM: exclusive GPU)' if oom else ''}"
                    if oom: j["mem"] = self.gpu_mem
                else:
                    j["state"] = "failed"; j["note"] = f"rc={rc}{' OOM' if oom else ''}"; self.skip_dependents(jid)
        load = self.gpu_load()
        cand = sorted([j for j in self.jobs.values() if j["state"] == "pending"], key=lambda j: -j["prio"])
        for j in cand:
            if not self.ready(j): changed = True if j["state"] == "skipped" else changed; continue
            if not j["gpu"]:
                self.launch(j, None); changed = True; continue
            ok = [g for g in range(self.ngpu) if load[g]["n"] < 2 and load[g]["mem"] + min(j["mem"], self.gpu_mem) <= self.gpu_mem and not (j["kind"] == "decode" and load[g]["decode"] >= 1)]
            if not ok: continue
            g = min(ok, key=lambda g: (load[g]["mem"], load[g]["n"]))
            self.launch(j, g); load[g]["mem"] += min(j["mem"], self.gpu_mem); load[g]["n"] += 1; load[g]["decode"] += j["kind"] == "decode"; changed = True
        if changed: self.save()
        return any(j["state"] in ("pending", "running") for j in self.jobs.values())

    def run(self):
        self.save()
        while self.step(): time.sleep(self.poll)
        self.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--gpu_ids", default="0,1,2,3")
    ap.add_argument("--gpu_mem", type=float, default=GPU_MEM); ap.add_argument("--part", default="all", choices=["all", "dec", "ver"]); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default=None, help="regex on job ids (e.g. 'usL50_r15|usL10_r15|usL50_r4'): those jobs + their dependencies")
    a = ap.parse_args(); ids = [int(x) for x in a.gpu_ids.split(",")]
    plan = make_plan(a.subject, a.part, a.only)
    if a.dry:
        for j in plan: print(j["id"], j["kind"], j["deps"], j["rule"], j["prio"])
        sys.exit(0)
    W = f"{VP.WORK}/{a.subject}"; tag = ("" if a.part == "all" else f"_{a.part}") + ("" if not a.only else "_" + re.sub(r"[^A-Za-z0-9]+", "-", a.only)[:40])
    open(f"{W}/QUEUE_STARTED{tag}", "w").write(time.strftime("%FT%TZ", time.gmtime()))
    s = Sched(a.subject, len(ids), plan, f"{W}/phasec_status{tag}.json", os.path.expanduser(f"{VP.LOGS}/jobs/{a.subject}{tag}"), gpu_ids=ids, gpu_mem=a.gpu_mem)
    s.run()
    open(f"{W}/QUEUE_DONE{tag}", "w").write(json.dumps({k: sum(j["state"] == k for j in s.jobs.values()) for k in ("done", "skipped", "failed")}))
    subprocess.run(["bash", os.path.join(CODE, "collect_out_e.sh"), a.subject])
