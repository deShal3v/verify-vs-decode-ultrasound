"""Phase E2 (2026-10-19, Omer: one more rung in each of the gaps 48-64 and 64-82 stories, i.e. n56 and n73; Phase E kept untouched): Phase E plans on rungs
n56/n73 only (was n53/n59/n70/n76; changed before any job ran), every readout and condition of Phase E except usL50; each decode is also scored on the node against the
Phase D 4,000-null rule set (tang_bits_e.py, big_<cell> job), so no node is held. Original Phase E docstring follows.
Phase E node scheduler: Phase C's Sched (phasec_sched.py job runner, unchanged) with the Phase E plans (DESIGN_E.md).
Every fit uses tang_fit_e.py --preset $WORK_DIR/inputs/<S>/anat_cols.npy and every contrastive cell verify_e.py --preset (one anatomical
voxel set for the decoder, E-t-C and the contrastive verifier). Rungs n<k> = first k stories (k = 1 ... 82).
Roles: ver = text side + E-t-C features/pool + per VER cell (fit, E-t-C, C); dec = text side (for the bits step) + per DEC cell
(fit, frozen-times decode, 200-null story score, 200-null bits). --only REGEX keeps matching jobs + their dependencies.
Usage: python phasee_sched.py --subject S1 --role ver|dec [--only REGEX] [--gpu_ids 0] [--gpu_mem 22] [--dry]"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, subprocess, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phasec_sched as PC
from phasec_sched import J, once, PY, CODE, Sched

RUNGS = ["n56", "n73"]
VER_CELLS = [f"{c}_{r}" for c in ("native", "m6c", "m12c", "m18c", "usL90", "usB90") for r in RUNGS]
DEC_CELLS = [f"{c}_{r}" for c in ("native", "m6c", "m12c", "usL90", "usB90") for r in RUNGS]


def pre(S): return os.path.expanduser(f"{VP.INPUTS}/{S}/anat_cols.npy")


def fit_cmd(S, c):
    md = f"{VP.MODELS}/{S}"
    return f"{PY} tang_fit_e.py --subject {S} --cell {c} --out {md}/{c} --frozen {VP.WORK}/{S}/native_r15_decoded.npz --preset {pre(S)}"


def make_plan(S, role, only=None):
    md = f"{VP.MODELS}/{S}"; P = [J("text", "text", once("text", f"{PY} prep_text_words.py && ln -sfn {VP.WORK}/text {VP.WORK}/text_b"), prio=95)]
    if role == "ver":
        for i in range(4): P.append(J(f"feats{i}", "feats", once(f"feats{i}", f"{PY} etc_feats.py --shard {i} --nshards 4"), ["text"], prio=80))
        P.append(J("feats_merge", "merge", once("feats_merge", f"{PY} etc_feats_merge.py"), [f"feats{i}" for i in range(4)], prio=80, gpu=False))
        P.append(J("pool600", "pool", once("pool600", f"{PY} etc_pool600.py --subject {S} --cell native_r15 --build_pool_only"), ["text"], prio=78))
        for k, c in enumerate(VER_CELLS):
            P.append(J(f"fit_{c}", "fit_us" if c.startswith("us") else "fit", fit_cmd(S, c), prio=70))
            P.append(J(f"etc_{c}", "etc", f"{PY} etc_pool600.py --subject {S} --cell {c}", [f"fit_{c}", "feats_merge", "pool600"], prio=69))
            P.append(J(f"C_{c}", "C", f"{PY} verify_e.py --subject {S} --cells {c} --preset {pre(S)}", ["text"], prio=68))
    elif role == "dec":
        for c in DEC_CELLS:
            P.append(J(f"fit_{c}", "fit_us" if c.startswith("us") else "fit", fit_cmd(S, c), prio=100))
            P.append(J(f"decode_{c}", "decode", f"{PY} tang_decode_frozen.py --cell_dir {md}/{c}", [f"fit_{c}"], prio=90))
            P.append(J(f"score_{c}", "score", f"cp {VP.WORK}/{S}/nulls200.json {md}/{c}/nulls200.json && {PY} tang_eval2.py score --cell_dir {md}/{c} --n 200", [f"decode_{c}"], prio=92))
            P.append(J(f"bits_{c}", "bits", f"{PY} tang_bits.py {md}/{c} 200", [f"score_{c}", "text"], prio=92))
            P.append(J(f"big_{c}", "bits", f"mkdir -p {md}/{c}_big && cp -p {VP.INPUTS}/{S}/nulls_rule.json {VP.WORK}/{S}/nulls_rule.json && cp -p {md}/{c}/decoded.npz {md}/{c}_big/ && "
                       f"{PY} tang_bits_e.py {VP.WORK}/{S}/nulls_rule.json big {md}/{c}_big", [f"decode_{c}", "text"], prio=91, mem=8))
    else: raise KeyError(role)
    if only:
        rx = re.compile(only); keep = {j["id"] for j in P if rx.search(j["id"])}; byid = {j["id"]: j for j in P}; todo = list(keep)
        while todo:
            for d in byid[todo.pop()]["deps"]:
                if d not in keep: keep.add(d); todo.append(d)
        P = [j for j in P if j["id"] in keep]
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--role", required=True, choices=["ver", "dec"])
    ap.add_argument("--gpu_ids", default="0"); ap.add_argument("--gpu_mem", type=float, default=PC.GPU_MEM); ap.add_argument("--only", default=None); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); ids = [int(x) for x in a.gpu_ids.split(",")]
    plan = make_plan(a.subject, a.role, a.only)
    if a.dry:
        for j in plan: print(j["id"], j["kind"], j["deps"], j["prio"], "|", j["cmd"][-150:])
        sys.exit(0)
    W = f"{VP.WORK}/{a.subject}"; os.makedirs(W, exist_ok=True)
    tag = f"_E2{a.role}" + ("" if not a.only else "_" + re.sub(r"[^A-Za-z0-9]+", "-", a.only)[:40])
    open(f"{W}/QUEUE_STARTED{tag}", "w").write(time.strftime("%FT%TZ", time.gmtime()))
    s = Sched(a.subject, len(ids), plan, f"{W}/phasee_status{tag}.json", os.path.expanduser(f"{VP.LOGS}/jobs/{a.subject}{tag}"), gpu_ids=ids, gpu_mem=a.gpu_mem)
    s.run()
    open(f"{W}/QUEUE_DONE{tag}", "w").write(json.dumps({k: sum(j["state"] == k for j in s.jobs.values()) for k in ("done", "skipped", "failed")}))
    subprocess.run(["bash", os.path.join(CODE, "collect_out_e.sh"), a.subject])
