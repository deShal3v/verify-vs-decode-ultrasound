"""Load linkbudget/params.yaml (the single source of the link-budget constants) and expose them as plain Python values.
Every other link-budget script imports its constants from here."""
import os, math
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "params.yaml")) as _f:
    P = yaml.safe_load(_f)

_a, _t, _e, _l, _fn, _mc, _pv = (P["acoustics"], P["attenuation"], P["echo_table"], P["leakage"], P["functional"],
                                  P["monte_carlo"], P["per_voxel"])
C = _a["sound_speed_m_per_s"]["value"]                       # m/s
F_MHZ = _a["frequency_MHz"]["value"]                         # 2
APERTURE_MM = _a["aperture_mm"]["value"]                     # 20
AP20_SN_LOSS_DB = _a["aperture_20mm_correction"]["sn_loss_dB"]
AP20_CELLS_DIV = _a["aperture_20mm_correction"]["cells_divisor"]

BONE_COEF, BONE_EXP = _t["compact_bone"]["coef_dB_per_mm"], _t["compact_bone"]["exponent"]
BONE_SURF_DB = _t["bone_surfaces_dB"]["value"]
BRAIN_COEF, BRAIN_EXP = _t["brain"]["coef_dB_per_cm"], _t["brain"]["exponent"]
SCALP_DB_CM_MHZ = _t["scalp_dB_per_cm_per_MHz"]["value"]
SCALP_CM = _t["scalp_thickness_cm"]["value"]
BONE_REF_MM = _t["reference_bone_mm"]["value"]
ABERR_DB = _t["aberration_corrected_dB"]["value"]

TABLE = {(int(r[0]), int(r[1])): (r[2], r[3], r[4]) for r in _e["rows"]}
_w = _e["worked_example_inputs"]
AGENT = dict(delta=_w["delta"], lam=_w["lam"], K=_w["K"], T=_w["T_s"], v=_w["v_mm_per_s"])
S_TC_2MHZ = _l["clutter_2MHz_dB"]["value"]
S_REV_2MHZ = {int(k): v for k, v in _l["reverberation_2MHz_dB_by_depth_cm"]["value"].items()}

K_SCAN = _fn["K_images_per_scan"]["value"]                    # 2
T_IMAGE = _fn["T_image_s"]["value"]                          # 1.0 s
TR = _fn["TR_s"]["value"]                                    # 2.0 s
RHO = _fn["rho"]["value"]                                    # 0.26
V_LB = _fn["linkbudget_voxel_mm3"]["value"]                  # 27
V_FMRI = _fn["fmri_voxel_mm3"]["value"]                      # 17.83
DEF = _fn["function_defaults"]                               # delta, lam, K, v of the function signatures

MC_N = _mc["N"]["value"]; MC_SEED = _mc["seed"]["value"]
MC_RANGES = {"cal_db": ("uniform", _mc["cal_db"]["uniform"]), "delta": ("log_uniform", _mc["delta"]["log_uniform"]),
             "bone_mm": ("uniform", _mc["bone_mm"]["uniform"]), "leak_db": ("uniform", _mc["leak_db"]["uniform"]),
             "lam": ("log_uniform", _mc["lam"]["log_uniform"]), "v": ("log_uniform", _mc["v_mm_per_s"]["log_uniform"])}
MC_Z_CHOICE = _mc["z_cm_draw"]["choice"]

_g = P["depth"]["grid_cm"]; Z_GRID_SPEC = (_g["start"], _g["stop"], _g["step"])
FLOOR_DB = _pv["floor_dB"]["value"]
DMIN, DMAX = _pv["validity_mm"]["value"]
GAINS_DB = _pv["uniform_gain_dB"]["value"]
Z_REF_CM = P["reference_point"]["z_cm"]["value"]


def analytic_medians():
    """medians of the Monte Carlo input distributions: midpoint (uniform), geometric mean (log-uniform)"""
    out = {}
    for k, (kind, (lo, hi)) in MC_RANGES.items():
        out[k] = math.sqrt(lo * hi) if kind == "log_uniform" else float((lo + hi) / 2)
    return out
