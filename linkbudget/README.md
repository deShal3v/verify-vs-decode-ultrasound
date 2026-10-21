# Link budget

`params.yaml` holds every parameter with its source; `params.py` loads it for all scripts.

| script | does | output ($WORK_DIR) |
|---|---|---|
| `snr_model.py` | the budget at the echo table's depths (2-5 cm); `python snr_model.py` checks it against the table's worked example | - |
| `snr_montecarlo.py` | the original 3-5 cm Monte Carlo (20,000 draws, seed 0) | `linkbudget/snr_montecarlo.json` |
| `depth_curves.py` | the budget at any depth (piecewise linear in dB per cm), p10/p50/p90 over the same draws, 10-s-trial form; checks against the two scripts above | `linkbudget/depth_curves.json` |
| `geom_lib.py`, `build_geometry.py` | the left and right temporal windows on each participant's scalp (25 mm behind the ear canal to the temporal pole, up to 5 cm above the zygomatic arch) and every voxel's distance to the nearest window point | `geometry/<S>_geometry.npz`, `<S>_sensor.json` |
| `k2_maps.py` | the paper's per-voxel conditions per 2-s scan (K = 2): target SNR, kept voxels and f_v (Eq. 2) | `inputs/<S>/us_maps.npz`, `linkbudget/k2_maps.json` |
| `fig1_linkbudget.py` | prints the Fig. 1A/B numbers, the Monte Carlo at 5 cm, the noise added to the readouts' voxels and the pessimistic reach, checks them against the paper, draws Fig. 1C | `linkbudget/fig1_numbers.json`, `fig1c.png/pdf` |

The echo SNR per Doppler sample is 1/s = 1/s_n + 1/s_c + 1/s_r. The thermal-noise term s_n comes from the echo table
(`params.yaml: echo_table`, -3 dB per cm at 2 MHz, for a 2.7-mm temporal bone and a 30-mm aperture); the attenuation
coefficients (compact bone 2.1 f^1.1 dB/mm plus 2.9 dB for its surfaces, brain 0.59 f^1.3 dB/cm, scalp 1 dB/cm/MHz) enter the
model when the bone is thicker or thinner than 2.7 mm (the Monte Carlo draws 2.0-3.8 mm) and split the 5-cm budget into the
components shown in Fig. 1A, where the echo with no loss (+38.8 dB at 5 cm) is what remains after the listed losses are
added back to the table's s_n. Validated outputs: `depth_curves.json` and the K = 2 keys of `us_maps.npz` are bit-identical
to the files the paper used.
