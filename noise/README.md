# Noise injection (Eq. 2)

For each voxel v the ten repeats of the test story give its fMRI SNR r_v (power of what repeats over power of what does not)
and its noise variance sigma_v^2 (`ncsnr_stream.py`, the same definitions as `pipeline/prep_resp.py`, plus a leave-one-repeat-out
jackknife standard error). Because ten repeats give noisy estimates, r_v is shrunk toward the distribution over all voxels
(`snr_maps.py: eb_hetero`): the prior is the nonparametric maximum-likelihood estimate on a grid (EM), and each estimate's
sampling noise is the median jackknife SE of voxels with similar r_v. Adding noise of variance f_v sigma_v^2 lowers the SNR to
r_v / (1 + f_v), so

    f_v = max(0, r_v / d~_v^2 - 1)                                                    (Eq. 2)

brings each kept voxel to the link budget's prediction d~_v (linkbudget/k2_maps.py); voxels already below it receive none.
On the GPU (`noise_us.py`) the noise sqrt(f_v clip(sigma_v^2, 0, 1)) eps is added on the full voxel grid, with eps drawn
from the fitted spatially correlated noise model of the fMRI (`noise_fit.py`; the fits used are in `fits/`), white in time and
independent for every story and repeat (seeded per story); each block is then re-z-scored and restricted to the condition's
voxels. The uniform ladder (x4, x16, x63 the noise power) adds the same correlated noise with f = 10^(dB/10) - 1 for every voxel
(dB = 6, 12, 18; f = 2.98, 14.8, 62.1).
