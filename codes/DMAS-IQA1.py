import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

# ==========================================
# PARAMETERS
# ==========================================
fs = 40e6
c = 1500
pitch = 0.3e-3
num_sensors = 128
num_samples = 1024
dt = 1 / fs

# ==========================================
# LOAD + PROCESS
# ==========================================
print("Loading data...")
data = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data[list(data.keys())[-1]]
x_sensor = (np.arange(num_sensors) - (num_sensors - 1) / 2) * pitch


def bandpass(sig, low=1e6, high=10e6, fs=40e6, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig, axis=1)


print("Filtering + envelope...")
filtered = bandpass(sensor_data)
envelope = np.abs(hilbert(filtered, axis=1))

# ==========================================
# RECONSTRUCTION GRID
# ==========================================
dx = 0.1e-3
x_grid = np.arange(-19.05e-3, 19.05e-3, dx)
z_grid = np.arange(0.5e-3, 20e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)
Nz, Nx = Z.shape

# ==========================================
# BACKPROJECTION + DMAS
# ==========================================
print("Computing delays...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i]) ** 2 + Z ** 2)
    idx = np.round(dist / (c * dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    tmp = np.zeros((Nz, Nx), dtype=np.float32)
    tmp[mask] = envelope[i, idx[mask]]
    delayed[i] = tmp

print("Running DMAS beamforming...")
img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors - 1):
    for j in range(i + 1, num_sensors):
        prod = delayed[i].astype(np.float64) * delayed[j].astype(np.float64)
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))

# Normalize
img_norm = np.abs(img_DMAS) / np.max(np.abs(img_DMAS))   # BUG FIX 1: was dividing by max(img_DMAS) which could be negative
img_log = 20 * np.log10(np.clip(img_norm, 1e-6, 1))

# ==========================================
# AUTO-DETECT TARGETS
# ==========================================
print("Auto-detecting 5 brightest targets...")
peaks = np.argwhere(maximum_filter(img_norm, size=20) == img_norm)
peak_intensities = img_norm[peaks[:, 0], peaks[:, 1]]
top5_indices = np.argsort(peak_intensities)[-5:][::-1]
dots_auto = []
for idx in top5_indices:
    z_peak, x_peak = peaks[idx]
    dots_auto.append((x_grid[x_peak] * 1e3, z_grid[z_peak] * 1e3))

print("Detected targets (mm):", [f"({lx:.1f},{dz:.1f})" for lx, dz in dots_auto])


# ==========================================
# IMAGE QUALITY METRICS
# ==========================================
def mm_to_xi(val, grid):
    return np.argmin(np.abs(grid - val * 1e-3))


def fwhm_gaussian(profile):
    def gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    try:
        popt, _ = curve_fit(gauss, np.arange(len(profile)), profile,
                            p0=[np.max(profile), np.argmax(profile), 5], maxfev=5000)
        fwhm = 2.355 * abs(popt[2])   # BUG FIX 2: abs() on sigma — Gaussian sigma must be positive;
                                       # curve_fit can return negative sigma, causing negative FWHM
        return min(fwhm, len(profile))
    except:
        return np.nan


def compute_all_metrics(img_norm, dots_auto, z_grid, x_grid, hw_sig=8, hw_bg=12):
    metrics = []
    for i, (lx, dz) in enumerate(dots_auto):
        zi, xi = mm_to_xi(dz, z_grid), mm_to_xi(lx, x_grid)

        # Signal ROI (centred on target)
        sig_roi = img_norm[max(0, zi - hw_sig):min(Nz, zi + hw_sig),
                           max(0, xi - hw_sig):min(Nx, xi + hw_sig)]

        # Background ROI (adjacent, to the right)
        bg_roi = img_norm[max(0, zi - hw_bg):min(Nz, zi + hw_bg),
                          max(0, xi + hw_bg):min(Nx, xi + 2 * hw_bg)]

        # 1. SBR  (signal max / background noise std) — per paper: S / sigma_b
        signal_max = np.max(sig_roi)
        noise_bg = np.std(bg_roi) if bg_roi.size > 0 else 1e-6   # BUG FIX 3: removed (bg_roi > 0) mask;
                                                                   # paper defines sigma_b over ALL background
                                                                   # pixels, not just positive ones
        noise_bg = max(noise_bg, 1e-6)
        sbr = 20 * np.log10(signal_max / noise_bg)

        # 2. SNR  (signal mean / background noise std) — per paper: S / sigma_b
        signal_mean = np.mean(sig_roi)
        snr = 20 * np.log10(signal_mean / noise_bg)

        # 3. CNR  — per paper: |mu_s - mu_b| / sigma_b
        bg_mean = np.mean(bg_roi) if bg_roi.size > 0 else 0.0
        cnr = abs(signal_mean - bg_mean) / noise_bg   # BUG FIX 4: paper formula uses sigma_b in denominator,
                                                       # NOT sqrt(var_s + var_b); the original code mixed up
                                                       # the CNR formula with a pooled-variance variant

        # 4. gCNR — per paper: 1 - sum_i min(p_s(i), p_b(i))
        #    Use histogram over joint pixel range of paired ROIs, not KS statistic
        if sig_roi.size > 0 and bg_roi.size > 0:
            joint_min = min(sig_roi.min(), bg_roi.min())
            joint_max = max(sig_roi.max(), bg_roi.max())
            bins = np.linspace(joint_min, joint_max, 51)   # BUG FIX 5: gCNR must be computed over joint range
            hist_s, _ = np.histogram(sig_roi.flatten(), bins=bins, density=True)
            hist_b, _ = np.histogram(bg_roi.flatten(), bins=bins, density=True)
            bin_width = bins[1] - bins[0]
            # Normalise to probability mass (density * bin_width)
            ps = hist_s * bin_width
            pb = hist_b * bin_width
            gcnr = 1.0 - float(np.sum(np.minimum(ps, pb)))   # BUG FIX 6: original code used KS statistic
                                                              # (ks_2samp) which is NOT gCNR; paper formula is
                                                              # 1 - sum_i min(p_s(i), p_b(i))
            gcnr = float(np.clip(gcnr, 0.0, 1.0))
        else:
            gcnr = 0.0

        # 5–6. FWHM (lateral and axial) — fitted Gaussian on max-projection profiles
        profile_lat = np.max(img_norm[max(0, zi - 3):min(Nz, zi + 3), :], axis=0)
        profile_ax = np.max(img_norm[:, max(0, xi - 3):min(Nx, xi + 3)], axis=1)
        fwhm_lat = fwhm_gaussian(profile_lat)
        fwhm_ax = fwhm_gaussian(profile_ax)

        # 7. Contrast (foreground/background mean ratio)
        fg_mask = img_norm > np.percentile(img_norm, 90)
        bg_mask = img_norm < np.percentile(img_norm, 10)
        contrast = np.mean(img_norm[fg_mask]) / (np.mean(img_norm[bg_mask]) + 1e-6)

        # 8. Speckle SNR (mean/std within local ROI around target)
        r = 20
        speckle_roi = img_norm[max(0, zi - r):min(Nz, zi + r),
                               max(0, xi - r):min(Nx, xi + r)]
        speckle_std = np.std(speckle_roi)
        speckle_snr = np.mean(speckle_roi) / speckle_std if speckle_std > 0 else 0.0

        # 9. Edge Sharpness (mean gradient magnitude around target)
        grad_x = np.gradient(img_norm, axis=1)
        grad_y = np.gradient(img_norm, axis=0)
        r_e = 10
        edge_sharp = np.mean(
            np.sqrt(grad_x ** 2 + grad_y ** 2)[
                max(0, zi - r_e):min(Nz, zi + r_e),
                max(0, xi - r_e):min(Nx, xi + r_e)
            ]
        )

        metrics.append({
            'Dot': i + 1,
            'Pos': f'({lx:.1f},{dz:.1f})',
            'SBR': f'{sbr:.1f}',
            'SNR': f'{snr:.1f}',
            'CNR': f'{cnr:.2f}',
            'gCNR': f'{gcnr:.3f}',
            'FWHM-Lat': f'{fwhm_lat * dx * 1e6:.0f}',
            'FWHM-Ax': f'{fwhm_ax * dx * 1e6:.0f}',
            'Contrast': f'{contrast:.2f}',
            'Speckle-SNR': f'{speckle_snr:.2f}',
            'Edge': f'{edge_sharp:.3f}'
        })
    return metrics


# ==========================================
# COMPUTE METRICS
# ==========================================
print("\nComputing 9 image quality metrics...")
metrics_table = compute_all_metrics(img_norm, dots_auto, z_grid, x_grid)

# ==========================================
# PUBLICATION-READY TABLE
# ==========================================
print("\n" + "=" * 120)
print("COMPLETE IMAGE QUALITY ASSESSMENT | DMAS Beamforming | 5 Auto-Detected Targets")
print("=" * 120)
print(f"{'Dot':<4} {'Pos(mm)':<12} {'SBR(dB)':<9} {'SNR(dB)':<9} {'CNR':<7} {'gCNR':<7} "
      f"{'FWHM-L(um)':<11} {'FWHM-A(um)':<11} {'Contrast':<10} {'Spk-SNR':<9} {'Edge':<8}")
print("-" * 120)

for m in metrics_table:
    print(f"{m['Dot']:<4} {m['Pos']:<12} {m['SBR']:<9} {m['SNR']:<9} {m['CNR']:<7} {m['gCNR']:<7} "
          f"{m['FWHM-Lat']:<11} {m['FWHM-Ax']:<11} {m['Contrast']:<10} {m['Speckle-SNR']:<9} {m['Edge']:<8}")

# Averages — BUG FIX 7: original average loop was broken (incorrect key extraction via split logic)
keys_numeric = ['SBR', 'SNR', 'CNR', 'gCNR', 'FWHM-Lat', 'FWHM-Ax', 'Contrast', 'Speckle-SNR', 'Edge']
avgs = {}
for k in keys_numeric:
    try:
        avgs[k] = np.mean([float(m[k]) for m in metrics_table])
    except (ValueError, KeyError):
        avgs[k] = float('nan')

print("\nAVERAGES: " + " | ".join([f"{k}: {avgs[k]:.2f}" for k in avgs]))

# ==========================================
# VISUALIZATION (5 panels)
# ==========================================
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("DMAS Reconstruction + 9 Image Quality Metrics\nAuto-Detected Targets",
             fontsize=16, fontweight='bold')

for k, (ax, dot, metric) in enumerate(zip(axes, dots_auto, metrics_table)):
    lx, dz = dot
    zi, xi = mm_to_xi(dz, z_grid), mm_to_xi(lx, x_grid)

    im = ax.imshow(img_log,
                   extent=[x_grid[0] * 1e3, x_grid[-1] * 1e3,
                           z_grid[-1] * 1e3, z_grid[0] * 1e3],
                   cmap='hot', vmin=-60, vmax=0, aspect='auto')

    # Signal ROI box (cyan)
    hw_sig = 8
    ax.add_patch(Rectangle(
        (x_grid[max(0, xi - hw_sig)] * 1e3, z_grid[max(0, zi - hw_sig)] * 1e3),
        2 * hw_sig * dx * 1e3, 2 * hw_sig * dx * 1e3,   # BUG FIX 8: box width/height now derived from
        edgecolor='cyan', facecolor='none', lw=2))        # actual grid spacing; original used hardcoded
                                                           # 1.6e-3/2.4e-3 which were wrong for this grid
    # Background ROI box (lime)
    hw_bg = 12
    ax.add_patch(Rectangle(
        (x_grid[max(0, xi + hw_bg)] * 1e3, z_grid[max(0, zi - hw_bg)] * 1e3),
        2 * hw_bg * dx * 1e3, 2 * hw_bg * dx * 1e3,
        edgecolor='lime', facecolor='none', lw=2, ls='--'))

    title = (f"Dot {k + 1}  {metric['Pos']} mm\n"
             f"SBR: {metric['SBR']} dB | SNR: {metric['SNR']} dB\n"
             f"CNR: {metric['CNR']} | gCNR: {metric['gCNR']}")
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel("Lateral (mm)")
    if k == 0:
        ax.set_ylabel("Depth (mm)")

plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label='dB')
plt.tight_layout()
plt.savefig('DMAS_9_metrics_complete.png', dpi=200, bbox_inches='tight')
plt.show()

print("\nDone. Saved: DMAS_9_metrics_complete.png")
print(f"9 metrics x {len(dots_auto)} targets = {9 * len(dots_auto)} quality values computed.")