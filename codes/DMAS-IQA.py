"""
DIAGNOSTIC + FIX SCRIPT
Run this BEFORE the main IQA script to:
1. Check if DMAS output looks correct
2. Manually verify or override dot positions
3. Zoom each panel around its target for clear ROI verification
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import curve_fit

# ==========================================
# PARAMETERS
# ==========================================
fs = 40e6;  c = 1500;  pitch = 0.3e-3
num_sensors = 128;  num_samples = 1024;  dt = 1/fs
dx = 0.1e-3

# ==========================================
# LOAD
# ==========================================
print("Loading data...")
data = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data[list(data.keys())[-1]]
x_sensor = (np.arange(num_sensors) - (num_sensors - 1)/2) * pitch

# ==========================================
# BANDPASS + ENVELOPE
# ==========================================
def bandpass(sig, low=1e6, high=10e6, fs=40e6, order=4):
    nyq = fs/2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=1)

filtered = bandpass(sensor_data)
envelope = np.abs(hilbert(filtered, axis=1))

# ==========================================
# GRID
# ==========================================
x_grid = np.arange(-19.05e-3, 19.05e-3, dx)
z_grid = np.arange(0.5e-3, 20e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)
Nz, Nx = Z.shape

# ==========================================
# STEP 1: QUICK DAS (faster than DMAS for diagnostics)
# Use this to verify target positions first
# ==========================================
print("Running DAS for diagnostics...")
img_DAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx = np.round(dist/(c*dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    tmp = np.zeros((Nz, Nx))
    tmp[mask] = envelope[i, idx[mask]]
    img_DAS += tmp

img_DAS_norm = img_DAS / np.max(img_DAS)
img_DAS_log  = 20 * np.log10(np.clip(img_DAS_norm, 1e-6, 1))

# ==========================================
# STEP 2: DMAS
# ==========================================
print("Running DMAS...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx  = np.round(dist/(c*dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    tmp  = np.zeros((Nz, Nx), dtype=np.float32)
    tmp[mask] = envelope[i, idx[mask]]
    delayed[i] = tmp

img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors - 1):
    for j in range(i + 1, num_sensors):
        prod = delayed[i].astype(np.float64) * delayed[j].astype(np.float64)
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))

img_norm = np.abs(img_DMAS) / np.max(np.abs(img_DMAS))
img_log  = 20 * np.log10(np.clip(img_norm, 1e-6, 1))

# ==========================================
# STEP 3: IMPROVED TARGET DETECTION
#
# Problem with the original approach:
#   maximum_filter on the raw image picks bright STREAK artefacts.
#
# Fix:
#   1. Smooth image slightly before peak finding (suppresses streaks)
#   2. Apply a minimum depth spacing — real dots are separated by
#      several mm, so two peaks within 3 mm of each other = same dot
#   3. Print candidates so you can manually override if needed
# ==========================================
def find_point_targets(img_norm, x_grid, z_grid, n_targets=5,
                       smooth_sigma=2, min_sep_mm=3.0):
    """
    Find n_targets isolated bright spots, ignoring streak artefacts.
    smooth_sigma : Gaussian smoothing (pixels) before peak search
    min_sep_mm   : minimum distance between accepted peaks (mm)
    """
    # Smooth to suppress elongated streak artefacts
    img_smooth = gaussian_filter(img_norm, sigma=smooth_sigma)

    # Local maxima
    footprint = int(min_sep_mm / (dx * 1e3)) * 2 + 1   # pixels for min_sep
    peaks_mask = (maximum_filter(img_smooth, size=footprint) == img_smooth)
    peak_coords = np.argwhere(peaks_mask)

    # Sort by intensity descending
    intensities = img_smooth[peak_coords[:, 0], peak_coords[:, 1]]
    order = np.argsort(intensities)[::-1]
    peak_coords = peak_coords[order]

    # Greedy non-max suppression with minimum separation
    accepted = []
    min_sep_px = min_sep_mm / (dx * 1e3)
    for pz, px in peak_coords:
        too_close = False
        for az, ax_ in accepted:
            dist_px = np.sqrt((pz - az)**2 + (px - ax_)**2)
            if dist_px < min_sep_px:
                too_close = True
                break
        if not too_close:
            accepted.append((pz, px))
        if len(accepted) == n_targets:
            break

    dots = [(x_grid[px] * 1e3, z_grid[pz] * 1e3) for pz, px in accepted]
    return dots, accepted


dots_auto, peak_px = find_point_targets(img_norm, x_grid, z_grid,
                                         n_targets=5, smooth_sigma=2,
                                         min_sep_mm=3.0)

print("\nDetected targets (x mm, z mm):")
for k, (lx, dz) in enumerate(dots_auto):
    print(f"  Dot {k+1}: x={lx:.2f} mm,  z={dz:.2f} mm")

# ==========================================
# STEP 4: DIAGNOSTIC FIGURE
# Panel A — full DAS image with detected dots
# Panel B — full DMAS image with detected dots
# Panels C-G — zoomed DMAS patches (±3mm) around each dot with ROI boxes
# ==========================================
print("\nGenerating diagnostic figure...")

def mm_to_xi(val, grid):
    return int(np.argmin(np.abs(grid - val * 1e-3)))

zoom_mm  = 4.0   # half-window for zoom panels (mm)
hw_sig   = 8     # signal ROI half-width (pixels)
hw_bg    = 12    # background ROI half-width (pixels)

fig = plt.figure(figsize=(22, 10))
fig.suptitle("DMAS-IQA  |  Diagnostic View", fontsize=14, fontweight='bold')

# ---- Panel A: DAS full image ----
ax_das = fig.add_subplot(2, 7, 1)
ax_das.imshow(img_DAS_log,
              extent=[x_grid[0]*1e3, x_grid[-1]*1e3,
                      z_grid[-1]*1e3, z_grid[0]*1e3],
              cmap='hot', vmin=-60, vmax=0, aspect='auto')
for lx, dz in dots_auto:
    ax_das.plot(lx, dz, 'c+', ms=10, mew=2)
ax_das.set_title("DAS (full)", fontsize=9)
ax_das.set_xlabel("Lateral (mm)", fontsize=8)
ax_das.set_ylabel("Depth (mm)", fontsize=8)

# ---- Panel B: DMAS full image ----
ax_dmas = fig.add_subplot(2, 7, 2)
im_ref = ax_dmas.imshow(img_log,
                         extent=[x_grid[0]*1e3, x_grid[-1]*1e3,
                                 z_grid[-1]*1e3, z_grid[0]*1e3],
                         cmap='hot', vmin=-60, vmax=0, aspect='auto')
for lx, dz in dots_auto:
    ax_dmas.plot(lx, dz, 'c+', ms=10, mew=2)
ax_dmas.set_title("DMAS (full)", fontsize=9)
ax_dmas.set_xlabel("Lateral (mm)", fontsize=8)

plt.colorbar(im_ref, ax=ax_dmas, shrink=0.8, label='dB')

# ---- Panels C-G: zoomed patches around each dot ----
for k, (lx, dz) in enumerate(dots_auto):
    zi = mm_to_xi(dz, z_grid)
    xi = mm_to_xi(lx, x_grid)

    # Zoom window indices
    zoom_px = int(zoom_mm / (dx * 1e3))
    zi0 = max(0, zi - zoom_px);  zi1 = min(Nz, zi + zoom_px)
    xi0 = max(0, xi - zoom_px);  xi1 = min(Nx, xi + zoom_px)

    ax = fig.add_subplot(2, 7, k + 3)
    ax.imshow(img_log[zi0:zi1, xi0:xi1],
              extent=[x_grid[xi0]*1e3, x_grid[xi1-1]*1e3,
                      z_grid[zi1-1]*1e3, z_grid[zi0]*1e3],
              cmap='hot', vmin=-60, vmax=0, aspect='auto')

    # Signal ROI (cyan solid)
    sig_x0 = x_grid[max(0, xi - hw_sig)] * 1e3
    sig_z0 = z_grid[max(0, zi - hw_sig)] * 1e3
    sig_w  = 2 * hw_sig * dx * 1e3
    ax.add_patch(plt.Rectangle((sig_x0, sig_z0), sig_w, sig_w,
                                edgecolor='cyan', facecolor='none', lw=1.5))

    # Background ROI (lime dashed) — same depth, adjacent laterally
    bg_x0 = x_grid[min(Nx - 1, xi + hw_bg)] * 1e3
    bg_z0 = z_grid[max(0, zi - hw_bg)] * 1e3
    bg_w  = 2 * hw_bg * dx * 1e3
    ax.add_patch(plt.Rectangle((bg_x0, bg_z0), bg_w, bg_w,
                                edgecolor='lime', facecolor='none',
                                lw=1.5, ls='--'))

    ax.plot(lx, dz, 'w+', ms=8, mew=1.5)
    ax.set_title(f"Dot {k+1}  ({lx:.1f},{dz:.1f}) mm", fontsize=8, fontweight='bold')
    ax.set_xlabel("Lateral (mm)", fontsize=7)
    if k == 0:
        ax.set_ylabel("Depth (mm)", fontsize=7)
    ax.tick_params(labelsize=7)

# ---- Bottom row: lateral & axial profiles for each dot ----
for k, (lx, dz) in enumerate(dots_auto):
    zi = mm_to_xi(dz, z_grid)
    xi = mm_to_xi(lx, x_grid)

    profile_lat = img_log[max(0, zi-3):min(Nz, zi+3), :].max(axis=0)
    profile_ax  = img_log[:, max(0, xi-3):min(Nx, xi+3)].max(axis=1)

    ax = fig.add_subplot(2, 7, k + 3 + 7)
    ax.plot(x_grid * 1e3, profile_lat, 'c-',  lw=1.2, label='Lateral')
    ax_r = ax.twinx()
    ax_r.plot(z_grid * 1e3, profile_ax, 'm--', lw=1.2, label='Axial')
    ax.axvline(lx, color='w', ls=':', lw=0.8)
    ax.axhline(-6, color='gray', ls=':', lw=0.8)    # -6 dB line for FWHM check
    ax.set_xlim(lx - 5, lx + 5)
    ax.set_ylim(-60, 2)
    ax.set_xlabel("mm", fontsize=7)
    ax.set_ylabel("dB (lat)", fontsize=7, color='c')
    ax_r.set_ylabel("dB (ax)", fontsize=7, color='m')
    ax_r.set_ylim(-60, 2)
    ax.set_title(f"Profiles Dot {k+1}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax_r.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('DMAS_IQA_diagnostic.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: DMAS_IQA_diagnostic.png")

# ==========================================
# STEP 5: CHECKLIST — print what to look for
# ==========================================
print("\n" + "="*60)
print("CHECKLIST — verify these in the diagnostic figure:")
print("="*60)
print("""
1. DAS panel (Panel A):
   - Should show 5 bright point targets on a diagonal
   - Cyan '+' markers should sit ON the bright spots
   - If markers are on streaks → increase smooth_sigma (try 3 or 4)

2. DMAS panel (Panel B):
   - Same 5 dots, streaks should be more suppressed than DAS
   - Cyan '+' markers should still be on dot centres

3. Zoomed panels (Dots 1-5):
   - The bright spot should be INSIDE or touching the cyan box
   - The lime dashed box should be in a DARK/uniform region
     (not on another bright target or streak)
   - If lime box overlaps a streak, increase hw_bg or change
     background ROI side (left vs right)

4. Profile panels (bottom row):
   - Lateral profile (cyan) should show a clear narrow peak
   - -6 dB crossing points → FWHM width
   - If profile is flat/noisy → detection missed the real target

5. If any dot is wrong:
   - Manually set dots_auto in the main IQA script like this:
     dots_auto = [(-12.4,4.8), (-6.2,7.4), (-0.1,9.9),
                  (6.2,12.4), (12.5,14.9)]
""")