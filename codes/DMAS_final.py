import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.patches import Rectangle

# ==========================================
# PARAMETERS
# ==========================================
fs          = 40e6        # Sampling frequency (Hz)
c           = 1500        # Speed of sound (m/s)
pitch       = 0.3e-3      # Sensor pitch (m)
num_sensors = 128
num_samples = 1024
dt          = 1 / fs

# ==========================================
# LOAD DATA
# ==========================================
data        = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data['sensor_data']   # shape: (128, 1024)

# ==========================================
# SENSOR POSITIONS (linear array, centered)
# ==========================================
x_sensor = (np.arange(num_sensors) - (num_sensors - 1) / 2) * pitch

# ==========================================
# BANDPASS FILTER (1–10 MHz)
# ==========================================
def bandpass(sig, low=1e6, high=10e6, fs=40e6, order=4):
    nyq  = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=1)

filtered = bandpass(sensor_data)

# ==========================================
# ENVELOPE (Hilbert)
# ==========================================
envelope = np.abs(hilbert(filtered, axis=1))   # (128, 1024)

# ==========================================
# RECONSTRUCTION GRID
# ==========================================
dx     = 0.1e-3
x_grid = np.arange(-19.05e-3, 19.05e-3, dx)   # lateral
z_grid = np.arange(0.5e-3, 20e-3, dx)          # depth
X, Z   = np.meshgrid(x_grid, z_grid)
Nz, Nx = Z.shape

# ==========================================
# DELAYED SIGNALS
# ==========================================
print("Computing delays...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx  = np.round(dist / (c * dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    idx[~mask] = 0
    tmp = np.zeros((Nz, Nx), dtype=np.float32)
    tmp[mask] = envelope[i, idx[mask]]
    delayed[i] = tmp

# ==========================================
# DMAS — Delay Multiply and Sum
# sign(si*sj) * sqrt(|si*sj|) summed over all pairs i<j
# ==========================================
print("Running DMAS (~1-2 min)...")
img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors - 1):
    for j in range(i + 1, num_sensors):
        prod      = delayed[i].astype(np.float64) * delayed[j].astype(np.float64)
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))
print("DMAS done!")

# ==========================================
# NORMALIZE
# ==========================================
img = np.abs(img_DMAS)
img /= np.max(img) + 1e-12

# ==========================================
# THRESHOLD — zero out below 50% of peak
# ==========================================
img_thresh = img.copy()
img_thresh[img_thresh < 0.5] = 0.0

# Log compress — 60 dB dynamic range
img_log = 20 * np.log10(img_thresh + 1e-6)
img_log = np.clip(img_log, -60, 0)

# ==========================================
# DOT POSITIONS & BACKGROUND ROIs
# (lateral_mm, depth_mm)
# ==========================================
dots = [
    (-10.0,  4.5),
    ( -5.0,  7.5),
    (  0.0, 10.0),
    (  5.0, 12.5),
    ( 10.0, 15.5),
]

# Background ROI per dot — placed away from all dots
backgrounds = [
    ( 15.0,  2.0),   # top-right  for dot 1
    ( 15.0,  2.0),   # top-right  for dot 2
    (-17.0,  2.0),   # top-left   for dot 3
    (-17.0,  2.0),   # top-left   for dot 4
    (-17.0,  2.0),   # top-left   for dot 5
]

hw_sig = 8    # signal ROI half-width (pixels)
hw_bg  = 12   # background ROI half-width (pixels)

def mm_to_xi(val, grid):
    return np.argmin(np.abs(grid - val * 1e-3))

# ==========================================
# PLOT — 5 subplots, one per dot
# ==========================================
fig, axes = plt.subplots(1, 5, figsize=(22, 7))

for k, (ax, dot, bg) in enumerate(zip(axes, dots, backgrounds)):
    lx,  dz  = dot
    blx, bdz = bg

    zi  = mm_to_xi(dz,  z_grid);  xi  = mm_to_xi(lx,  x_grid)
    bzi = mm_to_xi(bdz, z_grid);  bxi = mm_to_xi(blx, x_grid)

    # ── SBR ───────────────────────────────
    sig_roi = img_thresh[max(0, zi-hw_sig):zi+hw_sig,
                         max(0, xi-hw_sig):xi+hw_sig]
    bg_roi  = img_thresh[max(0, bzi-hw_bg):bzi+hw_bg,
                         max(0, bxi-hw_bg):bxi+hw_bg]
    signal  = np.max(sig_roi) if sig_roi.size > 0 else 1.0
    bg_nonzero = bg_roi[bg_roi > 0]  # ← Only non-zero background pixels
    noise   = np.std(bg_nonzero) if len(bg_nonzero) > 1 else 1e-6  # ← Real noise std
    sbr     = 20 * np.log10(signal / (noise + 1e-12))

    # ── Image ─────────────────────────────
    im = ax.imshow(img_log,
                   extent=[x_grid[0]*1e3, x_grid[-1]*1e3,
                            z_grid[-1]*1e3, z_grid[0]*1e3],
                   cmap='hot', aspect='auto', vmin=-60, vmax=0)

    # Signal ROI box — cyan
    ax.add_patch(Rectangle(
        (x_grid[max(0, xi-hw_sig)]*1e3, z_grid[max(0, zi-hw_sig)]*1e3),
        2*hw_sig*dx*1e3, 2*hw_sig*dx*1e3,
        edgecolor='cyan', facecolor='none', lw=1.5))

    # Background ROI box — green dashed
    ax.add_patch(Rectangle(
        (x_grid[max(0, bxi-hw_bg)]*1e3, z_grid[max(0, bzi-hw_bg)]*1e3),
        2*hw_bg*dx*1e3, 2*hw_bg*dx*1e3,
        edgecolor='lime', facecolor='none', lw=1.5, linestyle='--'))

    ax.set_title(f"Dot {k+1}  |  SBR: {sbr:.1f} dB", fontsize=10, fontweight='bold')
    ax.set_xlabel("Lateral (mm)", fontsize=9)
    if k == 0:
        ax.set_ylabel("Depth (mm)", fontsize=9)
    ax.tick_params(labelsize=8)

plt.colorbar(im, ax=axes[-1], label='Normalized Intensity (dB)')
plt.suptitle("DMAS Reconstruction — 5 Diagonal Targets  |  Threshold: 50% of peak\n"
             "Cyan = Signal ROI     Green dashed = Background ROI",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('DMAS_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: DMAS_final.png")
