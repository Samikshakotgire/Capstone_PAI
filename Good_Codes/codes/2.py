import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat'
data = sio.loadmat(mat_file)
var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name]

if p_sensor.shape[0] != 128:
    p_sensor = p_sensor.T

print(f"Sensor data shape: {p_sensor.shape}")  # Should be (128, 1024)

# ── Parameters ────────────────────────────────────────────────
fs       = 40e6       # 40 MHz
c        = 1500       # m/s
pitch    = 0.3e-3     # 0.3 mm
num_elements = 128
num_samples  = 1024
dt = 1 / fs

# ── Sensor positions (linear array, centered) ─────────────────
x_sensor = (np.arange(num_elements) - (num_elements - 1) / 2) * pitch
# x_sensor spans roughly -19.05mm to +19.05mm

# ── Reconstruction grid — matched to ground truth ─────────────
# Ground truth: x-pos 0–18mm (depth), y-pos 0–45mm (lateral)
dx = 0.1e-3                                   # 0.1 mm resolution
x_grid = np.arange(0, 19e-3, dx)             # Lateral: matches GT x-axis
z_grid = np.arange(0.5e-3, 19e-3, dx)        # Depth:   matches GT y-axis
X, Z = np.meshgrid(x_grid, z_grid)

# ==========================================
# 2. PRE-PROCESSING
# ==========================================

# ── Bandpass filter (1–10 MHz) to remove low/high freq noise ──
def bandpass_filter(data, lowcut=1e6, highcut=10e6, fs=40e6, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

print("Bandpass filtering...")
p_filtered = bandpass_filter(p_sensor)

# ── Envelope via Hilbert (correct: axis=1 = time axis) ─────────
p_envelope = np.abs(hilbert(p_filtered, axis=1))  # shape: (128, 1024)

# ==========================================
# 3. VECTORIZED DELAY CALCULATION
# ==========================================
def get_delayed_data(p_data):
    """
    Returns 3D array [N_sensors x N_z x N_x] of delay-corrected signals.
    Uses envelope data for cleaner images.
    """
    Nz, Nx = Z.shape
    delayed = np.zeros((num_elements, Nz, Nx), dtype=np.float32)

    for i in range(num_elements):
        dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
        idx  = np.round(dist / (c * dt)).astype(int)
        mask = (idx >= 0) & (idx < num_samples)
        tmp  = np.zeros((Nz, Nx), dtype=np.float32)
        tmp[mask] = p_data[i, idx[mask]]
        delayed[i] = tmp

    return delayed

print("Computing delays (this may take ~30s)...")
delayed_env = get_delayed_data(p_envelope)   # envelope-based (cleaner)
delayed_raw = get_delayed_data(p_filtered)   # raw filtered (for DMAS/SLSC)

# ==========================================
# 4. RECONSTRUCTION ALGORITHMS
# ==========================================

# ── A. Universal Back Projection (DAS) ────────────────────────
print("UBP/DAS...")
img_UBP = np.sum(delayed_env, axis=0)

# ── B. DMAS (Delay Multiply and Sum) — correct formula ─────────
print("DMAS...")
img_DMAS = np.zeros(X.shape, dtype=np.float64)
for i in range(num_elements - 1):
    for j in range(i + 1, num_elements):
        prod = delayed_raw[i] * delayed_raw[j]
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))
# Note: This is O(N²) — for 128 sensors takes a few minutes.
# Faster approximation if needed: comment above and uncomment below:
# s = np.sum(delayed_raw, axis=0)
# img_DMAS = np.sign(s) * s**2 / (num_elements**2)

# ── C. Short-Lag Spatial Coherence (SLSC) ─────────────────────
print("SLSC...")
M = 10
# Normalized cross-correlation per lag
norm_factor = np.sqrt(
    np.sum(delayed_raw**2, axis=0) + 1e-12
)
img_SLSC = np.zeros(X.shape)
for m in range(1, M + 1):
    for i in range(num_elements - m):
        num   = delayed_raw[i] * delayed_raw[i + m]
        denom = (np.sqrt(np.sum(delayed_raw[i]**2) + 1e-12) *
                 np.sqrt(np.sum(delayed_raw[i + m]**2) + 1e-12) + 1e-12)
        img_SLSC += num / denom
img_SLSC /= M

# ── D. Coherence Factor weighted DAS (MV proxy) ───────────────
print("CF-MV...")
sum_sig  = np.sum(delayed_raw, axis=0)
sum_sq   = np.sum(delayed_raw**2, axis=0)
CF       = (sum_sig**2) / (num_elements * sum_sq + 1e-12)
CF       = np.clip(CF, 0, 1)
img_MV   = img_UBP * CF

# ==========================================
# 5. SBR CALCULATION  — correct ROI boxes
# ==========================================
# Ground truth dots are at approx (depth_mm, lateral_mm):
# (5,15), (7,20), (9,25), (12,30), (14,35) — diagonal pattern
# Map to pixel indices
def mm_to_idx(val_mm, grid_m):
    return np.argmin(np.abs(grid_m - val_mm * 1e-3))

# Dot approximate positions in mm [depth, lateral]
dot_positions = [(5,15), (7,20), (9,25), (12,30), (14,35)]

def calculate_sbr_v2(image, dot_positions, z_grid, x_grid, hw=10):
    """
    hw = half-width in pixels for signal ROI.
    Background taken from corners away from dots.
    """
    img_abs = np.abs(image)
    
    # Signal: max over all dot ROIs
    signal_vals = []
    for (dz, dx_) in dot_positions:
        zi = mm_to_idx(dz,  z_grid)
        xi = mm_to_idx(dx_, x_grid)
        roi = img_abs[max(0,zi-hw):zi+hw, max(0,xi-hw):xi+hw]
        if roi.size > 0:
            signal_vals.append(np.max(roi))
    signal = np.mean(signal_vals) if signal_vals else 1.0

    # Background: top-left corner (no dots there)
    bg  = img_abs[0:20, 0:20]
    sbr = 20 * np.log10(signal / (np.std(bg) + 1e-9))
    return sbr

# ==========================================
# 6. VISUALIZATION  — match ground truth style
# ==========================================
methods = [img_UBP, img_DMAS, img_SLSC, img_MV]
names   = ["UBP (DAS)", "DMAS", "SLSC", "CF-MV"]

fig, axes = plt.subplots(1, 4, figsize=(18, 7))

for ax, img, name in zip(axes, methods, names):
    img_disp = np.abs(img).astype(np.float64)
    img_disp /= (np.max(img_disp) + 1e-12)

    # Log compression (like dB display) for better dot visibility
    img_log = 20 * np.log10(img_disp + 1e-6)
    img_log = np.clip(img_log, -40, 0)   # 40 dB dynamic range

    sbr = calculate_sbr_v2(img, dot_positions, z_grid, x_grid)

    im = ax.imshow(img_log,
                   extent=[x_grid[0]*1e3, x_grid[-1]*1e3,
                            z_grid[-1]*1e3, z_grid[0]*1e3],
                   cmap='hot', aspect='auto', vmin=-40, vmax=0)

    ax.set_title(f"{name}\nSBR: {sbr:.1f} dB", fontsize=11)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")

    # Draw GT dot positions as cyan crosses
    for (dz, dx_) in dot_positions:
        ax.plot(dx_, dz, 'c+', markersize=10, markeredgewidth=1.5)

plt.colorbar(im, ax=axes[-1], label='Normalized Intensity (dB)')
plt.suptitle("PA Reconstruction — 5 Diagonal Targets", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('PA_reconstruction_fixed.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done! Saved as PA_reconstruction_fixed.png")