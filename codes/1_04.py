import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter

# ==========================================
# PARAMETERS (SAME)
# ==========================================
fs, c, pitch, num_sensors, num_samples, dt = 40e6, 1500, 0.3e-3, 128, 1024, 1/40e6

# ==========================================
# LOAD + PROCESS (SAME)
# ==========================================
data = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data[list(data.keys())[-1]]

x_sensor = (np.arange(num_sensors) - (num_sensors - 1) / 2) * pitch

def bandpass(sig, low=1e6, high=10e6, fs=40e6, order=4):
    from scipy.signal import butter, filtfilt
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=1)

filtered = bandpass(sensor_data)
envelope = np.abs(hilbert(filtered, axis=1))

# ==========================================
# RECONSTRUCTION GRID (SAME)
# ==========================================
dx = 0.1e-3
x_grid = np.arange(-19.05e-3, 19.05e-3, dx)
z_grid = np.arange(0.5e-3, 20e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)
Nz, Nx = Z.shape

# Backprojection (SAME)
print("Computing delays...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx = np.round(dist / (c * dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    tmp = np.zeros((Nz, Nx), dtype=np.float32)
    tmp[mask] = envelope[i, idx[mask]]
    delayed[i] = tmp

# DMAS (SAME)
print("Running DMAS...")
img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors - 1):
    for j in range(i + 1, num_sensors):
        prod = delayed[i] * delayed[j]
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))

# Normalize + threshold + log
img_norm = np.abs(img_DMAS) / np.max(img_DMAS)
img_thresh = img_norm.copy()
img_thresh[img_thresh < 0.166] = 0.0
img_log = 20 * np.log10(img_thresh + 1e-6)
img_log = np.clip(img_log, -60, 0)

# 🔥 AUTO-DETECT 5 BRIGHTEST TARGETS
print("🔍 Auto-detecting 5 brightest targets...")
peaks = np.argwhere(maximum_filter(img_norm, size=20) == img_norm)
peak_intensities = img_norm[peaks[:, 0], peaks[:, 1]]
top5_indices = np.argsort(peak_intensities)[-5:][::-1]  # Top 5 brightest

dots_auto = []
for idx in top5_indices:
    z_peak, x_peak = peaks[idx]
    dots_auto.append((x_grid[x_peak]*1e3, z_grid[z_peak]*1e3))  # Convert to mm

print("Detected targets (mm):", dots_auto)

# Background ROIs (offset from detected targets)
backgrounds_auto = []
for lx, dz in dots_auto:
    # Place background 10mm right OR 10mm left (avoid other targets)
    bg_x = lx + 10 if lx < 0 else lx - 10
    bg_z = min(dz + 2, 2.0)  # Near surface
    backgrounds_auto.append((bg_x, bg_z))

hw_sig, hw_bg = 8, 12

def mm_to_xi(val, grid):
    return np.argmin(np.abs(grid - val * 1e-3))

# ==========================================
# PLOT — Perfect ROI boxes on REAL targets
# ==========================================
fig, axes = plt.subplots(1, 5, figsize=(25, 12))
all_sbrs = []

for k, (ax, dot, bg) in enumerate(zip(axes, dots_auto, backgrounds_auto)):
    lx, dz = dot  # Auto-detected positions (mm)
    blx, bdz = bg
    
    zi = mm_to_xi(dz, z_grid); xi = mm_to_xi(lx, x_grid)
    bzi = mm_to_xi(bdz, z_grid); bxi = mm_to_xi(blx, x_grid)
    
    # SBR (using img_norm - linear scale)
    sig_roi = img_norm[max(0, zi-hw_sig):zi+hw_sig, max(0, xi-hw_sig):xi+hw_sig]
    bg_roi = img_norm[max(0, bzi-hw_bg):bzi+hw_bg, max(0, bxi-hw_bg):bxi+hw_bg]
    
    signal = np.max(sig_roi)
    bg_nonzero = bg_roi[bg_roi > 0]
    noise = np.std(bg_nonzero) if len(bg_nonzero) > 1 else 1e-6
    sbr = 20 * np.log10(signal / max(noise, 1e-12))
    all_sbrs.append(sbr)
    
    print(f"Dot {k+1}: ({lx:.1f},{dz:.1f}mm), SBR={sbr:.1f} dB")
    
    # Plot
    im = ax.imshow(img_log,
                   extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3],
                   cmap='hot', aspect='auto', vmin=-60, vmax=0)
    
    # PERFECT ROI BOXES (on actual peaks)
    ax.add_patch(Rectangle(
        (x_grid[max(0, xi-hw_sig)]*1e3, z_grid[max(0, zi-hw_sig)]*1e3),
        2*hw_sig*dx*1e3, 2*hw_sig*dx*1e3,
        edgecolor='cyan', facecolor='none', lw=2))
    
    ax.add_patch(Rectangle(
        (x_grid[max(0, bxi-hw_bg)]*1e3, z_grid[max(0, bzi-hw_bg)]*1e3),
        2*hw_bg*dx*1e3, 2*hw_bg*dx*1e3,
        edgecolor='lime', facecolor='none', lw=2, linestyle='--'))
    
    ax.set_title(f"Dot {k+1}\nSBR: {sbr:.1f} dB", fontsize=10, fontweight='bold')
    ax.set_xlabel("Lateral (mm)")
    if k == 0: ax.set_ylabel("Depth (mm)")

plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label='Intensity (dB)')
plt.suptitle("DMAS Reconstruction — AUTO-DETECTED Targets | Threshold: 50%\n"
             "🔍 Cyan boxes = Auto-detected peaks | Green = Background ROIs",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('DMAS_auto_targets.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Auto-detected SBRs: {np.round(all_sbrs, 1)} dB")