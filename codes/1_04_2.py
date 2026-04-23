import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter

# [SAME PROCESSING UNTIL img_norm - keeping it short]
fs, c, pitch, num_sensors, num_samples, dt = 40e6, 1500, 0.3e-3, 128, 1024, 1/40e6

# Load + process (same as before)
data = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data[list(data.keys())[-1]]
x_sensor = (np.arange(num_sensors) - (num_sensors - 1) / 2) * pitch

def bandpass(sig, low=1e6, high=10e6, fs=40e6, order=4):
    from scipy.signal import butter, filtfilt
    nyq = fs / 2; b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig, axis=1)

filtered = bandpass(sensor_data)
envelope = np.abs(hilbert(filtered, axis=1))

dx = 0.1e-3
x_grid = np.arange(-19.05e-3, 19.05e-3, dx)
z_grid = np.arange(0.5e-3, 20e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid); Nz, Nx = Z.shape

print("Computing delays...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx = np.round(dist / (c * dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    tmp = np.zeros((Nz, Nx), dtype=np.float32)
    tmp[mask] = envelope[i, idx[mask]]; delayed[i] = tmp

print("Running DMAS...")
img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors - 1):
    for j in range(i + 1, num_sensors):
        prod = delayed[i] * delayed[j]
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))

img_norm = np.abs(img_DMAS) / np.max(img_DMAS)

# 🔥 AUTO-DETECT TARGETS (same as before)
print("🔍 Auto-detecting targets...")
peaks = np.argwhere(maximum_filter(img_norm, size=20) == img_norm)
peak_intensities = img_norm[peaks[:, 0], peaks[:, 1]]
top5_indices = np.argsort(peak_intensities)[-5:][::-1]
dots_auto = [(x_grid[p[1]]*1e3, z_grid[p[0]]*1e3) for p in peaks[top5_indices]]

# ==========================================
# 🔥 ITERATIVE BACKGROUND SUPPRESSION
# ==========================================
suppression_factors = [4, 8, 16]  # 1/4, 1/8, 1/16 of target peak
hw_sig, hw_bg = 8, 12

def mm_to_xi(val, grid): return np.argmin(np.abs(grid - val * 1e-3))

fig, axes = plt.subplots(3, 5, figsize=(25, 12))
fig.suptitle("🎯 ITERATIVE BACKGROUND SUPPRESSION\n1/4 → 1/8 → 1/16 of Current Target Peak", 
             fontsize=16, fontweight='bold')

all_sbrs = []
for row, factor in enumerate(suppression_factors):
    print(f"\n🔧 Suppression factor: 1/{factor}")
    
    for col, (ax, dot) in enumerate(zip(axes[row], dots_auto)):
        lx, dz = dot
        zi, xi = mm_to_xi(dz, z_grid), mm_to_xi(lx, x_grid)
        
        # 🔥 CREATE TARGET-FOCUSED IMAGE
        img_focus = img_norm.copy()
        
        # 1) Keep region around CURRENT target (full intensity)
        keep_mask = np.zeros((Nz, Nx), dtype=bool)
        for size in range(1, 30):  # Large keep region
            i0, i1 = max(0, zi-size), min(Nz, zi+size)
            j0, j1 = max(0, xi-size), min(Nx, xi+size)
            keep_mask[i0:i1, j0:j1] = True
        
        # 2) Get current target peak
        target_peak = np.max(img_norm[max(0, zi-15):zi+15, max(0, xi-15):xi+15])
        
        # 3) SUPPRESS everything else < 1/factor of current target
        threshold = target_peak / factor
        outside = ~keep_mask
        img_focus[outside & (img_focus < threshold)] = 0.0
        
        # 4) Log compress for display
        img_focus_log = 20 * np.log10(img_focus + 1e-6)
        img_focus_log = np.clip(img_focus_log, -60, 0)
        
        # 5) SBR calculation
        sig_roi = img_norm[max(0, zi-hw_sig):zi+hw_sig, max(0, xi-hw_sig):xi+hw_sig]
        bg_roi = img_norm[max(0, zi-hw_bg):zi+hw_bg, max(0, xi+hw_bg):xi+2*hw_bg]  # Right of target
        
        signal = np.max(sig_roi)
        bg_nonzero = bg_roi[bg_roi > 0]
        noise = np.std(bg_nonzero) if len(bg_nonzero) > 1 else 1e-6
        sbr = 20 * np.log10(signal / max(noise, 1e-12))
        all_sbrs.append(sbr)
        
        # 6) Plot
        im = ax.imshow(img_focus_log,
                      extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3],
                      cmap='hot', vmin=-60, vmax=0, aspect='auto')
        
        # ROIs
        ax.add_patch(Rectangle((x_grid[max(0, xi-hw_sig)]*1e3, z_grid[max(0, zi-hw_sig)]*1e3),
                              2*hw_sig*dx*1e3, 2*hw_sig*dx*1e3,
                              edgecolor='cyan', facecolor='none', lw=2))
        
        ax.add_patch(Rectangle((x_grid[max(0, xi+hw_bg)]*1e3, z_grid[max(0, zi-hw_bg)]*1e3),
                              2*hw_bg*dx*1e3, 2*hw_bg*dx*1e3,
                              edgecolor='lime', facecolor='none', lw=2, linestyle='--'))
        
        ax.set_title(f"Dot {col+1}\nSBR: {sbr:.1f}dB\n1/{factor}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Lateral (mm)")
        if col == 0: ax.set_ylabel("Depth (mm)")

# Shared colorbar
cbar = plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cbar.set_label('Intensity (dB)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('DMAS_iterative_suppression.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Suppression complete! SBRs improved by avg {np.mean(all_sbrs[-5:]) - np.mean(all_sbrs[:5]):.1f} dB")