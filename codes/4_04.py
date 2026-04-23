import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

# ==========================================
# PARAMETERS
# ==========================================
fs = 40e6; c = 1500; pitch = 0.3e-3; num_sensors = 128; num_samples = 1024; dt = 1/fs

# ==========================================
# LOAD + PROCESS
# ==========================================
print("Loading data...")
data = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data[list(data.keys())[-1]]   
x_sensor = (np.arange(num_sensors) - (num_sensors - 1)/2) * pitch

def bandpass(sig, low=1e6, high=10e6, fs=40e6, order=4):
    nyq = fs/2; b, a = butter(order, [low/nyq, high/nyq], btype='band')
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
X, Z = np.meshgrid(x_grid, z_grid); Nz, Nx = Z.shape

# ==========================================
# BACKPROJECTION + DMAS
# ==========================================
print("Computing delays...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx = np.round(dist/(c*dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    tmp = np.zeros((Nz, Nx), dtype=np.float32)
    tmp[mask] = envelope[i, idx[mask]]; delayed[i] = tmp

print("Running DMAS beamforming...")
img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)
for i in range(num_sensors-1):
    for j in range(i+1, num_sensors):
        prod = delayed[i].astype(np.float64) * delayed[j].astype(np.float64)
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))

# Normalize
img_norm = np.abs(img_DMAS) / np.max(img_DMAS)
img_log = 20 * np.log10(np.clip(img_norm, 1e-6, 1))

# ==========================================
# 🔥 AUTO-DETECT TARGETS
# ==========================================
print("🔍 Auto-detecting 5 brightest targets...")
peaks = np.argwhere(maximum_filter(img_norm, size=20) == img_norm)
peak_intensities = img_norm[peaks[:,0], peaks[:,1]]
top5_indices = np.argsort(peak_intensities)[-5:][::-1]
dots_auto = []
for idx in top5_indices:
    z_peak, x_peak = peaks[idx]
    dots_auto.append((x_grid[x_peak]*1e3, z_grid[z_peak]*1e3))

print("Detected targets (mm):", [f"({lx:.1f},{dz:.1f})" for lx,dz in dots_auto])

# ==========================================
# 🔥 9 IMAGE QUALITY METRICS
# ==========================================
def mm_to_xi(val, grid): return np.argmin(np.abs(grid - val*1e-3))

def fwhm_gaussian(profile):
    def gauss(x, a, x0, sigma): return a * np.exp(-(x-x0)**2/(2*sigma**2))
    try:
        popt, _ = curve_fit(gauss, np.arange(len(profile)), profile, 
                          p0=[np.max(profile), np.argmax(profile), 5], maxfev=5000)
        fwhm = 2.355 * popt[2]
        return min(fwhm, len(profile))
    except: return np.nan

def compute_all_metrics(img_norm, dots_auto, z_grid, x_grid, hw_sig=8, hw_bg=12):
    metrics = []
    for i, (lx, dz) in enumerate(dots_auto):
        zi, xi = mm_to_xi(dz, z_grid), mm_to_xi(lx, x_grid)
        
        # ROIs
        sig_roi = img_norm[max(0,zi-hw_sig):min(Nz,zi+hw_sig), 
                          max(0,xi-hw_sig):min(Nx,xi+hw_sig)]
        bg_roi = img_norm[max(0,zi-hw_bg):min(Nz,zi+hw_bg), 
                         max(0,xi+hw_bg):min(Nx,xi+2*hw_bg)]
        
        # 1. SBR
        signal_max = np.max(sig_roi)
        noise_bg = np.std(bg_roi[bg_roi > 0]) if np.any(bg_roi > 0) else 1e-6
        sbr = 20 * np.log10(signal_max / noise_bg)
        
        # 2. SNR
        signal_mean = np.mean(sig_roi)
        snr = 20 * np.log10(signal_mean / noise_bg)
        
        # 3. CNR
        bg_mean = np.mean(bg_roi)
        cnr = abs(signal_mean - bg_mean) / np.sqrt(np.var(sig_roi) + np.var(bg_roi))
        
        # 4. gCNR
        hist_s, _ = np.histogram(sig_roi.flatten(), bins=50, density=True)
        hist_b, _ = np.histogram(bg_roi.flatten(), bins=50, density=True)
        kl_s2b = ks_2samp(hist_s[hist_s>0], hist_b[hist_b>0])[0] if np.any(hist_s>0) and np.any(hist_b>0) else 0
        gcnr = 1 - min(kl_s2b, 0.5)
        
        # 5-6. FWHM
        profile_lat = np.max(img_norm[max(0,zi-3):min(Nz,zi+3), :], axis=0)
        profile_ax = np.max(img_norm[:, max(0,xi-3):min(Nx,xi+3)], axis=1)
        fwhm_lat = fwhm_gaussian(profile_lat)
        fwhm_ax = fwhm_gaussian(profile_ax)
        
        # 7. Contrast
        fg_mask = img_norm > np.percentile(img_norm, 90)
        bg_mask = img_norm < np.percentile(img_norm, 10)
        contrast = np.mean(img_norm[fg_mask]) / (np.mean(img_norm[bg_mask]) + 1e-6)
        
        # 8. Speckle SNR
        speckle_roi = img_norm[zi-20:zi+20, xi-20:xi+20]
        speckle_snr = np.mean(speckle_roi) / np.std(speckle_roi)
        
        # 9. Edge Sharpness
        grad_x = np.gradient(img_norm, axis=1)
        grad_y = np.gradient(img_norm, axis=0)
        edge_sharp = np.mean(np.sqrt(grad_x**2 + grad_y**2)[zi-10:zi+10, xi-10:xi+10])
        
        metrics.append({
            'Dot': i+1, 'Pos': f'({lx:.1f},{dz:.1f})',
            'SBR': f'{sbr:.1f}', 'SNR': f'{snr:.1f}', 'CNR': f'{cnr:.2f}',
            'gCNR': f'{gcnr:.3f}', 'FWHM-Lat': f'{fwhm_lat*dx*1e6:.0f}',
            'FWHM-Ax': f'{fwhm_ax*dx*1e6:.0f}', 'Contrast': f'{contrast:.2f}',
            'Speckle-SNR': f'{speckle_snr:.2f}', 'Edge': f'{edge_sharp:.3f}'
        })
    return metrics

# ==========================================
# COMPUTE METRICS
# ==========================================
print("\n🔬 Computing 9 image quality metrics...")
metrics_table = compute_all_metrics(img_norm, dots_auto, z_grid, x_grid)

# ==========================================
# PUBLICATION-READY TABLE
# ==========================================
print("\n" + "="*120)
print("📊 COMPLETE IMAGE QUALITY ASSESSMENT | DMAS Beamforming | 5 Auto-Detected Targets")
print("="*120)
print(f"{'Dot':<4} {'Pos(mm)':<10} {'SBR':<6} {'SNR':<6} {'CNR':<6} {'gCNR':<7} {'FWHM-L(μm)':<10} {'FWHM-A(μm)':<10} {'Contrast':<9} {'Spk-SNR':<8} {'Edge':<8}")
print("-"*120)

for m in metrics_table:
    print(f"{m['Dot']:<4} {m['Pos']:<10} {m['SBR']:<6} {m['SNR']:<6} {m['CNR']:<6} {m['gCNR']:<7} "
          f"{m['FWHM-Lat']:<10} {m['FWHM-Ax']:<10} {m['Contrast']:<9} {m['Speckle-SNR']:<8} {m['Edge']:<8}")

# Averages
avgs = {k: np.mean([float(m[k.split('-')[0]].replace('μm','')) for m in metrics_table]) 
        for k in ['SBR','SNR','CNR','gCNR','FWHM-Lat','FWHM-Ax','Contrast','Speckle-SNR','Edge'] 
        if k.split('-')[0] in [list(m.keys())[2] for m in metrics_table]}
print("\n✅ AVERAGES: " + " | ".join([f"{k}: {avgs[k]:.2f}" for k in avgs.keys()]))

# ==========================================
# VISUALIZATION (5 panels + metrics)
# ==========================================
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("DMAS Reconstruction + 9 Image Quality Metrics\n🔬 Auto-Detected Targets", fontsize=16, fontweight='bold')

for k, (ax, dot, metric) in enumerate(zip(axes, dots_auto, metrics_table)):
    lx, dz = dot
    zi, xi = mm_to_xi(dz, z_grid), mm_to_xi(lx, x_grid)
    
    im = ax.imshow(img_log, extent=[x_grid[0]*1e3, x_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3],
                   cmap='hot', vmin=-60, vmax=0, aspect='auto')
    
    # ROIs
    ax.add_patch(Rectangle((x_grid[max(0,xi-8)]*1e3, z_grid[max(0,zi-8)]*1e3),
                          1.6e-3, 1.6e-3, edgecolor='cyan', facecolor='none', lw=2))
    ax.add_patch(Rectangle((x_grid[max(0,xi+12)]*1e3, z_grid[max(0,zi-12)]*1e3),
                          2.4e-3, 2.4e-3, edgecolor='lime', facecolor='none', lw=2, ls='--'))
    
    # Metrics in title
    title = f"Dot {k+1}\nSBR: {metric['SBR']}dB\nCNR: {metric['CNR']}\ngCNR: {metric['gCNR']}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Lateral (mm)")
    if k == 0: ax.set_ylabel("Depth (mm)")

plt.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label='dB')
plt.tight_layout()
plt.savefig('DMAS_9_metrics_complete.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n🎉 COMPLETE! Saved: DMAS_9_metrics_complete.png")
print("✅ 9 metrics × 5 targets = 45 quality values computed!")
print("📄 Publication-ready table + visualization ready!")