import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hilbert

# ==========================================
# 1. DATA & RECONSTRUCTION (Optimized)
# ==========================================
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat' 
data = sio.loadmat(mat_file)
var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name].T if data[var_name].shape[0] != 128 else data[var_name]

fs, c, total_len = 40e6, 1500, 38.4
dt = 1/fs
x_sensor = np.linspace(0, total_len, 128) * 1e-3
dx = 0.1e-3
x_grid = np.arange(0, 38.5e-3, dx)
z_grid = np.arange(0, 20.1e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)

# Baseline Reconstruction (UBP)
p_env = np.abs(hilbert(p_sensor, axis=1))
img = np.zeros(X.shape)
for i in range(128):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx = np.round(dist / (c * dt)).astype(int)
    mask = (idx >= 0) & (idx < 1024)
    img[mask] += p_env[i, idx[mask]]
img_norm = img / np.max(img)

# ==========================================
# 2. GRID DEFINITIONS (5x5)
# ==========================================
target_labels = ['A', 'B', 'C', 'D', 'E']
target_x = [6.7, 12.95, 19.2, 25.45, 31.7]  # Lateral positions
depths = [5.0, 7.5, 10.0, 12.5, 15.0]       # Perpendicular depths
bg_x = 2.0                                  # Constant background column

def get_stats(image, x, z, size=1.6):
    ix1, ix2 = int((x-size/2)/0.1), int((x+size/2)/0.1)
    iz1, iz2 = int((z-size/2)/0.1), int((z+size/2)/0.1)
    # Selection safety
    ix1, ix2 = max(0, ix1), min(image.shape[1], ix2)
    iz1, iz2 = max(0, iz1), min(image.shape[0], iz2)
    region = image[iz1:iz2, ix1:ix2]
    return np.mean(region)

# ==========================================
# 3. CREATIVE VISUALIZATION: THE 25-PLOT MATRIX
# ==========================================
fig, axes = plt.subplots(5, 5, figsize=(20, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for r in range(5):    # Depths (Rows)
    depth = depths[r]
    for col in range(5): # Targets (Columns)
        ax = axes[r, col]
        target_label = target_labels[col]
        tx = target_x[col]
        
        # Calculate Local SBR
        mean_sig = get_stats(img_norm, tx, depth)
        mean_bg = get_stats(img_norm, bg_x, depth)
        sbr = abs(round(20 * np.log10(mean_sig / (mean_bg + 1e-12)), 2))
        
        # Define Crop Area for the "Small Image"
        # We show a window that captures both the BG and the Target at that depth
        ax.imshow(img_norm, extent=[0, 38.4, 20, 0], cmap='hot', aspect='auto')
        ax.set_xlim(0, 35)      # Show lateral span from BG to Target
        ax.set_ylim(depth+3, depth-3) # Zoom in on the specific depth
        
        # Draw Bounding Boxes
        # Signal (Cyan)
        rect_s = patches.Rectangle((tx-0.8, depth-0.8), 1.6, 1.6, linewidth=1.2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect_s)
        # Background (Green)
        rect_b = patches.Rectangle((bg_x-0.8, depth-0.8), 1.6, 1.6, linewidth=1.2, edgecolor='#00FF00', facecolor='none', linestyle='--')
        ax.add_patch(rect_b)
        
        # Labels and Styling
        if r == 0: ax.set_title(f"Target {target_label}", fontweight='bold', fontsize=12)
        if col == 0: ax.set_ylabel(f"Depth {depth}mm", fontweight='bold', fontsize=12)
        
        ax.text(tx, depth+2.5, f"SBR: {sbr} dB", color='yellow', fontsize=9, ha='center', fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=7)

plt.suptitle("Photoacoustic SBR Matrix Analysis: 25-Point Comparison", fontsize=22, y=0.95, fontweight='bold')
plt.show()