import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hilbert
import pandas as pd

# ==========================================
# 1. SETUP & ALIGNMENT
# ==========================================
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat' 
data = sio.loadmat(mat_file)
var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name].T if data[var_name].shape[0] != 128 else data[var_name]

fs, c, total_len = 40e6, 1500, 38.4
num_elements, dt = 128, 1/fs
x_sensor = np.linspace(0, total_len, num_elements) * 1e-3

# Positive Axes Grid (0.1mm resolution)
dx = 0.1e-3
x_grid = np.arange(0, 38.5e-3, dx)
z_grid = np.arange(0, 20.1e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)

# ==========================================
# 2. DEFINE ROI POSITIONS (5 Signal, 5 Background)
# ==========================================
# Depths (Perpendicular positions)
depths = [5.0, 7.5, 10.0, 12.5, 15.0]

# Lateral positions for targets A, B, C, D, E
target_x = [6.7, 12.95, 19.2, 25.45, 31.7]
target_labels = ['A', 'B', 'C', 'D', 'E']

# Background column (in a clear area, e.g., 2mm from the edge)
bg_x = 2.0 

# ==========================================
# 3. RECONSTRUCTION ENGINE
# ==========================================
p_env = np.abs(hilbert(p_sensor, axis=1))

def get_reconstruction(data_in):
    img = np.zeros(X.shape)
    for i in range(num_elements):
        dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
        idx = np.round(dist / (c * dt)).astype(int)
        mask = (idx >= 0) & (idx < 1024)
        img[mask] += data_in[i, idx[mask]]
    return img

print("Reconstructing Image...")
img_final = get_reconstruction(p_env)
img_norm = img_final / np.max(img_final)

def get_mean_intensity(img, x_mm, z_mm, box_size=1.6):
    ix1, ix2 = int((x_mm-box_size/2)/0.1), int((x_mm+box_size/2)/0.1)
    iz1, iz2 = int((z_mm-box_size/2)/0.1), int((z_mm+box_size/2)/0.1)
    return np.mean(img[iz1:iz2, ix1:ix2])

# ==========================================
# 4. SBR MATRIX CALCULATION (Grid of All Combinations)
# ==========================================
# Rows = Depths, Cols = Targets
sbr_matrix = np.zeros((5, 5))

for r, depth in enumerate(depths):
    # Get mean of background at this specific depth
    mean_bg = get_mean_intensity(img_norm, bg_x, depth)
    
    for c_idx, x_pos in enumerate(target_x):
        # Get mean of signal target at its specific position
        mean_sig = get_mean_intensity(img_norm, x_pos, depth)
        
        # SBR formula: ensure positive dB (Feedback point 8)
        sbr_matrix[r, c_idx] = abs(round(20 * np.log10(mean_sig / (mean_bg + 1e-12)), 2))

# Create Nice Table
df_sbr = pd.DataFrame(sbr_matrix, 
                      index=[f"Depth {d}mm" for d in depths], 
                      columns=[f"Target {lbl}" for lbl in target_labels])

print("\n--- 2D SBR MATRIX (Combination of Targets and Depths) ---")
print(df_sbr)

# ==========================================
# 5. VISUALIZATION WITH ALL BOUNDING BOXES
# ==========================================
plt.figure(figsize=(14, 8))
plt.imshow(img_norm, extent=[0, 38.4, 20, 0], cmap='hot', aspect='auto')
plt.colorbar(label="Intensity")
ax = plt.gca()

# Plot Signal Boxes (Cyan)
box_s = 1.6
for i, lbl in enumerate(target_labels):
    x, z = target_x[i], depths[i]
    rect = patches.Rectangle((x-box_s/2, z-box_s/2), box_s, box_s, linewidth=1.5, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, z-1.2, lbl, color='cyan', fontweight='bold', ha='center')

# Plot Background Boxes at the same depths (Green)
for i, d in enumerate(depths):
    rect = patches.Rectangle((bg_x-box_s/2, d-box_s/2), box_s, box_s, linewidth=1.5, edgecolor='#00FF00', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(bg_x, d-1.2, f"BG_{d}mm", color='#00FF00', fontweight='bold', ha='center', fontsize=8)

plt.title("Photoacoustic Reconstruction: SBR Grid Analysis with Bounding Boxes")
plt.xlabel("UST Lateral Position (y-position) [mm]")
plt.ylabel("Depth (x-position) [mm]")
plt.show()