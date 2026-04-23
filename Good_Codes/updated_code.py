import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hilbert
import pandas as pd

# ==========================================
# 1. SETUP & PARAMETERS (Corrected to Positive Axes)
# ==========================================
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat' 
data = sio.loadmat(mat_file)
var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name].T if data[var_name].shape[0] != 128 else data[var_name]

# System Parameters
fs, c = 40e6, 1500
total_length = 38.4  # mm
num_elements = 128
dt = 1/fs

# Positive X-axis: 0 to 38.4 mm
x_sensor = np.linspace(0, total_length, num_elements) * 1e-3 # convert to meters

# Reconstruction Grid (0 to 38.4mm width, 0 to 20mm depth)
dx = 0.1e-3
x_grid = np.arange(0, 38.5e-3, dx)
z_grid = np.arange(0, 20.0e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)

# ==========================================
# 2. RECONSTRUCTION (UBP for analysis)
# ==========================================
p_env = np.abs(hilbert(p_sensor, axis=1))

def get_img(data_in):
    img = np.zeros(X.shape)
    for i in range(num_elements):
        dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
        idx = np.round(dist / (c * dt)).astype(int)
        mask = (idx >= 0) & (idx < 1024)
        img[mask] += data_in[i, idx[mask]]
    return img

print("Reconstructing...")
img_final = get_img(p_env)
img_norm = img_final / np.max(img_final)

# ==========================================
# 3. TARGET POSITIONING & ROI DEFINITION (A, B, C, D, E)
# ==========================================
# Based on assignment: 5 diagonal dots, center is aligned with element 64 (19.2mm)
# Estimated positions in mm: [X, Z]
targets = {
    'A': [12.5, 5.0],
    'B': [15.8, 7.5],
    'C': [19.2, 10.0], # Center Dot
    'D': [22.6, 12.5],
    'E': [25.9, 15.0]
}
bg_roi = [2, 2, 4, 4] # [X_start, Z_start, width, height] in mm

def get_roi_stats(img, center_x, center_z, size=1.5):
    # Convert mm to grid indices
    x1, x2 = int((center_x-size/2)/0.1), int((center_x+size/2)/0.1)
    z1, z2 = int((center_z-size/2)/0.1), int((center_z+size/2)/0.1)
    region = img[z1:z2, x1:x2]
    return np.mean(region), (x1*0.1, z1*0.1, size, size)

# ==========================================
# 4. SBR CALCULATION & TABLE GENERATION
# ==========================================
bg_mean, bg_rect = get_roi_stats(img_norm, 4, 4, size=3)
results_table = []

for name, pos in targets.items():
    sig_mean, rect_coords = get_roi_stats(img_norm, pos[0], pos[1])
    sbr = 20 * np.log10(sig_mean / (bg_mean + 1e-9))
    results_table.append({"Target": name, "X (mm)": pos[0], "Z (mm)": pos[1], "SBR (dB)": round(abs(sbr), 2)})

df = pd.DataFrame(results_table)
print("\n--- Target Analysis Table ---")
print(df)

# ==========================================
# 5. VISUALIZATION WITH BOUNDING BOXES
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(img_norm, extent=[0, 38.4, 20, 0], cmap='hot', aspect='auto')
plt.colorbar(im, label="Normalized Intensity")

# Draw Target Boxes (Signal)
for name, pos in targets.items():
    _, (x, z, w, h) = get_roi_stats(img_norm, pos[0], pos[1])
    rect = patches.Rectangle((x, z), w, h, linewidth=1, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, z-0.5, name, color='cyan', fontweight='bold')

# Draw Background Box
rect_bg = patches.Rectangle((bg_rect[0], bg_rect[1]), bg_rect[2], bg_rect[3], 
                            linewidth=1, edgecolor='green', facecolor='none', linestyle='--')
ax.add_patch(rect_bg)
ax.text(bg_rect[0], bg_rect[1]-0.5, "BG", color='green', fontweight='bold')

plt.title("Refined Photoacoustic Image with SBR Bounding Boxes")
plt.xlabel("UST Position (X) [mm]")
plt.ylabel("Depth (Z) [mm]")
plt.show()