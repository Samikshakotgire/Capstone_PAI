import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 1. Load Data
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat' 
data = sio.loadmat(mat_file)
var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name].T if data[var_name].shape[0] != 128 else data[var_name]

# 2. Parameters
fs, c, pitch = 40e6, 1500, 0.3e-3
num_elements, num_samples, dt = 128, 1024, 1/fs
x_sensor = (np.arange(num_elements) - (num_elements-1)/2) * pitch
dx = 0.1e-3
x_grid = np.arange(-15e-3, 15e-3, dx)
z_grid = np.arange(0, 30e-3, dx)
X, Z = np.meshgrid(x_grid, z_grid)

# Helper for Delay-and-Sum logic
def get_delayed_matrix(data_in):
    delayed = np.zeros((num_elements, len(z_grid), len(x_grid)))
    for i in range(num_elements):
        dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
        idx = np.round(dist / (c * dt)).astype(int)
        mask = (idx >= 0) & (idx < num_samples)
        delayed[i, mask] = data_in[i, idx[mask]]
    return delayed

# Pre-process
p_env = np.abs(hilbert(p_sensor, axis=1))
delayed_data = get_delayed_matrix(p_env)

# --- THE 4 REQUIRED ALGORITHMS ---

# 1. UBP (Universal Back Projection)
img_UBP = np.sum(delayed_data, axis=0)

# 2. DMAS (Delay Multiply and Sum)
# Simplified fast version: (Sum sign(s)*sqrt(abs(s)))^2
img_DMAS = np.sum(np.sign(delayed_data) * np.sqrt(np.abs(delayed_data)), axis=0)**2

# 3. SLSC (Short-Lag Spatial Coherence)
M = 10 # Short lag
img_SLSC = np.zeros(X.shape)
for m in range(1, M + 1):
    for i in range(num_elements - m):
        img_SLSC += (delayed_data[i] * delayed_data[i+m])

# 4. MV (Minimum Variance - using Coherence Factor proxy)
sum_sq = np.sum(delayed_data, axis=0)**2
sq_sum = num_elements * np.sum(delayed_data**2, axis=0)
CF = sum_sq / (sq_sum + 1e-12)
img_MV = img_UBP * CF

# --- SBR CALCULATION ---
def calculate_sbr(img):
    # Normalize
    img = np.abs(img) / np.max(np.abs(img))
    # Signal ROI: Around the center dot (roughly center of grid)
    signal_roi = img[80:120, 140:160] 
    # Background ROI: Top corner where there is no source
    background_roi = img[0:40, 0:40]
    return 20 * np.log10(np.mean(signal_roi) / (np.mean(background_roi) + 1e-12))

results = [img_UBP, img_DMAS, img_SLSC, img_MV]
names = ["UBP", "DMAS", "SLSC", "MV"]

# Final Visualization
plt.figure(figsize=(16, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(np.abs(results[i]), extent=[-15, 15, 30, 0], cmap='hot', aspect='auto')
    sbr_val = calculate_sbr(results[i])
    plt.title(f"{names[i]}\nSBR: {sbr_val:.2f} dB")
    plt.xlabel("X (mm)")
    if i == 0: plt.ylabel("Depth (mm)")

plt.tight_layout()
plt.show()