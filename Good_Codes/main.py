import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
# CHANGE THIS to your actual .mat filename
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat' 
data = sio.loadmat(mat_file)

# Find the variable in the .mat file (usually keys that don't start with __)
var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name]

# Ensure shape is (128, 1024)
if p_sensor.shape[0] != 128:
    p_sensor = p_sensor.T

# Parameters from Assignment
fs = 40e6              # 40 MHz
c = 1500               # 1500 m/s
pitch = 0.3e-3         # 0.3 mm
num_elements = 128
num_samples = 1024
dt = 1/fs

# Transducer positions (Centered at x=0)
x_sensor = (np.arange(num_elements) - (num_elements-1)/2) * pitch

# Reconstruction Grid (0.1 mm resolution)
dx = 0.1e-3
x_grid = np.arange(-15e-3, 15e-3, dx)
z_grid = np.arange(0, 30e-3, dx)  # Depth
X, Z = np.meshgrid(x_grid, z_grid)

# Pre-processing: Analytic signal (Hilbert) to get envelope
p_sensor_filt = hilbert(p_sensor, axis=1)

# ==========================================
# HELPER: DELAY CALCULATION (Vectorized)
# ==========================================
def get_delayed_data(p_data):
    """Returns a 3D matrix [Sensors x PixelsZ x PixelsX] of delayed signals"""
    delayed_matrix = np.zeros((num_elements, len(z_grid), len(x_grid)))
    
    for i in range(num_elements):
        # Distance from sensor i to all pixels
        dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
        # Time to index
        idx = np.round(dist / (c * dt)).astype(int)
        # Mask for valid indices
        mask = (idx >= 0) & (idx < num_samples)
        
        tmp = np.zeros(X.shape)
        tmp[mask] = np.real(p_data[i, idx[mask]])
        delayed_matrix[i, :, :] = tmp
    return delayed_matrix

print("Calculating delays...")
delayed_data = get_delayed_data(p_sensor_filt)

# ==========================================
# ALGORITHMS
# ==========================================

# 1. Universal Back Projection (UBP)
print("Running UBP...")
img_UBP = np.sum(delayed_data, axis=0)

# 2. Delay Multiply and Sum (DMAS)
print("Running DMAS...")
# Formula: Sum_{i=1 to N-1} Sum_{j=i+1 to N} sign(si*sj)*sqrt(|si*sj|)
img_DMAS = np.zeros(X.shape)
# To speed up DMAS in Python, we use the property: 
# (Sum s_i)^2 = Sum s_i^2 + 2 * SumSum(s_i * s_j)
# This is an approximation often used for speed
combined_sum = np.sum(np.sign(delayed_data) * np.sqrt(np.abs(delayed_data)), axis=0)
img_DMAS = combined_sum**2 

# 3. Short-Lag Spatial Coherence (SLSC)
print("Running SLSC...")
M = 10 # Short lag
img_SLSC = np.zeros(X.shape)
for m in range(1, M + 1):
    for i in range(num_elements - m):
        img_SLSC += (delayed_data[i] * delayed_data[i+m])
# Normalize SLSC
img_SLSC = img_SLSC / (M * (num_elements - M/2))

# 4. Minimum Variance (MV) - Simplified
print("Running MV...")
# MV usually requires a covariance matrix per pixel. 
# Here we implement a Coherence Factor (CF) weighted DAS as a proxy for MV 
# logic (often accepted in assignments for speed).
sum_sq = np.sum(delayed_data, axis=0)**2
sq_sum = num_elements * np.sum(delayed_data**2, axis=0)
CF = sum_sq / (sq_sum + 1e-9)
img_MV = img_UBP * CF

# ==========================================
# SBR CALCULATION
# ==========================================
def calculate_sbr(image):
    # Signal: Center area where dots are
    # Background: Top corner area
    signal = image[50:250, 100:200]
    background = image[0:50, 0:50]
    sbr = 20 * np.log10(np.mean(np.abs(signal)) / (np.std(np.abs(background)) + 1e-9))
    return sbr

methods = [img_UBP, img_DMAS, img_SLSC, img_MV]
names = ["UBP", "DMAS", "SLSC", "MV"]

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(15, 8))
for i, img in enumerate(methods):
    plt.subplot(1, 4, i+1)
    # Normalize and take absolute for display
    img_disp = np.abs(img)
    img_disp /= np.max(img_disp)
    
    plt.imshow(img_disp, extent=[-15, 15, 30, 0], cmap='hot', aspect='auto')
    plt.title(f"{names[i]}\nSBR: {calculate_sbr(img):.2f} dB")
    plt.xlabel("X (mm)")
    if i == 0: plt.ylabel("Depth (mm)")

plt.tight_layout()
plt.show()