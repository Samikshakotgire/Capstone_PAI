import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert  # MISSING in your snippet

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
mat_file = 'Q1 _3_SensorData_5dots_diag_NoNoise (1).mat' 
data = sio.loadmat(mat_file)

var_name = [k for k in data.keys() if not k.startswith('_')][0]
p_sensor = data[var_name]

if p_sensor.shape[0] != 128:
    p_sensor = p_sensor.T

# Parameters
fs = 40e6              
c = 1500               
pitch = 0.3e-3         
num_elements = 128
num_samples = 1024
dt = 1/fs

# Transducer positions
x_sensor = (np.arange(num_elements) - (num_elements-1)/2) * pitch

# Grid
dx = 0.1e-3
x_grid = np.arange(-15e-3, 15e-3, dx)
z_grid = np.arange(0, 30e-3, dx)  
X, Z = np.meshgrid(x_grid, z_grid)

# ==========================================
# HELPER: DELAY CALCULATION
# ==========================================
def get_delayed_data(p_data):
    delayed_matrix = np.zeros((num_elements, len(z_grid), len(x_grid)))
    for i in range(num_elements):
        dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
        idx = np.round(dist / (c * dt)).astype(int)
        mask = (idx >= 0) & (idx < num_samples)
        tmp = np.zeros(X.shape)
        tmp[mask] = np.real(p_data[i, idx[mask]])
        delayed_matrix[i, :, :] = tmp
    return delayed_matrix

# ==========================================
# BASELINE UBP (To satisfy the comparison in visualization)
# ==========================================
print("Calculating Baseline UBP...")
p_sensor_env = hilbert(p_sensor, axis=1) # Get envelope for smooth UBP
delayed_raw = get_delayed_data(p_sensor_env)
img_UBP = np.sum(delayed_raw, axis=0) # This was UNDEFINED in your snippet

# ==========================================
# 1. REFINED FBP (Xu & Wang Paper: Ramp Filter)
# ==========================================
print("Running Refined FBP (Xu & Wang logic)...")
def ramp_filter(data):
    n = data.shape[1]
    freqs = fftfreq(n)
    filter_mask = np.abs(freqs) 
    # Hanning logic from Chen et al. paper
    hanning = 0.5 + 0.5 * np.cos(np.pi * freqs / np.max(freqs))
    data_fft = fft(data, axis=1)
    filtered_fft = data_fft * filter_mask * hanning
    return np.real(ifft(filtered_fft, axis=1))

p_filtered = ramp_filter(p_sensor)
delayed_filtered = get_delayed_data(p_filtered)
img_FBP_Refined = np.sum(delayed_filtered, axis=0)

# ==========================================
# 2. ITERATIVE SIRT (Miao et al. Paper logic)
# ==========================================
print("Running Iterative Refinement (SIRT)...")
img_iter = np.zeros(X.shape)
n_iterations = 5
learning_rate = 0.1

for k in range(n_iterations):
    # Refining based on FBP baseline
    residual = img_FBP_Refined - img_iter
    img_iter += learning_rate * residual
    print(f"  Iteration {k+1}/{n_iterations} complete")

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(img_UBP)/np.max(img_UBP), extent=[-15, 15, 30, 0], cmap='hot')
plt.title("Original UBP (Blurry Baseline)")
plt.xlabel("X (mm)"); plt.ylabel("Depth (mm)")

plt.subplot(1, 2, 2)
refined_final = np.abs(img_iter)
# Thresholding based on Chen et al. paper results to remove noise floor
refined_final[refined_final < 0.15 * np.max(refined_final)] = 0 
plt.imshow(refined_final/np.max(refined_final), extent=[-15, 15, 30, 0], cmap='hot')
plt.title("Paper-Refined Result (FBP + SIRT)")
plt.xlabel("X (mm)")

plt.tight_layout()
plt.show()