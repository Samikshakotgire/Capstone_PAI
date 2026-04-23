import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

# ==========================================
# PARAMETERS
# ==========================================
fs           = 40e6       # Sampling frequency (Hz)
c            = 1500       # Speed of sound (m/s)
pitch        = 0.3e-3     # Sensor pitch (m)
num_sensors  = 128
num_samples  = 1024
dt           = 1 / fs

# ==========================================
# LOAD DATA
# ==========================================
data         = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data  = data['sensor_data']   # shape: (128, 1024)

# ==========================================
# SENSOR POSITIONS (linear array, centered)
# ==========================================
x_sensor = (np.arange(num_sensors) - (num_sensors - 1) / 2) * pitch

# ==========================================
# BANDPASS FILTER (1–10 MHz)
# ==========================================
def bandpass(data, low=1e6, high=10e6, fs=40e6, order=4):
    nyq  = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

filtered = bandpass(sensor_data)

# ==========================================
# ENVELOPE (Hilbert)
# ==========================================
envelope = np.abs(hilbert(filtered, axis=1))   # shape: (128, 1024)

# ==========================================
# RECONSTRUCTION GRID
# ==========================================
dx     = 0.1e-3                                 # 0.1 mm resolution
x_grid = np.arange(-19.05e-3, 19.05e-3, dx)    # lateral
z_grid = np.arange(0.5e-3,    20e-3,    dx)    # depth
X, Z   = np.meshgrid(x_grid, z_grid)

# ==========================================
# DELAY INDICES FOR ALL SENSORS
# ==========================================
print("Computing delay indices...")
# Shape: (num_sensors, Nz, Nx)
Nz, Nx = Z.shape
delay_idx = np.zeros((num_sensors, Nz, Nx), dtype=np.int32)
valid_mask = np.ones((num_sensors, Nz, Nx),  dtype=bool)

for i in range(num_sensors):
    dist = np.sqrt((X - x_sensor[i])**2 + Z**2)
    idx  = np.round(dist / (c * dt)).astype(int)
    mask = (idx >= 0) & (idx < num_samples)
    idx[~mask] = 0
    delay_idx[i]  = idx
    valid_mask[i] = mask

# ==========================================
# DELAYED SIGNALS  (envelope-based)
# ==========================================
print("Extracting delayed signals...")
delayed = np.zeros((num_sensors, Nz, Nx), dtype=np.float32)
for i in range(num_sensors):
    tmp = np.zeros((Nz, Nx), dtype=np.float32)
    m   = valid_mask[i]
    tmp[m] = envelope[i, delay_idx[i][m]]
    delayed[i] = tmp

# ==========================================
# DMAS  — Delay Multiply and Sum
# Formula: sum over all pairs (i<j) of
#          sign(si*sj) * sqrt(|si*sj|)
# ==========================================
print("Running DMAS (this takes ~1–2 min)...")
img_DMAS = np.zeros((Nz, Nx), dtype=np.float64)

for i in range(num_sensors - 1):
    for j in range(i + 1, num_sensors):
        prod      = delayed[i].astype(np.float64) * delayed[j].astype(np.float64)
        img_DMAS += np.sign(prod) * np.sqrt(np.abs(prod))

print("DMAS done!")

# ==========================================
# DISPLAY
# ==========================================
img = np.abs(img_DMAS)
img /= np.max(img) + 1e-12

# Log compression — 40 dB dynamic range
img_log = 20 * np.log10(img + 1e-6)
img_log = np.clip(img_log, -40, 0)

plt.figure(figsize=(6, 8))
plt.imshow(img_log,
           extent=[x_grid[0]*1e3, x_grid[-1]*1e3,
                   z_grid[-1]*1e3, z_grid[0]*1e3],
           cmap='hot', aspect='auto', vmin=-40, vmax=0)
plt.colorbar(label='Normalized Intensity (dB)')
plt.title('DMAS Reconstruction — 5 Diagonal Targets')
plt.xlabel('Lateral (mm)')
plt.ylabel('Depth (mm)')
plt.tight_layout()
plt.savefig('DMAS_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: DMAS_result.png")
