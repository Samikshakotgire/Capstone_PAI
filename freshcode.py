#!/usr/bin/env python3
"""
PACTLab ULTRA v2.1 - FIXED Axes + SNR Simulation
✅ Proper axis labels | Clean sensor visualization
✅ 1/2 → 1/4 → 1/6 sensor degradation simulation
✅ Show ONLY current sensor + background ROIs
✅ Biomedical publication standard
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import hilbert
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

class PACTLabUltraFixed:
    def __init__(self, mat_file):
        print("🔬 PACTLab ULTRA v2.1 - Enhanced Edition")
        
        # Core parameters
        self.fs = 40e6; self.c = 1500; self.dt = 1/self.fs
        self.nelem = 128; self.aperture = 0.0384
        
        # Load data
        data = sio.loadmat(mat_file)
        key = [k for k in data if not k.startswith('_')][0]
        self.signals = data[key]
        if self.signals.shape[0] != self.nelem: self.signals = self.signals.T
        
        # Grid
        self.dx = 0.1e-3
        self.xvec = np.arange(0, 0.0385, self.dx)
        self.zvec = np.arange(0, 0.0201, self.dx)
        self.X, self.Z = np.meshgrid(self.xvec, self.zvec)
        self.xtrans = np.linspace(0, self.aperture, self.nelem)
        
        # 5 diagonal targets
        self.target_pos = np.array([[6.7e-3,5e-3], [12.95e-3,7.5e-3], 
                                   [19.2e-3,10e-3], [25.45e-3,12.5e-3], 
                                   [31.7e-3,15e-3]])
        self.bg_x = 2.0e-3
        
        print("✅ Setup complete")

    def simulate_degradation(self, signals, factor):
        """Simulate sensor degradation: keep 1/factor sensors"""
        print(f"   📉 Degrading to 1/{factor} sensors...")
        keep_sensors = np.random.choice(self.nelem, self.nelem//factor, replace=False)
        degraded = np.zeros_like(signals)
        degraded[keep_sensors] = signals[keep_sensors]
        return degraded

    def process_pipeline(self, signals):
        """Full processing pipeline"""
        # Envelope
        env = np.abs(hilbert(signals, axis=1))
        env = np.clip(env, 0, np.percentile(env, 98))
        
        # Backprojection
        cube = np.zeros((self.nelem, len(self.zvec), len(self.xvec)))
        for i in range(self.nelem):
            r = np.sqrt((self.X - self.xtrans[i])**2 + self.Z**2)
            t_idx = np.round(r / (self.c * self.dt)).astype(int)
            valid = (t_idx >= 0) & (t_idx < 1024) & (r < 0.018)
            if np.any(valid):
                cube[i, valid] = env[i, t_idx[valid]]
        
        # DMAS beamform
        dmas = np.sum(np.sign(cube) * np.sqrt(np.abs(cube)), 0)**2
        return dmas

    def plot_single_target(self, image, target_idx, snr_factor, current_sensor=None):
        """Plot SINGLE target with ONLY current sensor + background"""
        img_norm = image / np.max(image)
        x_tgt, z_tgt = self.target_pos[target_idx]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Image with axes
        im = ax.imshow(img_norm, extent=[0, 38.4, 20, 0], cmap='hot', 
                      vmin=0, vmax=0.6, aspect='auto')
        ax.set_xlabel('Lateral Position (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Depth (mm)', fontsize=12, fontweight='bold')
        ax.set_title(f'Target {chr(65+target_idx)} | 1/{snr_factor} Sensors', 
                    fontsize=14, fontweight='bold')
        
        # CURRENT SENSOR only (red box)
        if current_sensor is not None:
            sensor_x = self.xtrans[current_sensor]
            ax.add_patch(mpatches.Rectangle((sensor_x*1e3-0.2, 0), 0.4, 20, 
                                          fc='none', ec='red', lw=3, ls='-'))
            ax.text(sensor_x*1e3, 19, f'Sensor #{current_sensor}', 
                   color='red', fontweight='bold', ha='center')
        
        # Target ROI (cyan)
        ax.add_patch(mpatches.Rectangle((x_tgt*1e3-1.2, z_tgt*1e3-1.2), 2.4, 2.4, 
                                      fc='none', ec='cyan', lw=3))
        
        # Background ROI (green)
        ax.add_patch(mpatches.Rectangle((self.bg_x*1e3-1.2, z_tgt*1e3+0.7-1.2), 2.4, 2.4, 
                                      fc='none', ec='lime', ls='--', lw=3))
        
        # SBR text
        sig_roi = self._get_roi(img_norm, x_tgt, z_tgt)
        bg_roi = self._get_roi(img_norm, self.bg_x, z_tgt + 0.0007)
        sbr = 20 * np.log10(np.mean(sig_roi[sig_roi>np.mean(sig_roi)*0.2]) / 
                           np.mean(bg_roi[bg_roi>np.mean(bg_roi)*0.1]))
        ax.text(19, 18, f'SBR: {sbr:.1f} dB', fontsize=14, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.colorbar(im, ax=ax, label='Normalized Intensity')
        plt.tight_layout()
        plt.show()
        
        print(f"Target {chr(65+target_idx)} (1/{snr_factor}): SBR = {sbr:.1f} dB")
        return sbr

    def _get_roi(self, img, x, z, size=12):
        ix, iz = int(x/self.dx), int(z/self.dx)
        return img[iz-size:iz+size, ix-size:ix+size]

# ============================================================================
# 🎬 MAIN EXECUTION - SNR DEGRADATION SERIES
# ============================================================================
def main():
    print("🎬 PACTLab ULTRA v2.1 - Sensor Degradation Analysis")
    lab = PACTLabUltraFixed('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
    
    # Full sensor baseline
    print("\n📡 BASELINE: Full 128 sensors")
    base_img = lab.process_pipeline(lab.signals)
    
    # SNR degradation: 1/2 → 1/4 → 1/6 sensors
    snr_factors = [2, 4, 6]
    
    for target_id in range(5):  # Targets A-E
        print(f"\n🎯 TARGET {chr(65+target_id)} ANALYSIS:")
        x_tgt, z_tgt = lab.target_pos[target_id]
        
        # Find closest sensor to target
        closest_sensor = np.argmin(np.abs(lab.xtrans - x_tgt))
        print(f"   Closest sensor: #{closest_sensor} (x={lab.xtrans[closest_sensor]*1e3:.1f}mm)")
        
        for factor in snr_factors:
            # Degrade sensors
            degraded_signals = lab.simulate_degradation(lab.signals, factor)
            degraded_img = lab.process_pipeline(degraded_signals)
            
            # Plot ONLY current sensor + target + background
            lab.plot_single_target(degraded_img, target_id, factor, closest_sensor)
        
        print("-" * 50)

if __name__ == "__main__":
    main()
