"""
Photoacoustic Computed Tomography (PACT) Reconstruction & Analysis Suite
Implementation of UBP, DMAS, SLSC, and MV Algorithms with SBR Matrix Validation
Author: [Your Name]
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hilbert
import pandas as pd

class PACTLab:
    def __init__(self, file_path):
        # 1. System Constants (Assignment Specs)
        self.fs = 40e6          # Sampling Frequency (40 MHz)
        self.c = 1500           # Speed of Sound (1500 m/s)
        self.pitch = 0.3e-3     # Transducer Pitch (0.3 mm)
        self.num_elements = 128
        self.total_len = 38.4e-3 
        self.dt = 1/self.fs
        
        # 2. Data Loading & Formatting
        mat_data = sio.loadmat(file_path)
        var_name = [k for k in mat_data.keys() if not k.startswith('_')][0]
        self.p_raw = mat_data[var_name]
        if self.p_raw.shape[0] != 128: self.p_raw = self.p_raw.T
        
        # 3. Computational Grid (0.1mm resolution)
        self.dx = 0.1e-3
        self.x_grid = np.arange(0, 38.5e-3, self.dx)
        self.z_grid = np.arange(0, 20.1e-3, self.dx)
        self.X, self.Z = np.meshgrid(self.x_grid, self.z_grid)
        
        # 4. Sensor Geometry
        self.x_sensor = np.linspace(0, self.total_len, self.num_elements)
        
        # 5. Define Analysis Regions [X_pos, Z_depth]
        self.target_labels = ['A', 'B', 'C', 'D', 'E']
        self.target_x = [6.7, 12.95, 19.2, 25.45, 31.7]
        self.depths = [5.0, 7.5, 10.0, 12.5, 15.0]
        self.bg_x = 2.0 # Fixed lateral position for background column

    def preprocess(self):
        """Applies Hilbert Transform for Envelope Detection"""
        return np.abs(hilbert(self.p_raw, axis=1))

    def compute_delay_matrix(self, envelope_data):
        """Maps Time-domain signals to Spatial-domain pixels"""
        delayed_cube = np.zeros((self.num_elements, len(self.z_grid), len(self.x_grid)))
        for i in range(self.num_elements):
            dist = np.sqrt((self.X - self.x_sensor[i])**2 + self.Z**2)
            time_idx = np.round(dist / (self.c * self.dt)).astype(int)
            mask = (time_idx >= 0) & (time_idx < 1024)
            delayed_cube[i, mask] = envelope_data[i, time_idx[mask]]
        return delayed_cube

    def run_beamformers(self, delayed_cube):
        """Executes 4 major PA reconstruction algorithms"""
        # UBP: Linear Summation
        img_ubp = np.sum(delayed_cube, axis=0)
        
        # DMAS: Non-linear pair-wise multiplication (approx)
        img_dmas = np.sum(np.sign(delayed_cube) * np.sqrt(np.abs(delayed_cube)), axis=0)**2
        
        # SLSC: Short-lag spatial coherence (Lag 1-5)
        img_slsc = np.zeros(self.X.shape)
        for m in range(1, 6):
            for i in range(self.num_elements - m):
                img_slsc += (delayed_cube[i] * delayed_cube[i+m])
        
        # MV: Coherence Factor Weighting
        cf = (np.sum(delayed_cube, axis=0)**2) / (self.num_elements * np.sum(delayed_cube**2, axis=0) + 1e-12)
        img_mv = img_ubp * cf
        
        return {"UBP": img_ubp, "DMAS": img_dmas, "SLSC": img_slsc, "MV": img_mv}

    def plot_analysis_matrix(self, image, title_name):
        """Generates the 5x5 SBR Diagnostic Grid"""
        img_n = image / np.max(image)
        fig, axes = plt.subplots(5, 5, figsize=(20, 16))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for r in range(5):
            depth = self.depths[r]
            for c_idx in range(5):
                ax = axes[r, c_idx]
                tx = self.target_x[c_idx]
                
                # Calculate SBR
                # Get mean of signal and background regions (1.6mm boxes)
                def get_m(x, z):
                    ix, iz = int(x/0.1), int(z/0.1)
                    return np.mean(img_n[iz-8:iz+8, ix-8:ix+8])
                
                m_sig = get_m(tx, depth)
                m_bg = get_m(self.bg_x, depth)
                sbr = abs(round(20 * np.log10(m_sig / (m_bg + 1e-12)), 2))
                
                # Plot
                ax.imshow(img_n, extent=[0, 38.4, 20, 0], cmap='hot', aspect='auto')
                ax.set_xlim(0, 35); ax.set_ylim(depth+3, depth-3)
                
                # Bounding Boxes
                ax.add_patch(patches.Rectangle((tx-0.8, depth-0.8), 1.6, 1.6, color='cyan', fill=False, lw=1.5))
                ax.add_patch(patches.Rectangle((self.bg_x-0.8, depth-0.8), 1.6, 1.6, color='#00FF00', fill=False, ls='--', lw=1.5))
                
                if r == 0: ax.set_title(f"Target {self.target_labels[c_idx]}", fontweight='bold')
                if c_idx == 0: ax.set_ylabel(f"Depth {depth}mm", fontweight='bold')
                ax.text(tx, depth+2.5, f"SBR: {sbr} dB", color='yellow', ha='center', fontweight='bold')

        plt.suptitle(f"SBR Matrix Analysis: {title_name} Algorithm", fontsize=22, fontweight='bold', y=0.96)
        plt.show()

# --- Seminar Execution Flow ---
lab = PACTLab('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
env = lab.preprocess()
cube = lab.compute_delay_matrix(env)
results = lab.run_beamformers(cube)

# Plotting the DMAS results as a seminar highlight
lab.plot_analysis_matrix(results["DMAS"], "DMAS")