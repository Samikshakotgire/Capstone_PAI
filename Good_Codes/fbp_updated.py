import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hilbert
from scipy.fft import fft, ifft, fftfreq

class ChenInspiredPACT:
    def __init__(self, file_path):
        # 1. System Constants (Assignment Specs)
        self.fs = 40e6
        self.c = 1500
        self.num_elements = 128
        self.num_samples = 1024
        self.dt = 1/self.fs
        
        # 2. Data Loading & Transpose Correction
        mat_data = sio.loadmat(file_path)
        var_name = [k for k in mat_data.keys() if not k.startswith('_')][0]
        self.p_raw = mat_data[var_name]
        if self.p_raw.shape[0] != 128: self.p_raw = self.p_raw.T
        
        # 3. Computational Grid (0-38.4mm lateral, 0-20mm depth)
        self.dx = 0.1e-3
        self.x_grid = np.arange(0, 38.5e-3, self.dx)
        self.z_grid = np.arange(0, 20.1e-3, self.dx)
        self.X, self.Z = np.meshgrid(self.x_grid, self.z_grid)
        self.x_sensor = np.linspace(0, 38.4e-3, self.num_elements)
        
        # Labels for Analysis
        self.target_labels = ['A', 'B', 'C', 'D', 'E']
        self.target_x = [6.7, 12.95, 19.2, 25.45, 31.7]
        self.depths = [5.0, 7.5, 10.0, 12.5, 15.0]

    def physics_informed_preprocessing(self):
        """
        Implements Logic from Chen et al. (2026):
        1. Reflection Padding (Section III-D) to handle limited view edges.
        2. Gaussian Low-Pass Regularization (Section IV-C).
        """
        # Step A: Reflection Padding on the 'Ring' (Sensor) dimension
        # This simulates data 'beyond the arc' as proposed in the paper
        p_padded = np.pad(self.p_raw, ((10, 10), (0, 0)), mode='reflect')
        
        # Step B: Regularized Filtering in Frequency Domain
        n = self.num_samples
        freqs = fftfreq(n)
        
        # The Paper's linear filter logic (Approximated by a Gaussian-Ramp)
        # Ramp filters sharpen, Gaussian suppresses noise (Regularization)
        ramp = np.abs(freqs)
        gaussian_sigma = 0.1  # Regularization parameter from paper logic
        regularizer = np.exp(-freqs**2 / (2 * gaussian_sigma**2))
        
        data_fft = fft(p_padded, axis=1)
        filtered_fft = data_fft * ramp * regularizer
        p_filtered = np.real(ifft(filtered_fft, axis=1))
        
        # Remove padding and take envelope
        p_final = p_filtered[10:-10, :]
        return np.abs(hilbert(p_final, axis=1))

    def reconstruct(self, data_processed):
        """Standard Adjoint Backprojection H^T (as used in FBP)"""
        img = np.zeros(self.X.shape)
        for i in range(self.num_elements):
            dist = np.sqrt((self.X - self.x_sensor[i])**2 + self.Z**2)
            t_idx = np.round(dist / (self.c * self.dt)).astype(int)
            mask = (t_idx >= 0) & (t_idx < self.num_samples)
            img[mask] += data_processed[i, t_idx[mask]]
        return img / np.max(img)

    def generate_sbr_matrix_viz(self, img_norm):
        """Generates the 5x5 seminar visualization"""
        fig, axes = plt.subplots(5, 5, figsize=(18, 16))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        bg_x = 2.0 # local background column
        
        for r in range(5):
            depth = self.depths[r]
            for c_idx in range(5):
                ax = axes[r, c_idx]
                tx = self.target_x[c_idx]
                
                # SBR logic
                def get_m(x, z):
                    ix, iz = int(x/0.1), int(z/0.1)
                    return np.mean(img_norm[iz-8:iz+8, ix-8:ix+8])
                
                s_mean = get_m(tx, depth)
                b_mean = get_m(bg_x, depth)
                sbr = abs(round(20 * np.log10(s_mean / (b_mean + 1e-12)), 2))
                
                ax.imshow(img_norm, extent=[0, 38.4, 20, 0], cmap='hot', aspect='auto')
                ax.set_xlim(0, 35); ax.set_ylim(depth+3, depth-3)
                ax.add_patch(patches.Rectangle((tx-0.8, depth-0.8), 1.6, 1.6, color='cyan', fill=False, lw=1.5))
                ax.add_patch(patches.Rectangle((bg_x-0.8, depth-0.8), 1.6, 1.6, color='#00FF00', fill=False, ls='--', lw=1.5))
                
                if r == 0: ax.set_title(f"Target {self.target_labels[c_idx]}")
                if c_idx == 0: ax.set_ylabel(f"Depth {depth}mm")
                ax.text(tx, depth+2.5, f"SBR: {sbr} dB", color='yellow', ha='center', fontweight='bold')

        plt.suptitle("Seminar Analysis: Learned FBP Logic (Chen et al. 2026)", fontsize=20, y=0.96)
        plt.show()

# --- RUN SEMINAR SUITE ---
lab = ChenInspiredPACT('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
processed_data = lab.physics_informed_preprocessing()
final_image = lab.reconstruct(processed_data)
lab.generate_sbr_matrix_viz(final_image)