import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import hilbert

class XuWangUniversalPACT:
    def __init__(self, file_path):
        # 1. System Constants (Assignment Specs)
        self.fs = 40e6
        self.c = 1500
        self.num_elements = 128
        self.num_samples = 1024
        self.dt = 1/self.fs
        
        # 2. Data Loading
        mat_data = sio.loadmat(file_path)
        var_name = [k for k in mat_data.keys() if not k.startswith('_')][0]
        self.p_raw = mat_data[var_name]
        if self.p_raw.shape[0] != 128: self.p_raw = self.p_raw.T
        
        # 3. Grid (0-38.4mm lateral, 0-20mm depth)
        self.dx = 0.1e-3
        self.x_grid = np.arange(0, 38.5e-3, self.dx)
        self.z_grid = np.arange(0, 20.1e-3, self.dx)
        self.X, self.Z = np.meshgrid(self.x_grid, self.z_grid)
        self.x_sensor = np.linspace(0, 38.4e-3, self.num_elements)
        
        # Labels for Analysis
        self.target_labels = ['A', 'B', 'C', 'D', 'E']
        self.target_x = [6.7, 12.95, 19.2, 25.45, 31.7]
        self.depths = [5.0, 7.5, 10.0, 12.5, 15.0]

    def xu_wang_filtering(self):
        """
        Implements Equation 20: b = 2p - 2t*(dp/dt)
        This is the 'Universal' part that makes the reconstruction exact.
        """
        p = self.p_raw
        t = np.arange(self.num_samples) * self.dt
        
        # Calculate time derivative: dp/dt
        dp_dt = np.gradient(p, self.dt, axis=1)
        
        # Formula: b = 2p - 2t * (dp/dt)
        # Note: In most high-freq systems, the 2t*dp/dt term dominates
        b_term = 2 * p - 2 * t * dp_dt
        return b_term

    def exact_backprojection(self, b_data):
        """
        Implements Equation 22: Summation weighted by Solid Angle (cos theta / R^2)
        """
        img = np.zeros(self.X.shape)
        total_weight = np.zeros(self.X.shape)
        
        for i in range(self.num_elements):
            # Distance R
            dX = self.X - self.x_sensor[i]
            dZ = self.Z # Since sensors are at Z=0
            dist = np.sqrt(dX**2 + dZ**2)
            
            # Time to index
            t_idx = np.round(dist / (self.c * self.dt)).astype(int)
            mask = (t_idx >= 0) & (t_idx < self.num_samples)
            
            # Solid Angle Weight (Eq. 22): cos(theta) / R^2
            # cos(theta) = dZ / dist
            weight = dZ / (dist**3 + 1e-9) 
            
            # Apply weighted backprojection
            img[mask] += weight[mask] * b_data[i, t_idx[mask]]
            total_weight[mask] += weight[mask]
            
        # Normalize by total solid angle to correct for limited view (Section III)
        img_final = img / (total_weight + 1e-9)
        return img_final / np.max(np.abs(img_final))

    def generate_seminar_plot(self, img_norm):
        fig, axes = plt.subplots(5, 5, figsize=(18, 16))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        bg_x = 2.0
        
        for r in range(5):
            depth = self.depths[r]
            for c_idx in range(5):
                ax = axes[r, c_idx]
                tx = self.target_x[c_idx]
                
                def get_m(x, z):
                    ix, iz = int(x/0.1), int(z/0.1)
                    return np.mean(np.abs(img_norm[iz-8:iz+8, ix-8:ix+8]))
                
                s_mean = get_m(tx, depth)
                b_mean = get_m(bg_x, depth)
                sbr = abs(round(20 * np.log10(s_mean / (b_mean + 1e-12)), 2))
                
                ax.imshow(np.abs(img_norm), extent=[0, 38.4, 20, 0], cmap='hot', aspect='auto')
                ax.set_xlim(0, 35); ax.set_ylim(depth+3, depth-3)
                ax.add_patch(patches.Rectangle((tx-0.8, depth-0.8), 1.6, 1.6, color='cyan', fill=False, lw=1.5))
                ax.add_patch(patches.Rectangle((bg_x-0.8, depth-0.8), 1.6, 1.6, color='#00FF00', fill=False, ls='--', lw=1.5))
                
                if r == 0: ax.set_title(f"Target {self.target_labels[c_idx]}")
                if c_idx == 0: ax.set_ylabel(f"Depth {depth}mm")
                ax.text(tx, depth+2.5, f"SBR: {sbr} dB", color='yellow', ha='center', fontweight='bold')

        plt.suptitle("Seminar Analysis: Universal Back-Projection (Xu & Wang 2005)", fontsize=20, y=0.96)
        plt.show()

# --- EXECUTE ---
lab = XuWangUniversalPACT('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
b_filtered = lab.xu_wang_filtering()
final_image = lab.exact_backprojection(b_filtered)
lab.generate_seminar_plot(final_image)