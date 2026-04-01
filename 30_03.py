#!/usr/bin/env python3
"""
PACTLab ULTRA v2.5 - 5 Targets + SHARED COLORBAR
✅ 5 panels in one window + single colorbar for intensity 0–1
✅ Each panel: 1 focused target, others dimmed
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
        print("🔬 PACTLab ULTRA v2.5 - 5 Targets + Colorbar")
        
        self.fs = 40e6; self.c = 1500; self.dt = 1/self.fs
        self.nelem = 128; self.aperture = 0.0384
        
        data = sio.loadmat(mat_file)
        key = [k for k in data if not k.startswith('_')][0]
        self.signals = data[key]
        if self.signals.shape[0] != self.nelem: self.signals = self.signals.T

        self.dx = 0.1e-3
        self.xvec = np.arange(0, 0.0385, self.dx)
        self.zvec = np.arange(0, 0.0201, self.dx)
        self.X, self.Z = np.meshgrid(self.xvec, self.zvec)
        self.xtrans = np.linspace(0, self.aperture, self.nelem)

        self.target_pos = np.array([
            [6.7e-3, 5e-3], [12.95e-3, 7.5e-3], 
            [19.2e-3, 10e-3], [25.45e-3, 12.5e-3], 
            [31.7e-3, 15e-3]
        ])
        self.bg_x = 2.0e-3
        print("✅ Setup complete")


    def simulate_degradation(self, signals, factor):
        print(f"   📉 1/{factor} sensors ({self.nelem//factor} sensors)...")
        keep_sensors = np.random.choice(self.nelem, self.nelem//factor, replace=False)
        degraded = np.zeros_like(signals)
        degraded[keep_sensors] = signals[keep_sensors]
        return degraded


    def process_pipeline(self, signals):
        env = np.abs(hilbert(signals, axis=1))
        env = np.clip(env, 0, np.percentile(env, 98))
        
        cube = np.zeros((self.nelem, len(self.zvec), len(self.xvec)))
        for i in range(self.nelem):
            r = np.sqrt((self.X - self.xtrans[i])**2 + self.Z**2)
            t_idx = np.round(r / (self.c * self.dt)).astype(int)
            valid = (t_idx >= 0) & (t_idx < 1024) & (r < 0.018)
            if np.any(valid):
                cube[i, valid] = env[i, t_idx[valid]]
        
        dmas = np.sum(np.sign(cube) * np.sqrt(np.abs(cube)), axis=0)**2
        return dmas


    def _get_roi(self, img, x, z, size=12):
        ix, iz = int(x/self.dx), int(z/self.dx)
        return img[iz-size:iz+size, ix-size:ix+size]


    def create_all_5_panels(self, image, snr_factor):
        """5‑panel figure WITH SHARED COLORBAR."""
        img_norm = image / np.max(image)
        
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
        fig.suptitle(f'PACTLab ULTRA v2.5 | 1/{snr_factor} Sensors (5 Targets Focused)',
                     fontsize=16, fontweight='bold')

        all_sbrs = []
        ims = []  # Store imshow objects for colorbar
        
        for target_idx, ax in enumerate(axes):
            x_tgt, z_tgt = self.target_pos[target_idx]
            roi_tgt = self._get_roi(img_norm, x_tgt, z_tgt)
            max_tgt = np.max(roi_tgt)

            # Thresholds
            thresh_1_2 = max_tgt * 0.5
            thresh_1_4 = max_tgt * 0.25
            thresh_1_6 = max_tgt * 0.1667

            # Mask around current target
            mask_keep = np.zeros_like(img_norm, dtype=bool)
            for size in range(1, 25):
                ix, iz = int(x_tgt/self.dx), int(z_tgt/self.dx)
                i0, i1 = max(0, iz-size), min(img_norm.shape[0], iz+size)
                j0, j1 = max(0, ix-size), min(img_norm.shape[1], ix+size)
                mask_keep[i0:i1, j0:j1] = True

            # Apply thresholding
            img_focused = img_norm.copy()
            outside = ~mask_keep
            img_focused[outside & (img_focused < thresh_1_2)] = 0.0
            img_focused[outside & (img_focused < thresh_1_4)] = 0.0
            img_focused[outside & (img_focused < thresh_1_6)] = 0.0

            # Plot imshow (store for colorbar)
            im = ax.imshow(
                img_focused,
                extent=[0, 38.4, 20, 0],
                cmap='hot',
                vmin=0.0, vmax=1.0,
                aspect='auto'
            )
            ims.append(im)

            # Contours
            contour_levels = [level * max_tgt for level in [0.2, 0.4, 0.6, 0.8]]
            ax.contour(img_focused, levels=contour_levels,
                      extent=[0, 38.4, 20, 0], colors='white', linewidths=1.2)

            # ROIs
            ax.add_patch(mpatches.Rectangle((x_tgt*1e3-1.2, z_tgt*1e3-1.2), 2.4, 2.4,
                                          fc='none', ec='cyan', lw=2))
            ax.add_patch(mpatches.Rectangle((self.bg_x*1e3-1.2, z_tgt*1e3+0.7-1.2), 2.4, 2.4,
                                          fc='none', ec='lime', lw=2, ls='--'))

            # SBR
            sig_roi = self._get_roi(img_focused, x_tgt, z_tgt)
            bg_roi = self._get_roi(img_focused, self.bg_x, z_tgt + 0.0007)
            sig_thresh = np.mean(sig_roi) * 0.2 if sig_roi.size > 0 else 0.0
            bg_thresh = np.mean(bg_roi) * 0.1 if bg_roi.size > 0 else 0.0
            sig_vals = sig_roi[sig_roi > sig_thresh]
            bg_vals = bg_roi[bg_roi > bg_thresh]
            mean_sig = np.mean(sig_vals) if sig_vals.size > 0 else 1e-10
            mean_bg = np.mean(bg_vals) if bg_vals.size > 0 else 1e-10
            sbr = 20 * np.log10(mean_sig / mean_bg)
            all_sbrs.append(sbr)

            ax.text(0.02, 0.98, f'{chr(65+target_idx)}\nSBR: {sbr:.1f}dB',
                   transform=ax.transAxes, va='top', ha='left',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Lateral (mm)')
            ax.set_ylabel('Depth (mm)')

        # ✅ SHARED COLORBAR for all 5 panels
        cbar = fig.colorbar(ims[0], ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('Normalized Intensity (0–1)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()
        print(f"1/{snr_factor} sensors → SBRs A–E: {np.round(all_sbrs, 1)} dB")
        return all_sbrs


# ============================================================================
# 🎬 MAIN
# ============================================================================
def main():
    print("🎬 PACTLab ULTRA v2.5 - 5 Targets + Colorbar")
    lab = PACTLabUltraFixed('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')

    print("\n📡 BASELINE: Full 128 sensors")
    base_img = lab.process_pipeline(lab.signals)
    lab.create_all_5_panels(base_img, 1)

    snr_factors = [2, 4, 6]
    for factor in snr_factors:
        print(f"\n🔧 1/{factor} sensors")
        degraded_signals = lab.simulate_degradation(lab.signals, factor)
        degraded_img = lab.process_pipeline(degraded_signals)
        lab.create_all_5_panels(degraded_img, factor)


if __name__ == "__main__":
    main()