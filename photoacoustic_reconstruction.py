"""
Photoacoustic Imaging Reconstruction Algorithms
Implements: UBP, DMAS, SLSC, and MV
Author: AI Assistant
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

class PhotoacousticReconstruction:
    """
    Class for photoacoustic imaging reconstruction using linear array transducer
    """
    
    def __init__(self, sensor_data_path):
        """
        Initialize with transducer and imaging parameters
        
        Parameters:
        -----------
        sensor_data_path : str
            Path to .mat file containing sensor_data
        """
        # Load sensor data
        mat_data = loadmat(sensor_data_path)
        self.sensor_data = mat_data['sensor_data']  # Shape: (128, 1024)
        
        # Transducer parameters
        self.n_elements = 128
        self.n_samples = 1024
        self.fs = 40e6  # Sampling frequency: 40 MHz
        self.pitch = 0.3e-3  # Element pitch: 0.3 mm
        self.total_length = 38.4e-3  # Total length: 38.4 mm
        self.c = 1500  # Speed of sound: 1500 m/s
        
        # Time axis
        self.dt = 1 / self.fs
        self.t = np.arange(self.n_samples) * self.dt
        
        # Element positions (centered at origin, then shifted to top)
        element_indices = np.arange(self.n_elements)
        self.element_positions = (element_indices - self.n_elements/2 + 0.5) * self.pitch
        
        # Define imaging grid (positive coordinates only as per requirement)
        # Y-axis: lateral (along transducer), X-axis: depth (axial)
        self.y_min, self.y_max = 0, 50e-3  # 0 to 50 mm lateral
        self.x_min, self.x_max = 0, 20e-3  # 0 to 20 mm depth
        
        self.ny = 200  # Number of lateral pixels
        self.nx = 100  # Number of axial pixels
        
        self.y_grid = np.linspace(self.y_min, self.y_max, self.ny)
        self.x_grid = np.linspace(self.x_min, self.x_max, self.nx)
        
        # Create meshgrid
        self.Y, self.X = np.meshgrid(self.y_grid, self.x_grid)
        
        # Shift element positions to align with grid
        # Center element should align with y = 25 mm (middle of grid)
        self.element_y_positions = self.element_positions + 25e-3
        self.element_x_positions = np.zeros(self.n_elements)  # At x = 0 (top)
        
        # Apply envelope detection (Hilbert transform)
        self.sensor_data_analytic = np.abs(hilbert(self.sensor_data, axis=1))
        
        print(f"Sensor data shape: {self.sensor_data.shape}")
        print(f"Imaging grid: Y [{self.y_min*1e3:.1f}, {self.y_max*1e3:.1f}] mm, X [{self.x_min*1e3:.1f}, {self.x_max*1e3:.1f}] mm")
        print(f"Grid size: {self.ny} x {self.nx}")
        
    def calculate_delays(self, pixel_y, pixel_x):
        """
        Calculate time delays from each element to a pixel
        
        Parameters:
        -----------
        pixel_y : float
            Y-coordinate of pixel (lateral)
        pixel_x : float
            X-coordinate of pixel (axial/depth)
            
        Returns:
        --------
        delays : ndarray
            Time delays for each element
        """
        # Distance from each element to the pixel
        distances = np.sqrt((self.element_y_positions - pixel_y)**2 + 
                          (self.element_x_positions - pixel_x)**2)
        
        # Time delays
        delays = distances / self.c
        
        return delays
    
    def reconstruct_ubp(self):
        """
        Universal Back Projection (UBP) / Delay and Sum (DAS)
        
        Returns:
        --------
        image : ndarray
            Reconstructed image
        """
        print("\n" + "="*60)
        print("Reconstructing with Universal Back Projection (UBP)...")
        print("="*60)
        
        image = np.zeros((self.nx, self.ny))
        
        for i in range(self.nx):
            if i % 20 == 0:
                print(f"Processing row {i}/{self.nx}")
            
            for j in range(self.ny):
                pixel_x = self.X[i, j]
                pixel_y = self.Y[i, j]
                
                # Calculate delays for this pixel
                delays = self.calculate_delays(pixel_y, pixel_x)
                
                # Convert to sample indices
                sample_indices = (delays / self.dt).astype(int)
                
                # Sum the contributions from all elements
                pixel_value = 0
                for elem in range(self.n_elements):
                    idx = sample_indices[elem]
                    if 0 <= idx < self.n_samples:
                        pixel_value += self.sensor_data_analytic[elem, idx]
                
                image[i, j] = pixel_value
        
        print("UBP reconstruction complete!")
        return image
    
    def reconstruct_dmas(self):
        """
        Delay Multiply and Sum (DMAS)
        
        Returns:
        --------
        image : ndarray
            Reconstructed image
        """
        print("\n" + "="*60)
        print("Reconstructing with Delay Multiply and Sum (DMAS)...")
        print("="*60)
        
        image = np.zeros((self.nx, self.ny))
        
        for i in range(self.nx):
            if i % 20 == 0:
                print(f"Processing row {i}/{self.nx}")
            
            for j in range(self.ny):
                pixel_x = self.X[i, j]
                pixel_y = self.Y[i, j]
                
                # Calculate delays for this pixel
                delays = self.calculate_delays(pixel_y, pixel_x)
                
                # Convert to sample indices
                sample_indices = (delays / self.dt).astype(int)
                
                # Get delayed signals
                delayed_signals = np.zeros(self.n_elements)
                for elem in range(self.n_elements):
                    idx = sample_indices[elem]
                    if 0 <= idx < self.n_samples:
                        delayed_signals[elem] = self.sensor_data_analytic[elem, idx]
                
                # DMAS: multiply pairs and sum
                pixel_value = 0
                count = 0
                for m in range(self.n_elements):
                    for n in range(m+1, self.n_elements):
                        pixel_value += delayed_signals[m] * delayed_signals[n]
                        count += 1
                
                if count > 0:
                    image[i, j] = np.sqrt(pixel_value / count)  # Normalize and take sqrt
        
        print("DMAS reconstruction complete!")
        return image
    
    def reconstruct_slsc(self, M=10):
        """
        Short Lag Spatial Coherence (SLSC)
        
        Parameters:
        -----------
        M : int
            Maximum lag for coherence calculation
            
        Returns:
        --------
        image : ndarray
            Reconstructed image
        """
        print("\n" + "="*60)
        print(f"Reconstructing with Short Lag Spatial Coherence (SLSC, M={M})...")
        print("="*60)
        
        image = np.zeros((self.nx, self.ny))
        
        for i in range(self.nx):
            if i % 20 == 0:
                print(f"Processing row {i}/{self.nx}")
            
            for j in range(self.ny):
                pixel_x = self.X[i, j]
                pixel_y = self.Y[i, j]
                
                # Calculate delays for this pixel
                delays = self.calculate_delays(pixel_y, pixel_x)
                
                # Convert to sample indices
                sample_indices = (delays / self.dt).astype(int)
                
                # Get delayed signals
                delayed_signals = np.zeros(self.n_elements)
                for elem in range(self.n_elements):
                    idx = sample_indices[elem]
                    if 0 <= idx < self.n_samples:
                        delayed_signals[elem] = self.sensor_data_analytic[elem, idx]
                
                # Calculate spatial coherence for short lags
                R = 0
                count = 0
                for m in range(1, min(M+1, self.n_elements)):
                    for n in range(self.n_elements - m):
                        # Normalized correlation
                        if delayed_signals[n] != 0 or delayed_signals[n+m] != 0:
                            corr = (delayed_signals[n] * delayed_signals[n+m]) / \
                                   (np.sqrt(delayed_signals[n]**2 + 1e-10) * 
                                    np.sqrt(delayed_signals[n+m]**2 + 1e-10))
                            R += corr
                            count += 1
                
                if count > 0:
                    image[i, j] = R / count
        
        print("SLSC reconstruction complete!")
        return image
    
    def reconstruct_mv(self, subarray_size=32):
        """
        Minimum Variance (MV) beamforming
        
        Parameters:
        -----------
        subarray_size : int
            Size of subarrays for MV calculation
            
        Returns:
        --------
        image : ndarray
            Reconstructed image
        """
        print("\n" + "="*60)
        print(f"Reconstructing with Minimum Variance (MV, subarray={subarray_size})...")
        print("="*60)
        
        image = np.zeros((self.nx, self.ny))
        
        for i in range(self.nx):
            if i % 20 == 0:
                print(f"Processing row {i}/{self.nx}")
            
            for j in range(self.ny):
                pixel_x = self.X[i, j]
                pixel_y = self.Y[i, j]
                
                # Calculate delays for this pixel
                delays = self.calculate_delays(pixel_y, pixel_x)
                
                # Convert to sample indices
                sample_indices = (delays / self.dt).astype(int)
                
                # Use subarray for MV
                n_subarrays = self.n_elements - subarray_size + 1
                mv_sum = 0
                
                for sub_start in range(n_subarrays):
                    sub_end = sub_start + subarray_size
                    
                    # Get delayed signals for subarray
                    y = np.zeros(subarray_size, dtype=complex)
                    for k, elem in enumerate(range(sub_start, sub_end)):
                        idx = sample_indices[elem]
                        if 0 <= idx < self.n_samples:
                            y[k] = self.sensor_data_analytic[elem, idx]
                    
                    # Covariance matrix
                    R = np.outer(y, np.conj(y)) + 1e-6 * np.eye(subarray_size)
                    
                    # Steering vector (uniform)
                    a = np.ones(subarray_size) / np.sqrt(subarray_size)
                    
                    # MV weight
                    try:
                        R_inv = np.linalg.inv(R)
                        w = R_inv @ a / (a.T @ R_inv @ a + 1e-10)
                        
                        # Beamformer output
                        mv_sum += np.abs(w.T @ y)
                    except:
                        mv_sum += np.abs(np.sum(y)) / subarray_size
                
                image[i, j] = mv_sum / n_subarrays
        
        print("MV reconstruction complete!")
        return image
    
    def calculate_sbr(self, image, signal_positions, background_positions):
        """
        Calculate Signal-to-Background Ratio (SBR)
        
        Parameters:
        -----------
        image : ndarray
            Reconstructed image
        signal_positions : list of tuples
            List of (y, x) positions for signal regions (in mm)
        background_positions : list of tuples
            List of (y, x) positions for background regions (in mm)
            
        Returns:
        --------
        sbr_db : float
            SBR in dB
        signal_mean : float
            Mean signal intensity
        background_mean : float
            Mean background intensity
        """
        # Define region size (in pixels)
        region_size_y = 5  # pixels
        region_size_x = 5  # pixels
        
        signal_values = []
        background_values = []
        
        # Extract signal regions
        for y_mm, x_mm in signal_positions:
            # Convert mm to pixel indices
            j = int((y_mm * 1e-3 - self.y_min) / (self.y_max - self.y_min) * self.ny)
            i = int((x_mm * 1e-3 - self.x_min) / (self.x_max - self.x_min) * self.nx)
            
            # Extract region
            i_start = max(0, i - region_size_x // 2)
            i_end = min(self.nx, i + region_size_x // 2 + 1)
            j_start = max(0, j - region_size_y // 2)
            j_end = min(self.ny, j + region_size_y // 2 + 1)
            
            region = image[i_start:i_end, j_start:j_end]
            signal_values.extend(region.flatten())
        
        # Extract background regions
        for y_mm, x_mm in background_positions:
            # Convert mm to pixel indices
            j = int((y_mm * 1e-3 - self.y_min) / (self.y_max - self.y_min) * self.ny)
            i = int((x_mm * 1e-3 - self.x_min) / (self.x_max - self.x_min) * self.nx)
            
            # Extract region
            i_start = max(0, i - region_size_x // 2)
            i_end = min(self.nx, i + region_size_x // 2 + 1)
            j_start = max(0, j - region_size_y // 2)
            j_end = min(self.ny, j + region_size_y // 2 + 1)
            
            region = image[i_start:i_end, j_start:j_end]
            background_values.extend(region.flatten())
        
        # Calculate means
        signal_mean = np.mean(signal_values)
        background_mean = np.mean(background_values)
        
        # Calculate SBR in dB
        sbr = signal_mean / (background_mean + 1e-10)
        sbr_db = 20 * np.log10(sbr + 1e-10)
        
        return sbr_db, signal_mean, background_mean
    
    def plot_reconstruction(self, image, title, filename):
        """
        Plot and save reconstructed image
        
        Parameters:
        -----------
        image : ndarray
            Reconstructed image
        title : str
            Plot title
        filename : str
            Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize image
        image_norm = image / np.max(image)
        image_db = 20 * np.log10(image_norm + 1e-10)
        image_db = np.clip(image_db, -60, 0)
        
        # Plot
        im = ax.imshow(image_db, aspect='auto', cmap='hot', 
                       extent=[self.y_min*1e3, self.y_max*1e3, 
                              self.x_max*1e3, self.x_min*1e3],
                       vmin=-60, vmax=0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity (dB)', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('y-position (mm)', fontsize=12)
        ax.set_ylabel('x-position (mm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add transducer position indicator
        ax.plot([self.y_min*1e3, self.y_max*1e3], [0, 0], 'b-', linewidth=3, 
                label='Transducer Array')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")


def main():
    """
    Main function to run all reconstructions and calculate SBR
    """
    print("\n" + "="*60)
    print("PHOTOACOUSTIC IMAGING RECONSTRUCTION")
    print("="*60)
    
    # Initialize reconstruction
    reconstructor = PhotoacousticReconstruction(
        '/mnt/user-data/uploads/Q1__3_SensorData_5dots_diag_NoNoise__1_.mat'
    )
    
    # Define target positions (5 sources placed diagonally)
    # Based on the image: sources are at diagonal positions
    # Estimating positions from the reference image
    signal_positions = [
        (10, 5),   # Target 1 (y=10mm, x=5mm)
        (17, 8),   # Target 2
        (25, 10),  # Target 3 (center, aligned with center element)
        (32, 13),  # Target 4
        (40, 15),  # Target 5
    ]
    
    # Define background positions (away from targets)
    background_positions = [
        (5, 5),    # Background 1
        (15, 15),  # Background 2
        (25, 3),   # Background 3
        (35, 8),   # Background 4
        (45, 12),  # Background 5
    ]
    
    # Dictionary to store results
    results = {}
    
    # 1. Universal Back Projection (UBP)
    image_ubp = reconstructor.reconstruct_ubp()
    sbr_ubp, sig_ubp, bg_ubp = reconstructor.calculate_sbr(
        image_ubp, signal_positions, background_positions
    )
    results['UBP'] = {
        'image': image_ubp,
        'sbr': sbr_ubp,
        'signal_mean': sig_ubp,
        'background_mean': bg_ubp
    }
    reconstructor.plot_reconstruction(
        image_ubp, 
        'Universal Back Projection (UBP)', 
        '/home/claude/ubp_reconstruction.png'
    )
    
    # 2. Delay Multiply and Sum (DMAS)
    image_dmas = reconstructor.reconstruct_dmas()
    sbr_dmas, sig_dmas, bg_dmas = reconstructor.calculate_sbr(
        image_dmas, signal_positions, background_positions
    )
    results['DMAS'] = {
        'image': image_dmas,
        'sbr': sbr_dmas,
        'signal_mean': sig_dmas,
        'background_mean': bg_dmas
    }
    reconstructor.plot_reconstruction(
        image_dmas, 
        'Delay Multiply and Sum (DMAS)', 
        '/home/claude/dmas_reconstruction.png'
    )
    
    # 3. Short Lag Spatial Coherence (SLSC)
    image_slsc = reconstructor.reconstruct_slsc(M=10)
    sbr_slsc, sig_slsc, bg_slsc = reconstructor.calculate_sbr(
        image_slsc, signal_positions, background_positions
    )
    results['SLSC'] = {
        'image': image_slsc,
        'sbr': sbr_slsc,
        'signal_mean': sig_slsc,
        'background_mean': bg_slsc
    }
    reconstructor.plot_reconstruction(
        image_slsc, 
        'Short Lag Spatial Coherence (SLSC)', 
        '/home/claude/slsc_reconstruction.png'
    )
    
    # 4. Minimum Variance (MV)
    image_mv = reconstructor.reconstruct_mv(subarray_size=32)
    sbr_mv, sig_mv, bg_mv = reconstructor.calculate_sbr(
        image_mv, signal_positions, background_positions
    )
    results['MV'] = {
        'image': image_mv,
        'sbr': sbr_mv,
        'signal_mean': sig_mv,
        'background_mean': bg_mv
    }
    reconstructor.plot_reconstruction(
        image_mv, 
        'Minimum Variance (MV)', 
        '/home/claude/mv_reconstruction.png'
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    algorithms = ['UBP', 'DMAS', 'SLSC', 'MV']
    
    for idx, (ax, alg) in enumerate(zip(axes.flat, algorithms)):
        image = results[alg]['image']
        image_norm = image / np.max(image)
        image_db = 20 * np.log10(image_norm + 1e-10)
        image_db = np.clip(image_db, -60, 0)
        
        im = ax.imshow(image_db, aspect='auto', cmap='hot',
                      extent=[reconstructor.y_min*1e3, reconstructor.y_max*1e3,
                             reconstructor.x_max*1e3, reconstructor.x_min*1e3],
                      vmin=-60, vmax=0)
        
        ax.set_xlabel('y-position (mm)', fontsize=11)
        ax.set_ylabel('x-position (mm)', fontsize=11)
        ax.set_title(f'{alg} (SBR: {results[alg]["sbr"]:.2f} dB)', 
                    fontsize=13, fontweight='bold')
        
        # Add transducer
        ax.plot([reconstructor.y_min*1e3, reconstructor.y_max*1e3], 
               [0, 0], 'b-', linewidth=2)
        
        plt.colorbar(im, ax=ax, label='Intensity (dB)')
    
    plt.tight_layout()
    plt.savefig('/home/claude/all_reconstructions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("SIGNAL-TO-BACKGROUND RATIO (SBR) RESULTS")
    print("="*60)
    
    # Print SBR results
    print(f"\n{'Algorithm':<15} {'SBR (dB)':<12} {'Signal Mean':<15} {'Background Mean':<15}")
    print("-" * 60)
    for alg in algorithms:
        print(f"{alg:<15} {results[alg]['sbr']:>10.2f}  {results[alg]['signal_mean']:>13.2e}  "
              f"{results[alg]['background_mean']:>13.2e}")
    
    # SBR Variability Analysis
    print("\n" + "="*60)
    print("SBR VARIABILITY ANALYSIS")
    print("="*60)
    
    # Calculate SBR for each individual signal position across all algorithms
    print("\nPer-Signal SBR Analysis (all algorithms):")
    print(f"\n{'Position':<20} {'UBP (dB)':<12} {'DMAS (dB)':<12} {'SLSC (dB)':<12} {'MV (dB)':<12}")
    print("-" * 80)
    
    for idx, (y_mm, x_mm) in enumerate(signal_positions):
        sbr_values = []
        for alg in algorithms:
            sbr_individual, _, _ = reconstructor.calculate_sbr(
                results[alg]['image'],
                [(y_mm, x_mm)],  # Single signal position
                background_positions
            )
            sbr_values.append(sbr_individual)
        
        print(f"Signal {idx+1} ({y_mm},{x_mm})mm  {sbr_values[0]:>10.2f}  "
              f"{sbr_values[1]:>10.2f}  {sbr_values[2]:>10.2f}  {sbr_values[3]:>10.2f}")
    
    # Statistical summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    
    sbr_array = np.array([[results[alg]['sbr'] for alg in algorithms]])
    print(f"\nMean SBR across all algorithms: {np.mean(sbr_array):.2f} dB")
    print(f"Std Dev of SBR: {np.std(sbr_array):.2f} dB")
    print(f"Min SBR: {np.min(sbr_array):.2f} dB ({algorithms[np.argmin(sbr_array)]})")
    print(f"Max SBR: {np.max(sbr_array):.2f} dB ({algorithms[np.argmax(sbr_array)]})")
    
    print("\n" + "="*60)
    print("RECONSTRUCTION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - ubp_reconstruction.png")
    print("  - dmas_reconstruction.png")
    print("  - slsc_reconstruction.png")
    print("  - mv_reconstruction.png")
    print("  - all_reconstructions_comparison.png")
    print("="*60)


if __name__ == "__main__":
    main()
