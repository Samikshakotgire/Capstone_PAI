import numpy as np
import scipy.io as sio

def delay_and_sum(mat_file_path):
    """
    Algorithm 1: Delay-and-Sum (DAS) for Photoacoustic Image Reconstruction
    
    Inputs (from .mat file):
        S       : sensor data [N_sensors x N_timesamples]
        N       : number of sensors
        roi_n, roi_m : ROI size (n x m pixels)
        xp, yp  : pixel positions (meshgrid arrays)
        xs, ys  : sensor positions
        c       : speed of sound
        fs      : sampling frequency
    
    Output:
        I_DAS   : reconstructed image [n x m]
    """

    # ── Load .mat file ────────────────────────────────────────────────────
    data = sio.loadmat(mat_file_path)

    S   = data['S']           # sensor data
    xs  = data['xs'].flatten()
    ys  = data['ys'].flatten()
    xp  = data['xp']          # pixel x-positions (2D grid)
    yp  = data['yp']          # pixel y-positions (2D grid)
    c   = float(data['c'])    # speed of sound
    fs  = float(data['fs'])   # sampling frequency
    N   = int(data['N'])      # number of sensors

    n, m = xp.shape           # ROI size

    # ── Algorithm 1 ──────────────────────────────────────────────────────
    I_DAS = np.zeros((n, m))  # Line 1: I_DAS = 0

    for i in range(N):                          # Line 2: for i ← 1, N do
        for row in range(n):                    # Line 3: for j ← 1, n×m do
            for col in range(m):

                # Line 4: d_ij = sqrt((xpj - xsi)^2 + (ypj - ysi)^2)
                d_ij = np.sqrt((xp[row, col] - xs[i])**2 +
                               (yp[row, col] - ys[i])**2)

                # Line 5: delay(xpj, ypj, i) = (d_ij / c) * fs
                delay = (d_ij / c) * fs
                delay_idx = int(round(delay))

                # Line 6: I_DAS(j) = I_DAS(j) + S(i, delay(xpj, ypj, i))
                if 0 <= delay_idx < S.shape[1]:
                    I_DAS[row, col] += S[i, delay_idx]

    # Line 9: return I_DAS
    return I_DAS


# ── Vectorized (fast) version ─────────────────────────────────────────────
def delay_and_sum_fast(mat_file_path):
    """Same algorithm but fully vectorized with numpy — much faster."""

    data = sio.loadmat(mat_file_path)
    S   = data['S']
    xs  = data['xs'].flatten()
    ys  = data['ys'].flatten()
    xp  = data['xp']
    yp  = data['yp']
    c   = float(data['c'])
    fs  = float(data['fs'])
    N   = int(data['N'])

    n, m   = xp.shape
    I_DAS  = np.zeros((n, m))

    xp_flat = xp.flatten()   # (n*m,)
    yp_flat = yp.flatten()

    for i in range(N):
        # d_ij for all pixels at once  →  shape (n*m,)
        d_ij = np.sqrt((xp_flat - xs[i])**2 + (yp_flat - ys[i])**2)

        # delay index for all pixels
        delay_idx = np.round((d_ij / c) * fs).astype(int)

        # mask valid indices
        valid = (delay_idx >= 0) & (delay_idx < S.shape[1])

        # accumulate
        contrib = np.zeros(n * m)
        contrib[valid] = S[i, delay_idx[valid]]
        I_DAS += contrib.reshape(n, m)

    return I_DAS


# ── Run & plot ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mat_path = "Q1 _3_SensorData_5dots_diag_NoNoise (1).mat"   # ← change to your file path

    print("Running DAS reconstruction...")
    I = delay_and_sum_fast(mat_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(I, cmap='hot', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.title('DAS Reconstructed PA Image')
    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')
    plt.tight_layout()
    plt.savefig('DAS_result.png', dpi=150)
    plt.show()
    print("Done. Image saved as DAS_result.png")