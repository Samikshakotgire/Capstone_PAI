"""
Paper-style DMAS reconstruction for .mat sensor data.

Implements the standard DMAS formulation used in ultrasound/photoacoustic literature:
    y_DMAS(p) = sum_{i=1}^{N-1} sum_{j=i+1}^{N} sign(s_i(p)s_j(p)) * sqrt(|s_i(p)s_j(p)|)

where s_i(p) is the delayed envelope sample for sensor i at image pixel p.

Usage examples:
    python codes/dmas_paper_mat.py --mat "Q1 _3_SensorData_5dots_diag_NoNoise (1).mat"
    python codes/dmas_paper_mat.py --mat your_data_file.mat --key sensor_data
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DMAS reconstruction from .mat data")
    parser.add_argument("--mat", type=str, required=True, help="Path to .mat file")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Variable name inside .mat (optional, auto-detected if omitted)",
    )
    parser.add_argument("--fs", type=float, default=40e6, help="Sampling frequency (Hz)")
    parser.add_argument("--c", type=float, default=1500.0, help="Speed of sound (m/s)")
    parser.add_argument("--pitch", type=float, default=0.3e-3, help="Sensor pitch (m)")
    parser.add_argument("--dx", type=float, default=0.1e-3, help="Pixel spacing (m)")
    parser.add_argument("--x_min", type=float, default=-19.05e-3, help="Min lateral (m)")
    parser.add_argument("--x_max", type=float, default=19.05e-3, help="Max lateral (m)")
    parser.add_argument("--z_min", type=float, default=0.5e-3, help="Min depth (m)")
    parser.add_argument("--z_max", type=float, default=20e-3, help="Max depth (m)")
    parser.add_argument("--bp_low", type=float, default=1e6, help="Bandpass low cut (Hz)")
    parser.add_argument("--bp_high", type=float, default=10e6, help="Bandpass high cut (Hz)")
    parser.add_argument("--bp_order", type=int, default=4, help="Butterworth order")
    parser.add_argument(
        "--out",
        type=str,
        default="dmas_paper_reconstruction.png",
        help="Output image file",
    )
    return parser.parse_args()


def bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    nyq = fs / 2.0
    wn = [low / nyq, high / nyq]
    b, a = butter(order, wn, btype="band")
    return filtfilt(b, a, sig, axis=1)


def load_sensor_data(mat_path: Path, key: str | None = None) -> np.ndarray:
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    data = sio.loadmat(mat_path)

    if key is not None:
        if key not in data:
            raise KeyError(f"Key '{key}' not found. Available keys: {list(data.keys())}")
        arr = data[key]
    else:
        candidates: list[tuple[str, np.ndarray]] = []
        for k, v in data.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2:
                candidates.append((k, v))

        if not candidates:
            raise ValueError("No 2D array found in MAT file. Please pass --key explicitly.")

        # Prefer array that looks like [num_sensors, num_samples] with sensors <= samples.
        candidates.sort(key=lambda kv: kv[1].size, reverse=True)
        chosen = None
        for k, v in candidates:
            if v.shape[0] <= v.shape[1]:
                chosen = (k, v)
                break
        if chosen is None:
            chosen = candidates[0]

        key, arr = chosen
        print(f"Auto-selected key: {key} with shape {arr.shape}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sensor matrix, got shape {arr.shape}")

    # Ensure orientation is [num_sensors, num_samples]
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
        print(f"Transposed sensor data to shape {arr.shape} (sensors x samples)")

    return arr


def main() -> None:
    args = parse_args()

    mat_path = Path(args.mat)
    sensor_data = load_sensor_data(mat_path, key=args.key)

    num_sensors, num_samples = sensor_data.shape
    dt = 1.0 / args.fs

    print(f"Loaded sensor data: {sensor_data.shape}")
    print("Bandpass + envelope detection...")
    filtered = bandpass(sensor_data, fs=args.fs, low=args.bp_low, high=args.bp_high, order=args.bp_order)
    envelope = np.abs(hilbert(filtered, axis=1))

    x_sensor = (np.arange(num_sensors) - (num_sensors - 1) / 2.0) * args.pitch
    x_grid = np.arange(args.x_min, args.x_max, args.dx)
    z_grid = np.arange(args.z_min, args.z_max, args.dx)
    X, Z = np.meshgrid(x_grid, z_grid)
    nz, nx = Z.shape

    print("Computing delayed data cube...")
    delayed = np.zeros((num_sensors, nz, nx), dtype=np.float32)
    for i in range(num_sensors):
        dist = np.sqrt((X - x_sensor[i]) ** 2 + Z**2)
        idx = np.round(dist / (args.c * dt)).astype(np.int32)
        valid = (idx >= 0) & (idx < num_samples)

        tmp = np.zeros((nz, nx), dtype=np.float32)
        tmp[valid] = envelope[i, idx[valid]]
        delayed[i] = tmp

    print("Running paper-style DMAS pairwise summation...")
    img_dmas = np.zeros((nz, nx), dtype=np.float64)
    for i in range(num_sensors - 1):
        si = delayed[i].astype(np.float64)
        for j in range(i + 1, num_sensors):
            prod = si * delayed[j]
            img_dmas += np.sign(prod) * np.sqrt(np.abs(prod))

    img_abs = np.abs(img_dmas)
    img_norm = img_abs / (np.max(img_abs) + 1e-12)
    img_db = 20.0 * np.log10(np.clip(img_norm, 1e-6, 1.0))

    print(f"Saving figure: {args.out}")
    plt.figure(figsize=(9, 6))
    im = plt.imshow(
        img_db,
        extent=[x_grid[0] * 1e3, x_grid[-1] * 1e3, z_grid[-1] * 1e3, z_grid[0] * 1e3],
        cmap="hot",
        vmin=-60,
        vmax=0,
        aspect="auto",
    )
    plt.title("DMAS Reconstruction (Paper Formulation)")
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Depth (mm)")
    plt.colorbar(im, label="dB")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
