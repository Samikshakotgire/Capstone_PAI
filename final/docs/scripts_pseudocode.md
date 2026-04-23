# Pseudocode Reference for Final Scripts

Use this as the code explanation handout for Gamma or PDF export.

## 1) `photoacoustic_reconstruction.py`
Purpose: baseline reconstruction comparison with UBP, DMAS, SLSC, and MV.

Pseudocode:
- Load sensor data from the `.mat` file.
- Set acquisition parameters, sensor positions, and reconstruction grid.
- Apply Hilbert transform to obtain analytic/envelope signal.
- For each sensor:
  - Compute distance from sensor to every pixel.
  - Convert distance to delayed sample index.
  - Fill delayed matrix with valid samples.
- Reconstruct images:
  - UBP: sum delayed signals.
  - DMAS: pairwise multiply-and-sum beamforming.
  - SLSC: short-lag spatial coherence estimate.
  - MV: coherence-factor weighted approximation.
- Compute SBR values for each method.
- Plot and save reconstruction images.

## 2) `DMAS_final.py`
Purpose: clean DMAS reconstruction with manual ROIs and thresholding.

Pseudocode:
- Load sensor data and define linear array geometry.
- Apply bandpass filter and Hilbert envelope.
- Build delayed matrix over reconstruction grid.
- Compute DMAS image using pairwise delayed-signal products.
- Normalize image and threshold low-intensity pixels.
- Define five target dots and background ROIs.
- For each target:
  - Extract signal ROI.
  - Extract background ROI.
  - Compute SBR.
  - Draw signal box and background box.
- Save DMAS figure.

## 3) `freshcode.py`
Purpose: sparse-sensor / degradation analysis with one sensor marker and one target at a time.

Pseudocode:
- Load sensor data and initialize PACT reconstruction parameters.
- Define target positions and a fixed background position.
- For each target:
  - Find the closest sensor to the target.
  - For each degradation factor (1/2, 1/4, 1/6 sensors):
    - Randomly keep only a fraction of sensors.
    - Zero the remaining channels.
    - Reconstruct DMAS image.
    - Normalize image.
    - Extract target ROI and background ROI.
    - Compute SBR.
    - Plot image with:
      - red current-sensor box
      - cyan target ROI
      - green background ROI
- Display and print SBR result.

## 4) `for_noisy_data.py`
Purpose: noisy-data DMAS reconstruction and ROI-based evaluation.

Pseudocode:
- Load noisy `.mat` sensor dataset.
- Apply bandpass filter and Hilbert envelope.
- Build delayed reconstruction matrix.
- Compute DMAS image.
- Normalize image.
- Apply lower threshold suitable for noisy data.
- Detect the five brightest targets automatically.
- For each detected target:
  - Place a signal ROI around the target.
  - Place a background ROI away from the target.
  - Compute SBR from the normalized image.
  - Plot the thresholded log image with ROI boxes.
- Save noisy DMAS result.

## 5) `4_04.py`
Purpose: compute multiple image quality metrics for DMAS.

Pseudocode:
- Load sensor data and reconstruct DMAS image.
- Normalize the image.
- Auto-detect the five brightest targets.
- For each target:
  - Extract signal ROI and background ROI.
  - Compute SBR.
  - Compute SNR.
  - Compute CNR.
  - Compute gCNR.
  - Estimate lateral and axial FWHM.
  - Compute contrast.
  - Compute speckle SNR.
  - Compute edge sharpness.
- Print a table of all metrics.
- Plot the image with target-specific metric labels.
- Save the final 9-metric figure.

## 6) `sbr_analysis.py`
Purpose: compare SBR values across algorithms and targets.

Pseudocode:
- Define algorithm names and stored SBR values.
- Define signal positions and SBR results per algorithm.
- Create a multi-panel figure:
  - overall SBR bar chart
  - per-signal heatmap
  - SBR line plot by target position
  - box plot for distribution
  - SBR standard deviation plot
  - summary table
- Save the comparison figures.

## Short Presentation Summary
- `photoacoustic_reconstruction.py`: baseline reconstruction methods.
- `DMAS_final.py`: clean DMAS with ROI boxes.
- `freshcode.py`: sparse sensor degradation case.
- `for_noisy_data.py`: noisy-data robustness case.
- `4_04.py`: image quality metrics beyond SBR.
- `sbr_analysis.py`: final SBR comparison summary.
