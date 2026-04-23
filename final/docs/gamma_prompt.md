# Gamma Prompt for PAI Presentation

Create a professional PowerPoint presentation on Photoacoustic Imaging (PAI) reconstruction and image quality analysis. Use a clear biomedical style, minimal clutter, and strong section headers. The deck should explain the progression from baseline reconstruction to DMAS improvement, sparse sensing, noisy data, and final image quality metrics.

## Use these figures from the `final/images` folder
- ubp_reconstruction.png
- DMAS_final_fixed.png
- single_sensor.png
- DMAS_noisy_final.png
- DMAS_9_metrics_complete.png
- sbr_detailed_analysis.png
- all_reconstructions_comparison.png

## Use these scripts from the `final/scripts` folder as the code reference
- photoacoustic_reconstruction.py
- DMAS_final.py
- freshcode.py
- for_noisy_data.py
- 4_04.py
- sbr_analysis.py

## Create 8 to 10 slides with this flow
1. Title slide: Photoacoustic Imaging Reconstruction and Quality Assessment
2. Problem statement: why target detection in PAI is challenging
3. Baseline reconstruction: UBP result
4. Improved reconstruction: DMAS clean result with bounding boxes and SBR
5. ROI and thresholding concept: signal ROI, background ROI, and background thresholding
6. Sparse sensing case: `single_sensor.png` and sensor degradation
7. Noisy data case: `DMAS_noisy_final.png` and robustness under noise
8. Image quality metrics: SBR, SNR, CNR, gCNR, FWHM, contrast, speckle SNR, edge sharpness
9. SBR comparison summary across algorithms
10. Conclusion: summarize the best method and main takeaway

## Important notes
- Explain that UBP is the baseline.
- Explain that DMAS improves contrast and target visibility.
- Explain that bounding boxes define signal and background ROIs.
- Explain that thresholding must be used carefully.
- Explain that noisy and sparse-sensor cases test robustness.
- Explain that image quality should not rely on SBR alone.
- Add short speaker notes under each slide, 2 to 4 lines each.
- Use the exact filenames where possible.
- Keep equations simple and readable.
- End with the takeaway that ROI selection, thresholding, and metric choice are as important as the reconstruction algorithm.
