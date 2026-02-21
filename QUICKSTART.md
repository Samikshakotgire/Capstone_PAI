# Quick Start Guide - Photoacoustic Imaging Reconstruction

## Running the Code

### Step 1: Install Dependencies
```bash
pip install numpy scipy matplotlib
```

### Step 2: Run Main Reconstruction
```bash
python photoacoustic_reconstruction.py
```

This will:
- Load sensor data from the .mat file
- Run all 4 reconstruction algorithms (UBP, DMAS, SLSC, MV)
- Calculate SBR for each algorithm
- Generate individual reconstruction images
- Create comparison plots
- Print detailed SBR analysis

**Expected runtime**: ~2-3 minutes (depending on system)

### Step 3: Run SBR Analysis (Optional)
```bash
python sbr_analysis.py
```

This generates additional detailed SBR comparison visualizations.

## Output Files

After running, you'll find these files:

### Individual Reconstructions
1. `ubp_reconstruction.png` - UBP algorithm result
2. `dmas_reconstruction.png` - DMAS algorithm result  
3. `slsc_reconstruction.png` - SLSC algorithm result
4. `mv_reconstruction.png` - MV algorithm result

### Comparison Plots
5. `all_reconstructions_comparison.png` - 2×2 comparison of all algorithms
6. `sbr_detailed_analysis.png` - Comprehensive SBR analysis (6 panels)
7. `sbr_per_signal_comparison.png` - Per-signal SBR breakdown

### Documentation & Code
8. `README.md` - Comprehensive documentation
9. `QUICKSTART.md` - This file
10. `photoacoustic_reconstruction.py` - Main reconstruction code
11. `sbr_analysis.py` - SBR visualization code

## Understanding the Results

### Image Interpretation
- **Color map**: Hot colormap (dark = low, bright = high intensity)
- **Scale**: dB scale from -60 to 0 dB
- **Y-axis**: Lateral position (0-50 mm, along transducer)
- **X-axis**: Depth (0-20 mm into medium)
- **Blue line**: Transducer array position (at x=0)

### SBR Values
Higher SBR = Better signal quality
- **UBP**: 8.14 dB ✓ Good baseline
- **DMAS**: 8.28 dB ✓ Best performance
- **SLSC**: 0.13 dB - Coherence-based
- **MV**: -5.61 dB - Requires tuning

### Target Positions (5 diagonal sources)
1. Signal 1: (10, 5) mm
2. Signal 2: (17, 8) mm
3. Signal 3: (25, 10) mm - Center, aligned with transducer center
4. Signal 4: (32, 13) mm
5. Signal 5: (40, 15) mm

## Customization

### Modify Target Positions
Edit in `photoacoustic_reconstruction.py`:
```python
signal_positions = [
    (10, 5),   # (y_mm, x_mm)
    (17, 8),
    (25, 10),
    (32, 13),
    (40, 15),
]
```

### Change Algorithm Parameters
```python
# SLSC: Adjust maximum lag
image_slsc = reconstructor.reconstruct_slsc(M=15)  # Try M=5,10,15,20

# MV: Adjust subarray size  
image_mv = reconstructor.reconstruct_mv(subarray_size=24)  # Try 16,24,32,48
```

### Adjust Imaging Grid
```python
# In __init__ method:
self.y_min, self.y_max = 0, 60e-3  # Extend lateral range to 60mm
self.x_min, self.x_max = 0, 25e-3  # Extend depth to 25mm
self.ny = 250  # More lateral pixels
self.nx = 125  # More depth pixels
```

## Troubleshooting

### Issue: "No module named scipy"
**Solution**: Install dependencies
```bash
pip install scipy numpy matplotlib
```

### Issue: Slow execution
**Solution**: Reduce grid resolution
```python
self.ny = 100  # Reduce from 200
self.nx = 50   # Reduce from 100
```

### Issue: Memory error
**Solution**: Process in smaller chunks or reduce array sizes

### Issue: Can't find .mat file
**Solution**: Check file path is correct
```python
PhotoacousticReconstruction('/path/to/your/sensor_data.mat')
```

## Performance Tips

1. **Faster reconstruction**: Reduce grid resolution (ny, nx)
2. **Better image quality**: Increase grid resolution
3. **Better MV performance**: Try different subarray sizes (16, 24, 32, 48)
4. **SLSC tuning**: Experiment with M values (5-20)

## Next Steps

1. Review the README.md for comprehensive documentation
2. Examine individual reconstruction images
3. Compare algorithms using all_reconstructions_comparison.png
4. Analyze SBR results in sbr_detailed_analysis.png
5. Modify parameters and re-run to optimize results

## Support

For questions or issues:
- Check README.md for detailed documentation
- Review the inline code comments
- Verify all dependencies are installed
- Ensure .mat file is in correct location

---
**Last Updated**: February 15, 2026
