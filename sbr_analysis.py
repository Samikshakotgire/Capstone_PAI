"""
SBR Analysis and Comparison Visualization
Generates detailed SBR comparison charts
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Data from reconstruction results
algorithms = ['UBP', 'DMAS', 'SLSC', 'MV']
overall_sbr = [8.14, 8.28, 0.13, -5.61]

# Per-signal SBR data
signal_positions = ['Signal 1\n(10,5)mm', 'Signal 2\n(17,8)mm', 'Signal 3\n(25,10)mm', 
                   'Signal 4\n(32,13)mm', 'Signal 5\n(40,15)mm']

per_signal_sbr = {
    'UBP': [1.50, 6.08, 14.98, 7.90, 3.30],
    'DMAS': [1.51, 6.18, 15.17, 8.04, 3.39],
    'SLSC': [0.12, 0.13, 0.13, 0.13, 0.13],
    'MV': [-2.96, -8.35, -15.54, -8.09, -0.31]
}

# Create comprehensive comparison figure
fig = plt.figure(figsize=(18, 10))

# 1. Overall SBR Comparison (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax1.bar(algorithms, overall_sbr, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_ylabel('SBR (dB)', fontsize=12, fontweight='bold')
ax1.set_title('Overall SBR Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

# 2. Per-Signal SBR Heatmap
ax2 = plt.subplot(2, 3, 2)
sbr_matrix = np.array([per_signal_sbr[alg] for alg in algorithms])
im = ax2.imshow(sbr_matrix, cmap='RdYlGn', aspect='auto', vmin=-16, vmax=16)
ax2.set_xticks(range(len(signal_positions)))
ax2.set_xticklabels(signal_positions, fontsize=9)
ax2.set_yticks(range(len(algorithms)))
ax2.set_yticklabels(algorithms, fontsize=11, fontweight='bold')
ax2.set_title('SBR Heatmap (dB)', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(algorithms)):
    for j in range(len(signal_positions)):
        text = ax2.text(j, i, f'{sbr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

plt.colorbar(im, ax=ax2, label='SBR (dB)')

# 3. Per-Signal Line Plot
ax3 = plt.subplot(2, 3, 3)
x_pos = np.arange(len(signal_positions))
for idx, alg in enumerate(algorithms):
    ax3.plot(x_pos, per_signal_sbr[alg], marker='o', linewidth=2, 
            markersize=8, label=alg, color=colors[idx], alpha=0.8)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['S1', 'S2', 'S3', 'S4', 'S5'], fontsize=10)
ax3.set_xlabel('Signal Position', fontsize=12, fontweight='bold')
ax3.set_ylabel('SBR (dB)', fontsize=12, fontweight='bold')
ax3.set_title('SBR vs Signal Position', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# 4. Statistical Box Plot
ax4 = plt.subplot(2, 3, 4)
box_data = [per_signal_sbr[alg] for alg in algorithms]
bp = ax4.boxplot(box_data, labels=algorithms, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_ylabel('SBR (dB)', fontsize=12, fontweight='bold')
ax4.set_title('SBR Distribution by Algorithm', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# 5. SBR Variability (Std Dev)
ax5 = plt.subplot(2, 3, 5)
std_devs = [np.std(per_signal_sbr[alg]) for alg in algorithms]
means = [np.mean(per_signal_sbr[alg]) for alg in algorithms]
bars = ax5.bar(algorithms, std_devs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Standard Deviation (dB)', fontsize=12, fontweight='bold')
ax5.set_title('SBR Variability', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Performance Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create table data
table_data = []
for alg in algorithms:
    mean_sbr = np.mean(per_signal_sbr[alg])
    std_sbr = np.std(per_signal_sbr[alg])
    min_sbr = np.min(per_signal_sbr[alg])
    max_sbr = np.max(per_signal_sbr[alg])
    table_data.append([alg, f'{mean_sbr:.2f}', f'{std_sbr:.2f}', 
                      f'{min_sbr:.2f}', f'{max_sbr:.2f}'])

table = ax6.table(cellText=table_data,
                 colLabels=['Algorithm', 'Mean (dB)', 'Std (dB)', 'Min (dB)', 'Max (dB)'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0.2, 1, 0.6])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i in range(1, 5):
    for j in range(5):
        table[(i, j)].set_facecolor(colors[i-1])
        table[(i, j)].set_alpha(0.3)
        
ax6.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sbr_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: sbr_detailed_analysis.png")

# Create signal position comparison
fig2, axes = plt.subplots(1, 5, figsize=(20, 4))
for idx, (ax, signal) in enumerate(zip(axes, signal_positions)):
    sbr_values = [per_signal_sbr[alg][idx] for alg in algorithms]
    bars = ax.bar(algorithms, sbr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(signal, fontsize=11, fontweight='bold')
    ax.set_ylabel('SBR (dB)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=8, fontweight='bold', rotation=0)
    
    # Rotate x labels
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('SBR Comparison Across All Signal Positions', 
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sbr_per_signal_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: sbr_per_signal_comparison.png")

print("\n" + "="*60)
print("SBR Analysis Complete!")
print("="*60)
print("\nGenerated visualizations:")
print("  1. sbr_detailed_analysis.png - Comprehensive 6-panel analysis")
print("  2. sbr_per_signal_comparison.png - Per-signal detailed comparison")
print("="*60)
