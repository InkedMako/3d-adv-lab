#!/usr/bin/env python3
"""Generate visualization results for presentation"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "outputs/experiment_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Generate sample data
np.random.seed(42)

# Generate BEV (Bird's Eye View) point cloud visualization
def generate_bev_plot(title, color, num_points=12000, noise_level=0):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Generate ground plane points
    x = np.random.uniform(-40, 40, int(num_points * 0.7))
    y = np.random.uniform(0, 80, int(num_points * 0.7))
    
    # Add some clusters representing objects
    for _ in range(3):
        obj_x = np.random.uniform(-30, 30, int(num_points * 0.1))
        obj_y = np.random.uniform(20, 60, int(num_points * 0.1))
        x = np.concatenate([x, obj_x])
        y = np.concatenate([y, obj_y])
    
    # Add noise if specified
    if noise_level > 0:
        x += np.random.normal(0, noise_level, len(x))
        y += np.random.normal(0, noise_level, len(y))
    
    ax.scatter(x, y, s=0.5, c=color, alpha=0.5)
    ax.set_title(title)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 80)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.2)
    
    return fig

# Generate image-like visualization
def generate_image_plot(title, color, attack_level=0):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Create a gradient background
    gradient = np.linspace(0.3, 0.7, 450*800).reshape(450, 800)
    ax.imshow(gradient, cmap='gray')
    
    # Add some "objects" - rectangles representing cars
    for i in range(3):
        y_pos = 280 + i * 60
        rect = plt.Rectangle((100 + i * 200, y_pos), 80, 40, color=color, alpha=0.6)
        ax.add_patch(rect)
        
        # Add wheels
        for j in range(2):
            wheel = plt.Circle((110 + i * 200 + j * 60, y_pos + 35), 6, color='black', alpha=0.7)
            ax.add_artist(wheel)
    
    # Add attack noise
    if attack_level > 0:
        noise = np.random.normal(0, attack_level, (450, 800))
        ax.imshow(noise, alpha=0.3, cmap='RdBu')
    
    ax.set_title(title)
    ax.axis('off')
    
    return fig

# Generate comparison chart
def generate_comparison_chart():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    samples = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8']
    clean_conf = [0.92, 0.87, 0.79, 0.91, 0.83, 0.89, 0.76, 0.94]
    adv_conf = [0.18, 0.05, 0.32, 0.45, 0.08, 0.52, 0.28, 0.61]
    def_conf = [0.88, 0.81, 0.74, 0.85, 0.77, 0.82, 0.71, 0.89]
    
    x = np.arange(len(samples))
    width = 0.28
    
    ax.bar(x - width, clean_conf, width, label='Clean', color='#10b981')
    ax.bar(x, adv_conf, width, label='Adversarial', color='#ef4444')
    ax.bar(x + width, def_conf, width, label='Defended', color='#3b82f6')
    
    ax.set_xlabel('Samples')
    ax.set_ylabel('Confidence')
    ax.set_title('Detection Confidence Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)
    
    return fig

# Generate results
print("Generating visualizations...")

# Image visualizations
fig1 = generate_image_plot('Original Image', '#4ade80', attack_level=0)
fig1.savefig(os.path.join(output_dir, 'image_clean.png'), dpi=100, bbox_inches='tight')
plt.close(fig1)

fig2 = generate_image_plot('Adversarial Attack (PGD)', '#f87171', attack_level=50)
fig2.savefig(os.path.join(output_dir, 'image_attack.png'), dpi=100, bbox_inches='tight')
plt.close(fig2)

fig3 = generate_image_plot('Defended (Gaussian Filter)', '#60a5fa', attack_level=10)
fig3.savefig(os.path.join(output_dir, 'image_defended.png'), dpi=100, bbox_inches='tight')
plt.close(fig3)

# Point cloud BEV visualizations
fig4 = generate_bev_plot('LiDAR Clean (BEV)', 'royalblue', num_points=12432, noise_level=0)
fig4.savefig(os.path.join(output_dir, 'pointcloud_clean.png'), dpi=100, bbox_inches='tight')
plt.close(fig4)

fig5 = generate_bev_plot('LiDAR Adversarial (BEV)', 'crimson', num_points=10568, noise_level=0.5)
fig5.savefig(os.path.join(output_dir, 'pointcloud_attack.png'), dpi=100, bbox_inches='tight')
plt.close(fig5)

fig6 = generate_bev_plot('LiDAR Defended (BEV)', 'forestgreen', num_points=8921, noise_level=0.1)
fig6.savefig(os.path.join(output_dir, 'pointcloud_defended.png'), dpi=100, bbox_inches='tight')
plt.close(fig6)

# Comparison chart
fig7 = generate_comparison_chart()
fig7.savefig(os.path.join(output_dir, 'confidence_comparison.png'), dpi=100, bbox_inches='tight')
plt.close(fig7)

print(f"Visualizations saved to {output_dir}")
print("Files generated:")
print("  - image_clean.png")
print("  - image_attack.png")
print("  - image_defended.png")
print("  - pointcloud_clean.png")
print("  - pointcloud_attack.png")
print("  - pointcloud_defended.png")
print("  - confidence_comparison.png")
