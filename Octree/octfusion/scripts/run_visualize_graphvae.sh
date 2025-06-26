#!/bin/bash

# Setup paths
CONFIG_PATH="configs/vae_snet_eval.yaml"  # Path to VAE config file
CHECKPOINT_PATH="saved_ckpt/vae-ckpt/vae-shapenet-depth-8.pth"  # Path to VAE checkpoint

# Create example data directory and output directory
EXAMPLE_DATA_DIR="datasets/example_data"
OUTPUT_DIR="visualization_results/graphvae_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXAMPLE_DATA_DIR
mkdir -p $OUTPUT_DIR

# Create a simple spherical point cloud for testing
python -c "
import numpy as np
import os

# Create a spherical point cloud with 5000 points
n_points = 5000
phi = np.random.uniform(0, 2*np.pi, n_points)
theta = np.random.uniform(0, np.pi, n_points)

# Convert spherical to Cartesian coordinates
x = 0.5 * np.sin(theta) * np.cos(phi)  # scale to radius 0.5
y = 0.5 * np.sin(theta) * np.sin(phi)
z = 0.5 * np.cos(theta)

points = np.stack([x, y, z], axis=1)
# Normals point outward from the center
normals = points.copy() / np.linalg.norm(points, axis=1, keepdims=True)

os.makedirs('$EXAMPLE_DATA_DIR', exist_ok=True)
np.savez('$EXAMPLE_DATA_DIR/pointcloud.npz', points=points, normals=normals)
print('Created spherical point cloud with', n_points, 'points')
"

DATA_PATH="$EXAMPLE_DATA_DIR/pointcloud.npz"

echo "Using config: $CONFIG_PATH"
echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Using data: $DATA_PATH"
echo "Results will be saved to: $OUTPUT_DIR"

# Run the visualization script
python tools/visualize_graphvae.py \
    --vq_cfg $CONFIG_PATH \
    --vq_ckpt $CHECKPOINT_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device cuda

echo "Visualization complete. Results saved to $OUTPUT_DIR" 