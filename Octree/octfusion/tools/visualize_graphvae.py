import os
import sys
import torch
import argparse
import trimesh
import numpy as np
import skimage.measure
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks.graph_vae import GraphVAE
from models.networks.dualoctree_networks.dual_octree import DualOctree
from models.model_utils import load_dualoctree
from utils.util_dualoctree import calc_sdf
from utils.visualizer import Visualizer


def load_example_data(data_path, point_scale=1.0):
    """Load example point cloud data and convert to octree."""
    # Load point cloud data from .npz file
    raw = np.load(data_path)
    points = raw['points']
    normals = raw['normals']
    
    # Scale points to [-1, 1]
    points = points / point_scale
    
    # Create Points object
    points_obj = Points(
        points=torch.from_numpy(points).float(), 
        normals=torch.from_numpy(normals).float()
    )
    points_obj.clip(min=-1, max=1)
    
    # Create octree from points
    depth = 6  # Standard depth used in octfusion
    full_depth = 2  # Full depth for GraphVAE
    octree = Octree(depth, full_depth)
    octree.build_octree(points_obj)
    
    return octree, points_obj


def visualize_octree(octree, save_path, title="Octree Visualization"):
    """Visualize octree structure by rendering non-empty nodes."""
    # Extract non-empty nodes at different depths
    nodes_by_depth = {}
    
    # Get the maximum depth of the octree
    max_depth = octree.depth
    
    fig = plt.figure(figsize=(15, 5 * ((max_depth + 2) // 3)))
    
    # Create subplots for each depth
    for d in range(max_depth + 1):
        try:
            # Get non-empty nodes at this depth
            mask = octree.nempty_mask(d)
            if mask is None or mask.numel() == 0:
                continue
                
            # Calculate the number of nodes at this depth
            n_nodes = mask.sum().item()
            if n_nodes == 0:
                continue
                
            # Store node count
            nodes_by_depth[d] = n_nodes
            
            # If it's possible, visualize the non-empty nodes
            if d <= 5:  # Only for lower depths that can be visualized
                # Create a grid visualization
                plt.subplot(((max_depth + 2) // 3), 3, d + 1)
                plt.title(f"Depth {d}: {n_nodes} nodes")
                
                # For depth <= 3, we can visualize the actual octree structure
                if d <= 3:
                    # Create a 3D visualization
                    ax = plt.gca(projection='3d')
                    
                    # Get coordinates for this depth
                    size = 2**(max_depth - d)
                    coords = torch.nonzero(mask.view(2**d, 2**d, 2**d))
                    
                    # Plot each non-empty node as a cube
                    for coord in coords:
                        x, y, z = coord
                        # Create wireframe cube
                        xx = np.array([[x, x+1, x+1, x, x],
                                      [x, x+1, x+1, x, x]])
                        yy = np.array([[y, y, y+1, y+1, y],
                                      [y, y, y+1, y+1, y]])
                        zz = np.array([[z, z, z, z, z],
                                      [z+1, z+1, z+1, z+1, z+1]])
                        ax.plot_wireframe(xx, yy, zz, color='blue', alpha=0.5)
                    
                    ax.set_xlim(0, 2**d)
                    ax.set_ylim(0, 2**d)
                    ax.set_zlim(0, 2**d)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                else:
                    # For higher depths, just show a 2D slice
                    middle_slice = 2**(d-1)  # Middle slice along z-axis
                    slice_mask = mask.view(2**d, 2**d, 2**d)[:, :, middle_slice]
                    plt.imshow(slice_mask.cpu().numpy(), cmap='Blues')
                    plt.colorbar(label='Non-empty (1)')
        except Exception as e:
            print(f"Error visualizing depth {d}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return nodes_by_depth


def visualize_latent_code(z, save_dir):
    """Visualize multiple slices of the latent code."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the shape of the latent code
    batch_size, channels, *spatial_dims = z.shape
    
    # Create an overview plot with multiple channel slices
    n_channels_to_show = min(channels, 4)
    fig, axes = plt.subplots(1, n_channels_to_show, figsize=(n_channels_to_show * 5, 5))
    
    if n_channels_to_show == 1:
        axes = [axes]
    
    for c in range(n_channels_to_show):
        latent_slice = z[0, c].cpu().detach().numpy()
        
        # If 3D latent code, take a middle slice along z-axis
        if len(spatial_dims) == 3:
            middle_z = spatial_dims[2] // 2
            latent_slice = latent_slice[:, :, middle_z]
        
        axes[c].imshow(latent_slice, cmap='viridis')
        axes[c].set_title(f"Channel {c}")
        axes[c].colorbar = plt.colorbar(axes[c].imshow(latent_slice, cmap='viridis'), ax=axes[c])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "latent_code_overview.png"))
    plt.close()
    
    # Create a 3D visualization of the first channel if the latent code is 3D
    if len(spatial_dims) == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get the first channel
        latent_vol = z[0, 0].cpu().detach().numpy()
        
        # Create a threshold to show only significant values
        threshold = np.percentile(np.abs(latent_vol), 90)
        
        # Create a meshgrid for the 3D volume
        x, y, z_grid = np.meshgrid(
            np.arange(latent_vol.shape[0]),
            np.arange(latent_vol.shape[1]),
            np.arange(latent_vol.shape[2])
        )
        
        # Only show voxels above the threshold
        mask = np.abs(latent_vol) > threshold
        ax.scatter(x[mask], y[mask], z_grid[mask], c=latent_vol[mask], cmap='viridis', alpha=0.5)
        
        ax.set_title("3D Latent Space (Channel 0, significant values)")
        plt.savefig(os.path.join(save_dir, "latent_code_3d.png"))
        plt.close()
    
    return {
        'shape': z.shape,
        'min': z.min().item(),
        'max': z.max().item(),
        'mean': z.mean().item(),
        'std': z.std().item()
    }


def visualize_graphvae_process(model, octree_in, save_dir, device="cpu"):
    """Visualize the GraphVAE encoding-decoding process."""
    model.eval()
    
    # Create directories for different stages
    input_dir = os.path.join(save_dir, "input")
    latent_dir = os.path.join(save_dir, "latent")
    output_dir = os.path.join(save_dir, "output")
    recon_dir = os.path.join(save_dir, "reconstruction")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Visualize input octree
    print("Visualizing input octree...")
    input_octree_stats = visualize_octree(octree_in, os.path.join(input_dir, "input_octree.png"), "Input Octree")
    
    # Move input to device
    octree_in = octree_in.to(device)
    
    # Create DualOctree from input
    doctree_in = DualOctree(octree=octree_in)
    doctree_in.post_processing_for_docnn()
    
    # Encode input octree to get posterior distribution
    print("Encoding input octree...")
    posterior = model.octree_encoder(octree_in, doctree_in)
    
    # Sample from posterior
    z = posterior.sample()
    
    # Visualize latent code
    print("Visualizing latent code...")
    latent_stats = visualize_latent_code(z, latent_dir)
    
    # Create output octree for decoding
    update_octree = True
    octree_out = model.create_full_octree(octree_in)
    octree_out.depth = model.full_depth
    
    # Grow octree to depth_stop
    for d in range(model.full_depth, model.depth_stop):
        label = octree_in.nempty_mask(d).long()
        octree_out.octree_split(label, d)
        octree_out.octree_grow(d + 1)
        octree_out.depth += 1
    
    # Visualize initial output octree
    print("Visualizing initial output octree...")
    output_octree_stats = visualize_octree(octree_out, os.path.join(output_dir, "initial_output_octree.png"), "Initial Output Octree")
    
    # Create DualOctree for output
    doctree_out = DualOctree(octree_out)
    doctree_out.post_processing_for_docnn()
    
    # Decode latent code
    print("Decoding latent code...")
    out = model.octree_decoder(z, doctree_out, update_octree=update_octree)
    logits, reg_voxs, octree_out = out
    
    # Visualize reconstructed octree
    print("Visualizing reconstructed octree...")
    recon_octree_stats = visualize_octree(octree_out, os.path.join(recon_dir, "reconstructed_octree.png"), "Reconstructed Octree")
    
    # Extract neural MPU function from output
    def neural_mpu(pos):
        pred = model.neural_mpu(pos, reg_voxs, octree_out)
        return pred[model.depth_out][0]
    
    # Calculate SDF values at different resolutions
    print("Calculating SDF values...")
    bbmin, bbmax = -0.9, 0.9
    
    resolutions = [32, 64, 128]
    sdf_values = {}
    
    for res in resolutions:
        print(f"  Resolution: {res}x{res}x{res}")
        sdfs = calc_sdf(neural_mpu, 1, size=res, bbmin=bbmin, bbmax=bbmax)
        sdf_values[res] = sdfs
    
    # Visualize SDF slices
    print("Visualizing SDF values...")
    sdf_dir = os.path.join(recon_dir, "sdf")
    os.makedirs(sdf_dir, exist_ok=True)
    
    for res, sdfs in sdf_values.items():
        sdf = sdfs[0].cpu().numpy()
        
        # Plot middle slices along each axis
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        mid_x = sdf.shape[0] // 2
        mid_y = sdf.shape[1] // 2
        mid_z = sdf.shape[2] // 2
        
        im0 = axes[0].imshow(sdf[mid_x, :, :], cmap='viridis')
        axes[0].set_title(f"YZ Plane (X={mid_x})")
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(sdf[:, mid_y, :], cmap='viridis')
        axes[1].set_title(f"XZ Plane (Y={mid_y})")
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(sdf[:, :, mid_z], cmap='viridis')
        axes[2].set_title(f"XY Plane (Z={mid_z})")
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(sdf_dir, f"sdf_slices_res{res}.png"))
        plt.close()
    
    # Extract mesh using marching cubes
    print("Extracting mesh...")
    mesh_dir = os.path.join(recon_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Use the highest resolution SDF for mesh extraction
    highest_res = max(resolutions)
    sdf_value = sdf_values[highest_res][0].cpu().numpy()
    
    try:
        vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, 0)
        vtx = vtx * ((bbmax - bbmin) / highest_res) + bbmin
        mesh = trimesh.Trimesh(vtx, faces)
        mesh.export(os.path.join(mesh_dir, "reconstructed.obj"))
        print(f"Mesh saved to {os.path.join(mesh_dir, 'reconstructed.obj')}")
        
        # Save mesh visualization from different angles
        for angle_idx, angle in enumerate([0, 45, 90, 135, 180]):
            scene = trimesh.Scene(mesh)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle * np.pi / 180, [0, 1, 0], [0, 0, 0])
            scene.camera_transform = rotation_matrix
            png = scene.save_image(resolution=[640, 480])
            with open(os.path.join(mesh_dir, f"mesh_view_{angle_idx}.png"), 'wb') as f:
                f.write(png)
                
    except Exception as e:
        print(f"Warning: Marching cubes failed to generate a mesh: {e}")
    
    # Return statistics
    depth_stats = {}
    for d in range(model.depth_stop, model.depth_out + 1):
        if d in logits:
            depth_stats[d] = {
                'nnum': doctree_out.nnum[d],
                'logit_shape': logits[d].shape if d in logits else None,
                'reg_vox_shape': reg_voxs[d].shape if d in reg_voxs else None
            }
    
    return {
        'latent_stats': latent_stats,
        'kl_loss': posterior.kl().mean().item(),
        'input_octree': input_octree_stats,
        'initial_output_octree': output_octree_stats,
        'reconstructed_octree': recon_octree_stats,
        'depth_stats': depth_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize GraphVAE processing')
    parser.add_argument('--vq_cfg', type=str, required=True, help='Path to the VAE config file')
    parser.add_argument('--vq_ckpt', type=str, required=True, help='Path to the VAE checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to example data (.npz file)')
    parser.add_argument('--output_dir', type=str, default='graph_vae_vis', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from {args.vq_cfg}...")
    try:
        vq_conf = OmegaConf.load(args.vq_cfg)
        print("Config loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Load model
    print(f"Loading GraphVAE model from {args.vq_ckpt}...")
    try:
        model = load_dualoctree(conf=vq_conf, ckpt=args.vq_ckpt)
        model.to(args.device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Print model architecture summary
    print("\nGraphVAE Model Configuration:")
    print(f"  Depth: {model.depth}")
    print(f"  Full Depth: {model.full_depth}")
    print(f"  Depth Stop: {model.depth_stop}")
    print(f"  Depth Out: {model.depth_out}")
    print(f"  Code Channel: {model.code_channel}")
    
    # Load example data
    print(f"Loading example data from {args.data_path}...")
    try:
        point_scale = vq_conf.data.test.point_scale if 'point_scale' in vq_conf.data.test else 1.0
        octree_in, points_obj = load_example_data(args.data_path, point_scale)
        print(f"Data loaded successfully: {len(points_obj.points)} points")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Save input point cloud
    input_dir = os.path.join(args.output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    try:
        pointcloud = trimesh.PointCloud(vertices=points_obj.points.numpy())
        pointcloud.export(os.path.join(input_dir, "input.ply"))
        print(f"Input point cloud saved to {os.path.join(input_dir, 'input.ply')}")
    except Exception as e:
        print(f"Error saving point cloud: {e}")
    
    # Visualize GraphVAE processing
    print("\nStarting GraphVAE visualization process...")
    try:
        stats = visualize_graphvae_process(model, octree_in, args.output_dir, device=args.device)
        
        # Save stats
        with open(os.path.join(args.output_dir, "stats.txt"), "w") as f:
            f.write("GraphVAE Processing Statistics\n")
            f.write("=============================\n\n")
            
            f.write("Latent Code Statistics:\n")
            f.write(f"  Shape: {stats['latent_stats']['shape']}\n")
            f.write(f"  Min: {stats['latent_stats']['min']:.4f}\n")
            f.write(f"  Max: {stats['latent_stats']['max']:.4f}\n")
            f.write(f"  Mean: {stats['latent_stats']['mean']:.4f}\n")
            f.write(f"  Std: {stats['latent_stats']['std']:.4f}\n")
            f.write(f"  KL Loss: {stats['kl_loss']:.4f}\n\n")
            
            f.write("Octree Statistics:\n")
            f.write("  Input Octree:\n")
            for depth, count in stats['input_octree'].items():
                f.write(f"    Depth {depth}: {count} nodes\n")
            
            f.write("\n  Initial Output Octree:\n")
            for depth, count in stats['initial_output_octree'].items():
                f.write(f"    Depth {depth}: {count} nodes\n")
            
            f.write("\n  Reconstructed Octree:\n")
            for depth, count in stats['reconstructed_octree'].items():
                f.write(f"    Depth {depth}: {count} nodes\n")
            
            f.write("\nDepth-wise Statistics:\n")
            for depth, depth_stat in stats['depth_stats'].items():
                f.write(f"  Depth {depth}:\n")
                f.write(f"    Number of nodes: {depth_stat['nnum']}\n")
                if depth_stat['logit_shape'] is not None:
                    f.write(f"    Logit shape: {depth_stat['logit_shape']}\n")
                if depth_stat['reg_vox_shape'] is not None:
                    f.write(f"    Reg vox shape: {depth_stat['reg_vox_shape']}\n")
        
        print(f"Processing stats saved to {os.path.join(args.output_dir, 'stats.txt')}")
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 