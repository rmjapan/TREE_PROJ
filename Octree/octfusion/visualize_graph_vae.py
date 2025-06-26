import torch
import torch.nn as nn
import os
import sys
from graphviz import Digraph
import numpy as np
from torch.autograd import Variable

sys.path.append('/home/ryuichi/tree/Other/related_work/octfusion')
from models.networks.dualoctree_networks.graph_vae import GraphVAE

def make_dot(var, params=None, title="Network Architecture"):
    """
    Generate a visualization of a neural network from a PyTorch Variable
    using graphviz.
    
    Args:
        var: PyTorch Variable or tensor representing the output.
        params (dict, optional): Dictionary of parameter tensors to label in the graph.
        title (str): Title of the visualization.
    """
    if params is None:
        params = dict(var.named_parameters())
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr={'rankdir': 'LR', 'label': title})
    
    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(graph, var, prefix=""):
        if var not in seen:
            if torch.is_tensor(var):
                # tensor leaf
                node_name = prefix + str(id(var))
                graph.node(node_name, label=size_to_str(var.size()), fillcolor='lightblue')
                seen.add(var)
                if params is not None and var in params:
                    param_name = next((name for name, param in params.items() if param is var), None)
                    if param_name:
                        graph.node(param_name, label=f"{param_name}\n{size_to_str(var.size())}", fillcolor='orange')
                        graph.edge(param_name, node_name)
            elif hasattr(var, 'variable'):
                # autograd Variable
                add_nodes(graph, var.variable, prefix)
            elif hasattr(var, 'next_functions'):
                # autograd function
                node_name = prefix + str(id(var))
                # Get the name of the autograd function
                graph.node(node_name, label=type(var).__name__, fillcolor='lightgrey')
                seen.add(var)
                
                # Add edges from inputs
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(graph, u[0], prefix)
                        graph.edge(prefix + str(id(u[0])), node_name)
                
                # Add edges to outputs
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        add_nodes(graph, t, prefix)
                        graph.edge(node_name, prefix + str(id(t)))
    
    seen = set()
    output_nodes = var.grad_fn if hasattr(var, 'grad_fn') else var
    add_nodes(dot, output_nodes)
    
    return dot

def visualize_graph_vae(model, output_path="graph_vae_structure.pdf"):
    """
    Visualize GraphVAE model architecture using torchviz.
    
    Args:
        model (GraphVAE): The model to visualize
        output_path (str): Path to save the visualization
    """
    print("Creating dummy inputs for visualization...")
    
    # Create dummy inputs that avoid using Octree
    # We'll use simple tensors that match expected dimensions
    batch_size = 1
    depth_stop = model.depth_stop
    feature_dim = model.channels[depth_stop]
    
    # Create a dummy tensor for the KL conv input (encoder output)
    dummy_encoder_output = torch.zeros(batch_size, feature_dim, 1, 1, 1, requires_grad=True)
    
    # Create a dummy tensor for the post KL conv input (decoder input)
    # This would normally be the sampled latent code
    dummy_latent_code = torch.zeros(batch_size, model.embed_dim, 1, 1, 1, requires_grad=True)
    
    # Trace a forward pass through critical components
    print("Tracing KL conv (encoder output)...")
    kl_output = model.KL_conv(dummy_encoder_output)
    
    print("Tracing post KL conv (decoder input)...")
    decoder_input = model.post_KL_conv(dummy_latent_code)
    
    # Visualize the KL conv part (encoder output to latent space)
    print("Generating encoder visualization...")
    encoder_viz = make_dot(kl_output, dict(model.named_parameters()), "GraphVAE Encoder Output")
    encoder_viz.render(os.path.join(os.path.dirname(output_path), "graph_vae_encoder"), format="pdf")
    
    # Visualize the post KL conv part (latent space to decoder input)
    print("Generating decoder visualization...")
    decoder_viz = make_dot(decoder_input, dict(model.named_parameters()), "GraphVAE Decoder Input")
    decoder_viz.render(os.path.join(os.path.dirname(output_path), "graph_vae_decoder"), format="pdf")
    
    # Print model structure as text for additional reference
    print("\nModel Structure:")
    print(model)
    
    # Print model hyperparameters
    print("\nModel Hyperparameters:")
    print(f"depth: {model.depth}, full_depth: {model.full_depth}")
    print(f"depth_stop: {model.depth_stop}, depth_out: {model.depth_out}")
    print(f"channel_in: {model.channel_in}, nout: {model.nout}")
    print(f"bottleneck: {model.bottleneck}, embed_dim: {model.embed_dim}")
    
    print(f"\nVisualizations saved to {os.path.dirname(output_path)}")
    print(f"- Encoder: graph_vae_encoder.pdf")
    print(f"- Decoder: graph_vae_decoder.pdf")

if __name__ == "__main__":
    # Create model with the same parameters as in the original code
    model = GraphVAE(depth=6, channel_in=4, nout=4, full_depth=2, 
                     depth_stop=4, depth_out=6, code_channel=16, embed_dim=3)
    
    # Default output directory
    output_dir = "visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize the model
    visualize_graph_vae(model, os.path.join(output_dir, "graph_vae_structure.pdf")) 