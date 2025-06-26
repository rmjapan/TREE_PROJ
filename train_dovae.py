import argparse
import numpy as np
import torch
from utils import npz2dense
from Octree.octree import build_octree
from Octree.dual_octree import build_dual_octree_graph,visualize_dual_octree_graph_3d
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from model.networks.GraphConv import GraphConv
def load_svs_data(idx, svs_dataset_path):
        """--svs_dataset_path で指定されたディレクトリから svs_{idx+1}.npz を読み込み、前処理してテンソルとして返す
        """
        svs = np.load(f"{svs_dataset_path}/svs_{idx+1}.npz")
        svs = npz2dense(svs)
        svs[svs == 0] = -1.0
        svs[svs == 1] = 0.0
        svs[svs == 1.5] = 0.5
        svs[svs == 2] = 1.0

        svs = torch.tensor(svs).float()
        svs = torch.unsqueeze(svs, 0)
        return svs
train_param=argparse.ArgumentParser()
    #データセットのPath
train_param.add_argument(
        "--svs_dataset_path", 
        type=str, 
        default="/mnt/nas/rmjapan2000/tree/data_dir/svd_0.2"
        )
train_param=train_param.parse_args()









def test_dovae():
    voxel_data=load_svs_data(0, train_param.svs_dataset_path)
    #build octreeはNumpyのみしか対応していない
    voxel_data=voxel_data.numpy()
    #build octreeはshape[0]==shape[1]==shape[2]でないといけない
    voxel_data=np.squeeze(voxel_data,axis=0)
    octree=build_octree(voxel_data,depth=0)
    #build dual octreeは最初にGraphを作成する必要がある
    G=nx.Graph()
    G.add_node(octree)
    dual_octree=build_dual_octree_graph(G,0)
    # visualize_dual_octree_graph_3d(dual_octree,0)
    #dual_octreeノードをテンソルに変換する.
    import time 
    start_time=time.time()
    data=from_networkx(dual_octree)
    end_time=time.time()
    print(f"from_networkxの時間: {end_time-start_time}秒")
    data.edge_type=0
    print(data)
    #
    graph_conv=GraphConv(in_channels=1,out_channels=1,n_edge_type=7,avg_degree=7,n_node_type=0,use_bias=False)
    out=graph_conv(data)
    print(out)


test_dovae()