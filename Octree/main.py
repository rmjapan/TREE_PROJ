import time
import numpy as np
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from my_dataset.svsdataset import SvsDataset
from octree import build_octree,
from dual_octree import build_dual_octree_graph
import networkx as nx
import matplotlib.pyplot as plt
###############################################################################
# メイン処理（データ読み込み、オクツリー構築、エッジ抽出など）
###############################################################################

def main():
    """
    メイン処理: データセットの読み込み、Octreeの構築、グラフ抽出、モデルのトレーニングなどを行います
    """
    # データの読み込みとオクツリーの構築
    print("データセットを読み込んでいます...")
    svs = SvsDataset(data_dir="/mnt/nas/rmjapan2000/tree/data_dir/svd_0.2")
    voxels = svs[10].squeeze(0).numpy()
    print(f"ボクセルサイズ: {voxels.shape}")

    print("Octreeを構築しています...")
    start_time = time.time()
    tree = build_octree(voxels,depth=0)
    end_time = time.time()
    print(f"Octreeの構築時間: {end_time - start_time:.4f}秒")

    # オクツリーからグラフ構造（ノードリスト・エッジリスト）の抽出
    print("グラフ構造を抽出しています...")
    volume_size = voxels.shape[0]
    G=nx.Graph()
    G.add_node(tree)
    print(f"tree: {tree}")
    print(f"tree.id_list: {tree.id_list}")
    print(f"tree_children1: {tree.children[1].id_list}")
    G=build_dual_octree_graph(G,0)
    print(f"グラフのノード数: {len(G.nodes())}")
    print(f"グラフのエッジ数: {len(G.edges())}")
 
if __name__ == "__main__":
    # コマンドライン引数などで動作を制御するためのオプション
    import argparse
    parser = argparse.ArgumentParser(description='Octree Graph Autoencoder実験')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'experiment'],
                        help='実行モード (train: 単一サンプル, experiment: 複数サンプル)')
    parser.add_argument('--samples', type=int, default=3,
                        help='実験モードで使用するサンプル数')
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()

