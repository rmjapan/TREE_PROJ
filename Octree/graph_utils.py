import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from ocnn.octree.shuffled_key import key2xyz

def map_value_to_label(value):
    """
    Octree の node.value をクラスラベルに変換する関数の例
    ※ここでは、-1 を空（クラス0）、1 を占有（クラス2）、その他を中間（クラス1）として扱います
    """
    if value == -1:
        return 0
    elif value == 1:
        return 2
    else:
        return 1

def extract_features_from_nodes(nodes):
    """
    各ノードの情報（開始座標, サイズ, ラベル）から
    GraphConv の入力となる特徴ベクトルを作成する例です。
    
    特徴は以下の順番：
      [x, y, z, size, one-hot(label)]  （label 次元はここでは 3 と仮定）
    """
    indices = np.array([[n["x"], n["y"], n["z"]] for n in nodes])
    sizes = np.array([[n["size"]] for n in nodes])
    labels = np.array([map_value_to_label(n["value"]) for n in nodes])
    label_vecs = np.eye(3)[labels]  # クラス数 3 を仮定
    features = np.concatenate([indices, sizes, label_vecs], axis=1)
    return torch.tensor(features, dtype=torch.float32)

def quantize_features_tensor(feature_tensor, label_start_idx=4, num_classes=3, pow2_size=True):
    """
    feature_tensor の特徴を量子化します。
    
    Args:
        feature_tensor: 特徴テンソル [num_nodes, feature_dim]
            特徴の順番: [x, y, z, size, one-hot label値...]
        label_start_idx: ラベル部分の開始インデックス
        num_classes: クラス数
        pow2_size: サイズを2の累乗に丸めるかどうか
            
    Returns:
        quantized: 量子化された特徴テンソル
    """
    quantized = feature_tensor.clone()  # オリジナルを変更しないためコピー
    
    # 座標部分を整数に丸める
    quantized[:, :3] = torch.round(quantized[:, :3])
    
    # サイズ部分を調整
    if pow2_size:
        # サイズを2の累乗値に丸める
        size_values = quantized[:, 3].clamp(min=1.0)
        log2_size = torch.log2(size_values)
        rounded_log2 = torch.round(log2_size)
        quantized[:, 3] = 2.0 ** rounded_log2
    else:
        # 単純に整数に丸める
        quantized[:, 3] = torch.round(quantized[:, 3]).clamp(min=1.0)
    
    # ラベル部分を量子化（one-hot形式に変換）
    label_portion = quantized[:, label_start_idx:label_start_idx + num_classes]
    max_indices = torch.argmax(label_portion, dim=1)
    one_hot_labels = torch.zeros_like(label_portion)
    one_hot_labels[torch.arange(feature_tensor.size(0)), max_indices] = 1.0
    quantized[:, label_start_idx:label_start_idx + num_classes] = one_hot_labels
    
    return quantized

def to_power_of_two(value: float) -> int:
    """
    与えられた値に最も近い2のべき乗を返します
    
    Args:
        value: 入力値
        
    Returns:
        最も近い2のべき乗整数値
    """
    if value <= 0:
        return 1
    log2 = np.log2(value)
    rounded_log2 = round(log2)
    return int(2 ** rounded_log2)

def reconstruct_voxels_from_features(feature_tensor, volume_size, 
                                     enforce_pow2=True,
                                     post_process=True,
                                     label_start_idx=4):
    """
    特徴テンソルからボクセルを再構成する関数
    
    Args:
        feature_tensor: [num_nodes, feature_dim] のテンソルまたは NumPy 配列
                        特徴の順番は [x, y, z, size, one-hot_label...]
        volume_size: 再構成する立方体ボリュームのサイズ
        enforce_pow2: サイズを2のべき乗に強制するかどうか
        post_process: 後処理を適用するかどうか
        label_start_idx: ラベル部分の開始インデックス
        
    Returns:
        voxels: 再構成されたボクセルの 3次元 NumPy 配列
    """
    # 初期状態のボクセル領域は空（ここでは -1 としておく）
    voxels = np.full((volume_size, volume_size, volume_size), -1, dtype=np.float32)
    
    # 優先度（サイズの大きい順にレンダリング）のためのソート
    if isinstance(feature_tensor, torch.Tensor):
        feature_np = feature_tensor.detach().cpu().numpy()
    else:
        feature_np = feature_tensor
    
    # 各ノードの情報をサイズの降順にソート
    size_indices = np.argsort(-feature_np[:, 3])
    sorted_features = feature_np[size_indices]
    
    # 確実性スコア（サイズが大きく、ラベルの確信度が高いノードを優先）
    # これにより、大きな確実なノードが小さな不確実なノードを上書きする
    certainty_scores = np.zeros(voxels.shape, dtype=np.float32)
    
    # 各ノードの特徴から領域を再構成
    for feature in sorted_features:
        # インデックス・サイズの取得（整数に変換）
        x = int(np.round(feature[0]))
        y = int(np.round(feature[1]))
        z = int(np.round(feature[2]))
        
        # サイズを2のべき乗に調整
        if enforce_pow2:
            size = to_power_of_two(feature[3])
        else:
            size = max(1, int(np.round(feature[3])))
        
        # ボリュームの範囲外にある場合はスキップ
        if (x < 0 or y < 0 or z < 0 or 
            x >= volume_size or y >= volume_size or z >= volume_size):
            continue
        
        # 残りの部分はラベルのワンホット表現
        onehot = feature[label_start_idx:]
        label_idx = int(np.argmax(onehot))
        label_confidence = float(onehot[label_idx])
        
        # ここでラベルからボクセルの値への変換を行う
        if label_idx == 0:
            voxel_value = -1   # 空
        elif label_idx == 2:
            voxel_value = 1    # 占有
        else:
            voxel_value = 0.5  # 中間など
        
        # 対象領域の範囲を決定（volume_sizeを超えないように調整）
        x_end = min(x+size, volume_size)
        y_end = min(y+size, volume_size)
        z_end = min(z+size, volume_size)
        
        # ノードの確実性スコアを計算（サイズとラベル確信度の組み合わせ）
        node_certainty = size * label_confidence
        
        # 領域を更新（確実性が高い場合のみ）
        region_x, region_y, region_z = np.meshgrid(
            np.arange(x, x_end),
            np.arange(y, y_end),
            np.arange(z, z_end),
            indexing='ij'
        )
        
        # 更新すべき位置のマスクを作成
        if post_process:
            # 確実性スコアを比較して更新
            update_mask = node_certainty > certainty_scores[region_x, region_y, region_z]
            voxels[region_x[update_mask], region_y[update_mask], region_z[update_mask]] = voxel_value
            certainty_scores[region_x[update_mask], region_y[update_mask], region_z[update_mask]] = node_certainty
        else:
            # シンプルに全ての位置を更新
            voxels[x:x_end, y:y_end, z:z_end] = voxel_value

    return voxels 


# 変換処理
def edge_index_to_networkx(edge_index, directed=False, dual_octree_graph=None,depth=0):
    from Octree.octree import OctreeNode
    # edge_index を numpy に変換
    edge_index = edge_index.cpu().numpy()
    src = edge_index[0]
    dst = edge_index[1]

    # NetworkX グラフの作成
    G = nx.DiGraph() if directed else nx.Graph()
    
    
    # ベクトル化処理
    if len(src) > 0:
        # ノードIDをテンソルに変換
        src_tensor = torch.tensor(src)
        dst_tensor = torch.tensor(dst)

        #深さ情報を取得
        src_depth = dual_octree_graph.node_depth[src_tensor]
        dst_depth = dual_octree_graph.node_depth[dst_tensor]
        #(8^depth-1)/7を計算
        #完全分割の場合
        print(f"ncum:{dual_octree_graph.ncum}")
        src_key=dual_octree_graph.ncum[src_depth]
        dst_key=dual_octree_graph.ncum[dst_depth]
        src_key=src_key.int()
        dst_key=dst_key.int()


  
        src_true_tensor=src_tensor-src_key
        dst_true_tensor=dst_tensor-dst_key
        # 座標変換をベクトル化
 
    
        src_x, src_y, src_z, _ = key2xyz(src_true_tensor)
        dst_x, dst_y, dst_z, _ = key2xyz(dst_true_tensor)
        
        
  
  
        
        # サイズ計算 - テンソルの要素ごとに計算
        src_size = torch.pow(2, depth-src_depth)
        dst_size = torch.pow(2, depth-dst_depth)

        # depth_1_index=torch.where(src_depth==5)
        # pos_src=torch.stack([src_x[depth_1_index],src_y[depth_1_index],src_z[depth_1_index]],dim=1)
        # pos_dst=torch.stack([dst_x[depth_1_index],dst_y[depth_1_index],dst_z[depth_1_index]],dim=1)
        # print(pos_src)
        
        
        src_x=(src_x+0.5)*src_size
        src_y=(src_y+0.5)*src_size
        src_z=(src_z+0.5)*src_size
        dst_x=(dst_x+0.5)*dst_size
        dst_y=(dst_y+0.5)*dst_size
        dst_z=(dst_z+0.5)*dst_size

        #深さ1のindexを取得
        depth_1_index=torch.where(src_depth==5)
        pos_src=torch.stack([src_x[depth_1_index],src_y[depth_1_index],src_z[depth_1_index]],dim=1)
        pos_dst=torch.stack([dst_x[depth_1_index],dst_y[depth_1_index],dst_z[depth_1_index]],dim=1)
        print(pos_src)

        # Fix tensor comparison operations with element-wise operations
        # src_x = torch.where(src_x >=256, src_x - 256, src_x)
        # dst_x = torch.where(dst_x >=256, dst_x - 256, dst_x)
        # src_y = torch.where(src_y >=256, src_y - 256, src_y)
        # dst_y = torch.where(dst_y >=256, dst_y - 256, dst_y)
        # src_z = torch.where(src_z >=256, src_z - 256, src_z)
        # dst_z = torch.where(dst_z >=256, dst_z - 256, dst_z)
    
        # ノードとエッジを追加
        for i in range(len(src)):
            
            u_node = OctreeNode(is_leaf=True, value=0, children=None, 
                               depth=src_depth[i].item(), x=src_x[i].item(), 
                               y=src_y[i].item(), z=src_z[i].item(), 
                               size=src_size[i].item(), id=src_true_tensor[i].item()%8)

            v_node = OctreeNode(is_leaf=True, value=0, children=None, 
                               depth=dst_depth[i].item(), x=dst_x[i].item(), 
                               y=dst_y[i].item(), z=dst_z[i].item(), 
                               size=dst_size[i].item(), id=dst_true_tensor[i].item()%8)
            

            G.add_node(u_node)
            G.add_node(v_node)
            G.add_edge(u_node, v_node)
            G.add_edge(v_node, u_node)


    return G


