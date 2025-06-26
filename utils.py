import numpy as np
from scipy.sparse import csr_matrix
import sys
import torch

import svs 
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from svs import visualize_voxel_data
from typing import Union

# H,W,D=256,256,256
def npz2dense(npz_data,H=256,W=256,D=256):
    # 疎行列を復元
    sparse_matrix = csr_matrix((npz_data["data"], npz_data["indices"], npz_data["indptr"]), shape=tuple( npz_data["shape"]))
    # print(f"疎行列のshape:{sparse_matrix.shape}")
    dense_matrix=sparse_matrix.toarray().reshape(H,W,D)
    return dense_matrix
def dense2sparse(x):
    # 疎形式に変換
    sparse_x = x.to_sparse_coo()

    # 疎形式のインデックスを取得
    print(f"Sparse indices:\n{sparse_x.indices()}")
    print(f"Sparse values:\n{sparse_x.values()}")
    print(sparse_x)
    x=sparse_x
    return x

def svs_change_value():
    """
    元の値
    幹：1
    枝：2
    葉：3
    空白：0
    
    変更後の値
    幹：1
    枝：0.5
    葉：0
    空白：-1
    """
    data_path="/home/ryuichi/tree/TREE_PROJ/data_dir/svs/AcaciaClustered/svs_1.npz"
    data=np.load(data_path)
    dense_matrix=npz2dense(data)
    dense_matrix[dense_matrix==2]=0.5
    dense_matrix[dense_matrix==3]=0
    dense_matrix[dense_matrix==0]=-1

    


def weighted_mse_loss(output, target):
    
# カテゴリごとの重み設定（例: 空白は軽視、幹は重視）
    weights = torch.ones_like(target)
    weights[target < 0] = 0.1  # 空白
    weights[(target >= 0) & (target < 0.4)] =10# 葉
    weights[(target >= 0.4) & (target < 0.8)] = 1000  # 枝
    weights[target >= 0.8] = 1000  # 幹

    
    loss = (weights * (output - target) ** 2).mean()
    return loss
# import neuralnet_pytorch as nnt

# from torch_scatter import scatter_add
import trimesh
import numpy as np
import os 
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from visualize_func import visualize_with_timeout

def pyg_to_voxel(data, resolution=32, use_gpu=True):
    """
    PyTorch Geometricのデータ構造から直方体のボクセルグリッドに変換する高速化関数
    
    Args:
        data: PyTorch Geometricのデータオブジェクト (Data(pos=[N, 3], face=[3, M], y=[1]))
        resolution (int): ボクセルグリッドの解像度
        use_gpu (bool): 可能な場合GPUを使用するかどうか
    
    Returns:
        np.array: 解像度^3の直方体ボクセルグリッド（1はボクセルあり、0はボクセルなし）
    """
    device = data.pos.device if use_gpu and data.pos.is_cuda else 'cpu'
    
    # 頂点情報をTensorで高速に処理 (GPUが使用可能なら活用)
    vertices_tensor = data.pos.to(device)
    faces_tensor = data.face.t().to(device)  # [3, M] -> [M, 3]に転置
    
    # バウンディングボックスを高速計算 (Tensorの集約関数を使用)
    min_coords, _ = vertices_tensor.min(dim=0)
    max_coords, _ = vertices_tensor.max(dim=0) 
    
    # バウンディングボックスの中心を計算
    center = (min_coords + max_coords) / 2
    
    # バウンディングボックスのサイズを計算
    sizes = max_coords - min_coords
    
    # 最大の辺の長さを取得
    max_size = sizes.max().item()
    
    # スケーリング係数を計算（直方体の形状を維持するため）
    scale = 0.95 / max_size  # 0.95で少し小さくして余白を確保
    
    # 頂点を正規化（中心を原点に、最大の辺が1に近くなるように）- GPU上で計算
    normalized_vertices = ((vertices_tensor - center) * scale).cpu().numpy()
    
    # 必要な場合のみnumpyに変換（データ転送の最小化）
    faces = faces_tensor.cpu().numpy()
    
    # trimeshメッシュオブジェクトを作成 (validateとprocessを元に戻す)
    mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=faces)
    
    # ボクセル化 (元のパラメータを維持)
    voxel_grid = mesh.voxelized(pitch=1.0/resolution)
    
    # 空の結果配列を先に作成
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # ボクセルグリッドが存在する場合のみ処理
    if hasattr(voxel_grid, 'matrix') and voxel_grid.matrix is not None:
        # ボクセルグリッドのサイズを取得
        vg_shape = voxel_grid.matrix.shape
        
        # voxelsの中心にvoxel_gridを配置
        start_i = max(0, (resolution - vg_shape[0]) // 2)
        start_j = max(0, (resolution - vg_shape[1]) // 2)
        start_k = max(0, (resolution - vg_shape[2]) // 2)
        
        # 配置可能な範囲を計算
        end_i = min(start_i + vg_shape[0], resolution)
        end_j = min(start_j + vg_shape[1], resolution)
        end_k = min(start_k + vg_shape[2], resolution)
        
        # 配置するvoxel_gridの範囲を計算
        vg_end_i = min(vg_shape[0], end_i - start_i)
        vg_end_j = min(vg_shape[1], end_j - start_j)
        vg_end_k = min(vg_shape[2], end_k - start_k)
        
        # ボクセルの配置（スライス操作で一度に割り当て）
        voxels[start_i:start_i+vg_end_i, 
               start_j:start_j+vg_end_j, 
               start_k:start_k+vg_end_k] = voxel_grid.matrix[:vg_end_i, :vg_end_j, :vg_end_k]
    
    # visualize_with_timeout(voxels)
    return voxels

def process_tree_voxels(voxel, max_points_per_part=2000):
    """
    木の3Dボクセルデータから、要素(幹、枝、葉)ごとに点群を抽出する関数
    
    Args:
        voxel (torch.Tensor): 形状 [B,1,H,W,D] のボクセルデータ
        max_points_per_part (int): 各部分ごとの最大点数
    
    Returns:
        dict: 'trunk', 'branch', 'leaf' の各キーに対する点座標のリスト
    """
    B, C, H, W, D = voxel.shape
    device = voxel.device
    
    # スムージング領域を考慮したマスク
    # チャンネル次元を削除する処理(B,256,256,256)
    trunk_mask = (voxel.squeeze(1) >= 0.8)
    branch_mask = (voxel.squeeze(1) >= 0.4) & (voxel.squeeze(1) <= 0.8)
    # 葉のマスクは0以上0.4
    leaf_mask = (voxel.squeeze(1) >= 0) & (voxel.squeeze(1) <= 0.4)
    
    # 座標正規化(GRID)
    x, y, z = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        torch.linspace(-1, 1, D, device=device),
        indexing='ij'
    )
    
    points_dict = {}
    for part, mask in [('trunk', trunk_mask),
                     ('branch', branch_mask),
                     ('leaf', leaf_mask)]:
        # マスクされた座標を取得
        indices = torch.nonzero(mask)
        if len(indices) == 0:
            points_dict[part] = None
            continue
            
        # 一様サンプリングに変更（中心からの距離に依存しない）
        if len(indices) > max_points_per_part:
            # 全点から均等にサンプリングするように修正
            prob = torch.ones(len(indices), device=device)
            prob /= prob.sum()
            selected = torch.multinomial(prob, max_points_per_part)
            indices = indices[selected]
        
        # 座標取得
        part_points = torch.stack([
            x[indices[:,0], indices[:,1], indices[:,2]],
            y[indices[:,0], indices[:,1], indices[:,2]],
            z[indices[:,0], indices[:,1], indices[:,2]]
        ], dim=1)
        points_dict[part] = part_points
        
    return points_dict

def batch_cdist(x, y, chunk_size=512):
    """
    メモリ効率の良い距離計算
    
    Args:
        x (torch.Tensor): 形状 [N, D] の点群1
        y (torch.Tensor): 形状 [M, D] の点群2
        chunk_size (int): 一度に計算する点数
        
    Returns:
        torch.Tensor: 形状 [N, M] の距離行列
    """
    chunks = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i+chunk_size]
        chunks.append(torch.cdist(x_chunk, y))
    return torch.cat(chunks, dim=0)

def discretize_voxel(continuous_voxel):
    """
    連続値のボクセルデータを離散値に変換する関数
    
    Args:
        continuous_voxel (torch.Tensor): 連続値のボクセルデータ
        
    Returns:
        torch.Tensor: 離散化されたボクセルデータ (-1.0, 0.0, 0.5, 1.0 の値)
    """

    discrete_voxel = torch.full_like(continuous_voxel, -1.0)
    mask_trunk = (continuous_voxel >= 0.8)
    mask_branch = (continuous_voxel >= 0.4) & (continuous_voxel < 0.8)
    mask_leaf = (continuous_voxel >= 0.0) & (continuous_voxel < 0.4)
    
    discrete_voxel[mask_trunk] = 1.0   # 幹
    discrete_voxel[mask_branch] = 0.5  # 枝
    discrete_voxel[mask_leaf] = 0.0    # 葉
    
    return discrete_voxel
class DiscretizeWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 通常の離散化（非連続な操作）
        out = torch.full_like(input, -1.0)
        out[input >= 0.8] = 1.0     # 幹
        out[(input >= 0.4) & (input < 0.8)] = 0.5   # 枝
        out[(input >= 0.0) & (input < 0.4)] = 0.0   # 葉
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator（勾配そのまま通す）
        return grad_output

def discretize_voxel_ste(input: torch.Tensor) -> torch.Tensor:
    """
    STE（Straight-Through Estimator）付きのボクセル離散化関数

    Args:
        input (torch.Tensor): モデル出力など連続値テンソル（勾配を通したい）

    Returns:
        torch.Tensor: 離散化されたボクセル（-1.0, 0.0, 0.5, 1.0）ただし勾配はSTEで通る
    """
    return DiscretizeWithSTE.apply(input)


def voxel2xyzfile(voxel: Union[np.ndarray, torch.Tensor, str], filename: str = "output.xyz", color_mode: str = "default") -> None:
    """
    ボクセルデータを.xyzファイル形式で保存する関数（空白は保存しない）

    Args:
        voxel (Union[np.ndarray, torch.Tensor, str]): 3次元配列または.npzファイルパス
            値の範囲: -1.0 (空白), 0.0 (葉), 0.5 (枝), 1.0 (幹)
        filename (str): 出力する.xyzファイルのパス
        color_mode (str): 色の割り当てモード（現在は"default"のみ対応）

    Raises:
        ValueError: 無効な入力データまたは色モードの場合
        IOError: ファイルの読み書きに失敗した場合
    """
    import numpy as np
    from typing import Union
    import os

    # 色の割り当て定義
    COLOR_MAP = {
        1.0: (139, 69, 19),    # 幹: 茶色
        0.5: (205, 133, 63),    # 枝：薄い茶色
        0.0: (0, 255, 0),      # 葉: 明るい緑
    }

    try:
        # データの前処理
        if isinstance(voxel, str):
            if not os.path.exists(voxel):
                raise FileNotFoundError(f"Input file not found: {voxel}")
            voxel = np.load(voxel)
            if isinstance(voxel, np.lib.npyio.NpzFile):
                # 疎行列を復元
                sparse_matrix = csr_matrix((voxel["data"], voxel["indices"], voxel["indptr"]), 
                                         shape=tuple(voxel["shape"]))
                voxel = sparse_matrix.toarray()
        
        if hasattr(voxel, "detach"):
            voxel = voxel.detach().cpu().numpy()
        
        voxel = np.squeeze(voxel)
        
        # 入力データの検証
        if voxel.ndim != 3:
            
            raise ValueError(f"Expected 3D array, got {voxel.ndim}D array")

        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # 空でないボクセルのみを抽出
        valid_mask = (voxel != -1.0)  # 空白でないボクセルのマスク
        valid_indices = np.where(valid_mask)
        valid_values = voxel[valid_mask]
        
        if len(valid_indices[0]) == 0:
            print("Warning: No valid voxels found (all voxels are empty)")
            return

        # 出力ファイルを開く
        with open(filename, "w") as f:
            # 有効なボクセルのみを処理
            for (i, j, k), v in zip(zip(*valid_indices), valid_values):
                v = float(v)
                if v not in COLOR_MAP:
                    continue  # 無効な値はスキップ
                r, g, b = COLOR_MAP[v]
                # .xyz形式: x y z r g b
                # f.write(f"{i} {j} {k} {r} {g} {b}\n")
                f.write(f"{j} {i} {k} {r} {g} {b}\n")


        print(f"Voxel data successfully written to {filename}")

    except Exception as e:
        raise IOError(f"Failed to process voxel data: {str(e)}")

# 使用例
if __name__ == "__main__":
    from my_dataset.svsdataset import SvsDataset
    dataset=SvsDataset(f"/mnt/nas/rmjapan2000/tree/data_dir/train/svs_cgvi_256")
    voxel=dataset[0]
    voxel2xyzfile(voxel)

              
