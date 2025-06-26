import numpy as np
from scipy.sparse import csr_matrix
import sys
import torch
import svs 
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from svs import visualize_voxel_data

H,W,D=256,256,256
def npz2dense(npz_data):
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
