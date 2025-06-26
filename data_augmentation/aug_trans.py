from pyvista import voxelize
import torch
from torch_geometric.datasets import ModelNet
from pytorch3d.transforms import Translate
import os
import sys

import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from utils import pyg_to_voxel
from visualize_func import visualize_with_timeout
import numpy as np
import time
def translate_voxel(voxel, translation_vector):
    """ボクセルデータを指定された平行移動ベクトルで変換する関数"""
    voxel_size = voxel.shape[0]
    if type(voxel) == np.ndarray:
        voxel = torch.from_numpy(voxel)#計算量O(1)
    # 平行移動行列の取得
    translate_transform = Translate(translation_vector)
    translate_matrix = translate_transform.get_matrix().squeeze(0)#計算量O(1)
    print(f"translate_matrix.shape: {translate_matrix.shape}")
    print(f"translate_matrix[0]: {translate_matrix[0]}")
    
    # ボクセルの非ゼロ要素のインデックスを取得
    start_time = time.time()
    x,y,z=torch.meshgrid(
        torch.arange(voxel_size),
        torch.arange(voxel_size),
        torch.arange(voxel_size),
        indexing="ij",
    )
    indices=torch.stack([x,y,z],axis=-1).reshape(-1,3)
    
    # indices = torch.tensor([[x, y, z] for x in range(voxel_size) for y in range(voxel_size) for z in range(voxel_size)], dtype=torch.int)4
    
    end_time = time.time()
    print(f"indicesの生成時間: {end_time - start_time}秒")
    print(f"indices.shape: {indices.shape}")
    print(f"indices[0]: {indices[0]}")
    
    # 同次座標に変換
    ones = torch.ones((indices.shape[0], 1))
    print(f"ones.shape: {ones.shape}")
    print(f"ones[0]: {ones[0]}")
    homogeneous_indices = torch.cat((indices, ones), dim=1)
    print(f"同次変換後のhomogeneous_indices.shape: {homogeneous_indices.shape}")
    print(f"同次変換後のhomogeneous_indices[0]: {homogeneous_indices[0]}")
    # 平行移動の適用
    transformed_indices = (homogeneous_indices @ translate_matrix)[:, :3]
    print(f"transformed_indices.shape: {transformed_indices.shape}")
    print(f"transformed_indices[0]: {transformed_indices[0]}")
    # print(transformed_indices.shape)

    
    # 有効な範囲内のインデックスのみを保持
    valid_mask = (
        (transformed_indices[:, 0] >= 0) & (transformed_indices[:, 0] < voxel_size) &
        (transformed_indices[:, 1] >= 0) & (transformed_indices[:, 1] < voxel_size) &
        (transformed_indices[:, 2] >= 0) & (transformed_indices[:, 2] < voxel_size)
    )
    # print(valid_mask[0])
    # 新しいボクセルデータの作成
    valid_indices = transformed_indices[valid_mask].round().long()
    # print(valid_indices.shape)
    # print(type(valid_indices))
    translated_voxel = torch.zeros_like(voxel,dtype=torch.float32)
    translated_voxel[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = voxel[indices[valid_mask][:,0],indices[valid_mask][:,1],indices[valid_mask][:,2]]
    
    return translated_voxel

def main():
    # ModelNetデータセットの読み込み
    modelnet_10 = ModelNet("mydata1", name="10", train=True)
    
    # メッシュデータの取得とボクセル化
    mesh = modelnet_10[0]
    voxel_size = 32
    voxel = torch.from_numpy(pyg_to_voxel(mesh, voxel_size))
    
    # 平行移動量の設定
    translation = torch.tensor([[4,4,4]])
    
    # 平行移動の適用
    translated_voxel = translate_voxel(voxel, translation)
    
    # 元のボクセルデータの可視化
    visualize_with_timeout(voxel.numpy())
    
    # 平行移動後のボクセルデータの可視化
    visualize_with_timeout(translated_voxel.numpy())

def test_numpy():
    voxel_size=256
    x, y, z = np.meshgrid(
        np.arange(voxel_size),
        np.arange(voxel_size),
        np.arange(voxel_size),
        indexing='ij'
    )
    index=np.stack([x,y,z],axis=-1).reshape(-1,3)
    print(index[0])
    
if __name__ == "__main__":
    test_numpy()
    voxel=np.random.randint(0,2,(256,256,256))
    translation=np.array([[4,4,4]])
    translation=torch.from_numpy(translation)
    translated_voxel=translate_voxel(voxel,translation)
