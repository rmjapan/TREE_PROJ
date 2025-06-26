import torch
from torch_geometric.datasets import ModelNet
from pytorch3d.transforms import Translate
from p2v import pyg_to_voxel
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from visualize_func import visualize_with_timeout
def translate_voxel(voxel, translation_vector):
    """ボクセルデータを指定された平行移動ベクトルで変換する関数"""
    voxel_size = voxel.shape[0]
    
    # 平行移動行列の取得
    translate_transform = Translate(translation_vector)
    translate_matrix = translate_transform.get_matrix().squeeze(0)
    # print(translate_matrix)
    
    # ボクセルの非ゼロ要素のインデックスを取得

    indices = torch.tensor([[x, y, z] for x in range(voxel_size) for y in range(voxel_size) for z in range(voxel_size)], dtype=torch.int)
    # print(indices.shape)
    
    # 同次座標に変換
    ones = torch.ones((indices.shape[0], 1))
    homogeneous_indices = torch.cat((indices, ones), dim=1)
    
    # 平行移動の適用
    transformed_indices = (homogeneous_indices @ translate_matrix)[:, :3]
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

if __name__ == "__main__":
    main()
