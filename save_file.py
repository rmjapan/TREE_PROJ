from scipy.sparse import csr_matrix, save_npz
import os
import numpy as np

def save_npzForm(voxel_data,path,index):
    """
    -1,0,0.5,1の値->0,1,1.5,2に変換して疎行列として保存
    """
    #deepcopyする
    voxel_2d_flat=voxel_data.copy().reshape(voxel_data.shape[0],-1)
    print(f"シフト開始")
    voxel_2d_flat[voxel_2d_flat==1]=2
    voxel_2d_flat[voxel_2d_flat==0.5]=1.5
    voxel_2d_flat[voxel_2d_flat==0]=1
    voxel_2d_flat[voxel_2d_flat==-1]=0
    print(f"シフト完了")
    voxel_2d_flat_sparse=csr_matrix(voxel_2d_flat)
    save_path=path+f"/svs_{index}.npz"
    save_npz(save_path,voxel_2d_flat_sparse)
    
import os
import shutil
from pathlib import Path



def rename_and_move_files(folder_name, base_dir, start_indices):
    folder_path = base_dir / folder_name
    start_idx = start_indices[folder_name]
    
    # フォルダー内のすべての.npzファイルを取得
    npz_files = sorted(folder_path.glob('svs_*.npz'), 
                      key=lambda x: int(x.stem.split('_')[1]))
    
    for i, file_path in enumerate(npz_files):
        new_index = start_idx + i
        new_name = f'svs_{new_index}.npz'
        new_path = base_dir / new_name
        
        # ファイルを新しい名前で移動
        shutil.move(str(file_path), str(new_path))
        
        # 進捗を表示（1000ファイルごと）
        if (i + 1) % 1000 == 0:
            print(f'{folder_name}: {i + 1} files processed')

def main():
    # メイン処理
    # 元のディレクトリと移動先のディレクトリを指定
    base_dir = Path('/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2')
    folders = ['AcaciaClustered', 'BirchClustered', 'MapleClustered', 'OakClustered', 'PineClustered']

    # 各フォルダーの開始インデックス
    start_indices = {
        'AcaciaClustered': 1,
        'BirchClustered': 30001,
        'MapleClustered': 60001,
        'OakClustered': 90001,
        'PineClustered': 120001
    }
    for folder in folders:
        print(f'Processing {folder}...')
        rename_and_move_files(folder, base_dir, start_indices)
        print(f'Completed processing {folder}')
if __name__ == '__main__':
    main()