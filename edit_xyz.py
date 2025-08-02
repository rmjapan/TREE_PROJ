from sympy import O
from utils import voxel2xyzfile
# from my_dataset.svddata_loader import TreeDataLoader
from visualize_func import visualize_with_timeout4voxel


import os
import glob

def convert_npy_to_xyz(input_path, output_path):
    """
    npyファイルをxyzファイルに一括変換する関数
    
    Args:
        input_path (str): npyファイルが格納されているディレクトリのパス
        output_path (str): xyzファイルを出力するディレクトリのパス
    """
    # npyファイルを取得
    npy_files = glob.glob(os.path.join(input_path, "*.npy"))

    # 出力ディレクトリを作成
    os.makedirs(output_path, exist_ok=True)

    # 各npyファイルを処理
    for npy_file in npy_files:
        # ファイル名から拡張子を除いてxyzファイル名を作成
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        xyz_filename = f"{base_name}.xyz"
        xyz_filepath = os.path.join(output_path, xyz_filename)
        
        # npyファイルをxyzファイルに変換
        voxel2xyzfile(npy_file, xyz_filepath)
        print(f"Converted: {npy_file} -> {xyz_filepath}")
def convert_npz_to_xyz(input_path, output_path):
    """
    npzファイルをxyzファイルに一括変換する関数
    
    Args:
        input_path (str): npyファイルが格納されているディレクトリのパス
        output_path (str): xyzファイルを出力するディレクトリのパス
    """
    # npyファイルを取得
    # print(f"input_path: {input_path}")
    # 出力ディレクトリを作成
    os.makedirs(output_path, exist_ok=True)

    #ファイル名から拡張子を除いてxyzファイル名を作成
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    xyz_filename = f"{base_name}.xyz"
    xyz_filepath = os.path.join(output_path, xyz_filename)

    # npzファイルをxyzファイルに変換
    voxel2xyzfile(input_path, xyz_filepath)
    # print(f"Converted: {input_path} -> {xyz_filepath}")

def overview(output_path):

    import numpy as np
    
    # データローダーを作成
    dataloader = TreeDataLoader()
    train_dataloader = dataloader.train_dataloader()
    
    # 出力フォルダを作成
    sketch_folder = os.path.join(output_path, "sketches")
    voxel_folder = os.path.join(output_path, "voxels")
    xyz_folder = os.path.join(output_path, "xyzfiles")
    
    os.makedirs(sketch_folder, exist_ok=True)
    os.makedirs(voxel_folder, exist_ok=True)
    os.makedirs(xyz_folder, exist_ok=True)
    
    # データを取得して保存
    for batch_idx, batch in enumerate(train_dataloader):
        sketch = batch["img"]
        voxel = batch["voxel"]
        
        for i in range(sketch.shape[0]):
            # スケッチ画像を保存 (白黒画像として)
            sketch_image = sketch[i].squeeze(0).detach().cpu().numpy()
            
            # スケッチ画像をグレースケールとして処理
            if len(sketch_image.shape) == 3 and sketch_image.shape[0] == 1:
                # チャンネル次元を除去
                sketch_image = sketch_image.squeeze(0)
            elif len(sketch_image.shape) == 3:
                # RGBの場合はグレースケールに変換
                sketch_image = np.mean(sketch_image, axis=0)
            
            # 0-255の範囲に正規化
            sketch_image = sketch_image * 255.0
            sketch_image = sketch_image.astype(np.uint8)
            
            # ボクセルデータを保存
            voxel_data = voxel[i].detach().cpu().numpy()
            
            # ファイル名を作成
            file_index = batch_idx * sketch.shape[0] + i
            sketch_filename = os.path.join(sketch_folder, f"sketch_{file_index:04d}.png")
            voxel_filename = os.path.join(voxel_folder, f"voxel_{file_index:04d}.npy")
            xyz_filename = os.path.join(xyz_folder, f"voxel_{file_index:04d}.xyz")
            
            # ファイルを保存
            import cv2
            # 白黒画像として保存
            cv2.imwrite(sketch_filename, sketch_image)
            np.save(voxel_filename, voxel_data)
            voxel2xyzfile(voxel_data, xyz_filename)
            
            # ボクセル可視化
            visualize_with_timeout4voxel(voxel_data)
            
            print(f"Saved: sketch={sketch_filename}, voxel={voxel_filename}, xyz={xyz_filename}")
    
    
    

# 実行例
if __name__ == "__main__":
    input_path = "/mnt/nas/rmjapan2000/tree/eval/sixteen_data_Resnet21/voxeldataset"
    output_path = "/mnt/nas/rmjapan2000/tree/eval/sixteen_data_Resnet21/yxzfiles"
   
    # convert_npy_to_xyz(input_path, output_path)   
    
    output_path="/home/ryuichi/tree/TREE_PROJ/vc_overview"
    overview(output_path=output_path)
    
    
