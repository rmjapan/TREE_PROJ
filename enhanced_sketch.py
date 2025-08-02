import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy import ndimage
from skimage import morphology, filters


def make_enhanced_sketch_multicolor(
    voxel_data,
    output_size=(224, 224),
    save_path="enhanced_sketch.png",
    verbose=True
):
    """
    マルチカラーでより詳細なスケッチ画像を生成する関数
    
    Args:
        voxel_data: (H, W, D) ndarray, -1=空白, 0=葉, 0.5=枝, 1=幹
        output_size: tuple, 出力画像サイズ (width, height)
        save_path: str, 保存先ファイル名
        verbose: bool, ログ出力
    """
    # --- 1. 各構造要素を分離 ---
    trunk_mask = (voxel_data == 1.0)  # 幹
    branch_mask = (voxel_data == 0.5)  # 枝
    leaf_mask = (voxel_data == 0.0)    # 葉
    
    # --- 2. Y軸投影で2D画像を作成 ---
    trunk_proj = np.max(trunk_mask.astype(float), axis=0)
    branch_proj = np.max(branch_mask.astype(float), axis=0)
    leaf_proj = np.max(leaf_mask.astype(float), axis=0)
    
    # 上下反転
    trunk_proj = np.flip(trunk_proj, axis=0)
    branch_proj = np.flip(branch_proj, axis=0)
    leaf_proj = np.flip(leaf_proj, axis=0)
    
    # --- 3. 構造強調処理 ---
    # 幹の強調（太く、濃く）
    trunk_enhanced = enhance_trunk_structure(trunk_proj)
    
    # 枝の強調（細い線を太く、接続性を改善）
    branch_enhanced = enhance_branch_structure(branch_proj)
    
    # 葉の強調（クラスター化、視認性向上）
    leaf_enhanced = enhance_leaf_structure(leaf_proj)
    
    # --- 4. カラー合成 ---
    # RGBカラー画像を作成
    height, width = trunk_proj.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 幹: 濃い茶色 (139, 69, 19)
    trunk_color = np.array([139, 69, 19])
    # 枝: 明るい茶色 (160, 82, 45)
    branch_color = np.array([160, 82, 45])
    # 葉: 緑色 (34, 139, 34)
    leaf_color = np.array([34, 139, 34])
    
    # 各要素を合成（重複部分は上位要素が優先）
    for i in range(3):
        color_image[:, :, i] = (
            leaf_enhanced * leaf_color[i] +
            branch_enhanced * branch_color[i] * (1 - leaf_enhanced) +
            trunk_enhanced * trunk_color[i] * (1 - leaf_enhanced) * (1 - branch_enhanced)
        )
    
    # --- 5. エッジ強調とコントラスト改善 ---
    # グレースケール版も作成してエッジを抽出
    gray_combined = np.maximum(np.maximum(trunk_enhanced, branch_enhanced), leaf_enhanced)
    
    # エッジ検出
    edges = cv2.Canny((gray_combined * 255).astype(np.uint8), 50, 150)
    edges_normalized = edges / 255.0
    
    # エッジを黒線として追加
    edge_mask = edges_normalized > 0
    color_image[edge_mask] = [0, 0, 0]  # 黒いエッジ
    
    # --- 6. サイズ調整と保存 ---
    color_image_resized = cv2.resize(color_image, output_size, interpolation=cv2.INTER_CUBIC)
    
    # 保存先ディレクトリの作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # PIL Imageとして保存
    pil_image = Image.fromarray(color_image_resized)
    pil_image.save(save_path, format='PNG')
    
    if verbose:
        print(f"[INFO] Enhanced multicolor sketch saved to {save_path}")
        print(f"[INFO] Image size: {pil_image.size}")
    
    return color_image_resized


def enhance_trunk_structure(trunk_proj):
    """幹構造の強調"""
    if np.sum(trunk_proj) == 0:
        return trunk_proj
    
    # モルフォロジー演算で構造を強化
    trunk_binary = trunk_proj > 0
    
    # クロージング操作で小さな隙間を埋める
    kernel = np.ones((3, 3), np.uint8)
    trunk_closed = cv2.morphologyEx(trunk_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # ディレーション操作で太くする
    kernel_dilate = np.ones((2, 2), np.uint8)
    trunk_dilated = cv2.dilate(trunk_closed, kernel_dilate, iterations=1)
    
    return trunk_dilated.astype(float)


def enhance_branch_structure(branch_proj):
    """枝構造の強調"""
    if np.sum(branch_proj) == 0:
        return branch_proj
    
    # 枝の接続性を改善
    branch_binary = branch_proj > 0
    
    # スケルトン化してから太くする手法
    # まず細線化
    skeleton = morphology.skeletonize(branch_binary)
    
    # スケルトンを少し太くする
    kernel = np.ones((2, 2), np.uint8)
    branch_enhanced = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    
    # ガウシアンフィルタで滑らかにする
    branch_smooth = filters.gaussian(branch_enhanced.astype(float), sigma=0.5)
    
    return np.clip(branch_smooth, 0, 1)


def enhance_leaf_structure(leaf_proj):
    """葉構造の強調"""
    if np.sum(leaf_proj) == 0:
        return leaf_proj
    
    # 葉をクラスター化して視認性を向上
    leaf_binary = leaf_proj > 0
    
    # ガウシアンフィルタでぼかしてクラスター感を出す
    leaf_blurred = filters.gaussian(leaf_binary.astype(float), sigma=1.0)
    
    # 閾値処理で明確な領域に変換
    leaf_enhanced = (leaf_blurred > 0.1).astype(float)
    
    # 小さなノイズを除去
    kernel = np.ones((2, 2), np.uint8)
    leaf_cleaned = cv2.morphologyEx(leaf_enhanced.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    return leaf_cleaned.astype(float)


def make_enhanced_sketch_detailed(
    voxel_data,
    output_size=(224, 224),
    save_path="detailed_sketch.png",
    verbose=True
):
    """
    詳細なモノクロスケッチ画像を生成する関数
    """
    # --- 1. 構造要素の分離と重み付け ---
    trunk_weight = 1.0    # 幹は最も濃く
    branch_weight = 0.7   # 枝は中程度
    leaf_weight = 0.4     # 葉は薄く
    
    # 重み付き密度マップを作成
    density_map = np.zeros_like(voxel_data, dtype=float)
    density_map[voxel_data == 1.0] = trunk_weight    # 幹
    density_map[voxel_data == 0.5] = branch_weight   # 枝
    density_map[voxel_data == 0.0] = leaf_weight     # 葉
    
    # --- 2. Y軸投影（深度情報を考慮）---
    # 単純な最大値投影ではなく、重み付き投影
    projection = np.mean(density_map, axis=0)  # 平均値投影で密度を保持
    projection = np.flip(projection, axis=0)
    
    # --- 3. コントラスト強化 ---
    # ヒストグラム平坦化でコントラストを改善
    projection_uint8 = np.clip(projection * 255, 0, 255).astype(np.uint8)
    equalized = cv2.equalizeHist(projection_uint8)
    
    # --- 4. マルチスケールエッジ検出 ---
    # 異なるスケールでエッジを検出して合成
    edges_fine = cv2.Canny(equalized, 50, 150)      # 細かいエッジ
    edges_coarse = cv2.Canny(equalized, 100, 200)   # 太いエッジ
    
    # エッジを合成
    combined_edges = np.maximum(edges_fine, edges_coarse)
    
    # --- 5. 構造強調 ---
    # モルフォロジー演算で線を強化
    kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    enhanced_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_line)
    
    # --- 6. 最終処理 ---
    # サイズ調整
    final_image = cv2.resize(enhanced_edges, output_size, interpolation=cv2.INTER_CUBIC)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final_image)
    
    if verbose:
        print(f"[INFO] Enhanced detailed sketch saved to {save_path}")
    
    return final_image


def make_enhanced_sketch_directions(
    voxel_data,
    output_dir,
    index,
    enhancement_type="multicolor"
):
    """
    4方向の強化されたスケッチを生成
    
    Args:
        voxel_data: ボクセルデータ
        output_dir: 出力ディレクトリ
        index: ファイルインデックス
        enhancement_type: "multicolor" または "detailed"
    """
    # 各方向の投影
    directions_data = {
        "front": voxel_data,
        "back": np.rot90(voxel_data, k=1, axes=(1, 2)),
        "left": np.rot90(voxel_data, k=2, axes=(1, 2)),
        "right": np.rot90(voxel_data, k=3, axes=(1, 2))
    }
    
    results = {}
    
    for direction, data in directions_data.items():
        dir_path = os.path.join(output_dir, direction)
        os.makedirs(dir_path, exist_ok=True)
        
        if enhancement_type == "multicolor":
            save_path = os.path.join(dir_path, f"enhanced_multicolor_{direction}_{index}.png")
            result = make_enhanced_sketch_multicolor(data, save_path=save_path)
        else:  # detailed
            save_path = os.path.join(dir_path, f"enhanced_detailed_{direction}_{index}.png")
            result = make_enhanced_sketch_detailed(data, save_path=save_path)
        
        results[direction] = result
    
    return results


def compare_sketch_methods(voxel_data, output_dir, index):
    """
    複数のスケッチ手法を比較
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 元の手法
    from svs import make_sketch_canny_edge
    original_path = os.path.join(output_dir, f"original_canny_{index}.png")
    make_sketch_canny_edge(voxel_data, save_path=original_path)
    
    # 新しい手法1: マルチカラー
    multicolor_path = os.path.join(output_dir, f"enhanced_multicolor_{index}.png")
    make_enhanced_sketch_multicolor(voxel_data, save_path=multicolor_path)
    
    # 新しい手法2: 詳細モノクロ
    detailed_path = os.path.join(output_dir, f"enhanced_detailed_{index}.png")
    make_enhanced_sketch_detailed(voxel_data, save_path=detailed_path)
    
    print(f"Comparison sketches saved in {output_dir}")
    return {
        "original": original_path,
        "multicolor": multicolor_path,
        "detailed": detailed_path
    }


if __name__ == "__main__":
    # テスト用のダミーデータ
    test_voxel = np.random.choice([-1, 0, 0.5, 1.0], size=(64, 64, 64), p=[0.8, 0.1, 0.07, 0.03])
    
    # テスト実行
    compare_sketch_methods(test_voxel, "test_sketch_output", 1)
