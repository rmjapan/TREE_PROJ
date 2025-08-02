#!/usr/bin/env python3
"""
最適投影角度選択とスケッチ生成の改良版
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

def find_best_projection_angle(voxel_data, num_angles=8, verbose=False):
    """
    最も印象的な投影角度を見つける関数
    """
    min_val = voxel_data.min()
    structure_mask = voxel_data > min_val
    
    best_score = 0
    best_angle = 0
    best_projection = None
    
    # 360度を等分した角度でテスト
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    for i, angle in enumerate(angles):
        # 回転行列を作成（Y軸周りの回転）
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Y軸方向に投影（真横から見た図）
        h, w, d = voxel_data.shape
        projection = np.zeros((h, d))
        
        # 回転投影
        center_x, center_z = w // 2, d // 2
        for y in range(h):
            for x in range(w):
                for z in range(d):
                    if structure_mask[y, x, z]:
                        # 回転変換
                        x_centered = x - center_x
                        z_centered = z - center_z
                        x_rot = x_centered * cos_a - z_centered * sin_a + center_x
                        z_rot = x_centered * sin_a + z_centered * cos_a + center_z
                        
                        # 投影先の座標
                        z_proj = int(round(z_rot))
                        if 0 <= z_proj < d:
                            projection[y, z_proj] = max(projection[y, z_proj], voxel_data[y, x, z])
        
        # 上下反転
        projection = np.flip(projection, axis=0)
        
        # 投影の品質を評価
        score = evaluate_projection_quality(projection, verbose and i == 0)
        
        if verbose:
            print(f"角度 {np.degrees(angle):.1f}度: スコア = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_angle = angle
            best_projection = projection.copy()
    
    if verbose:
        print(f"最適角度: {np.degrees(best_angle):.1f}度 (スコア: {best_score:.3f})")
    
    return best_angle, best_projection, best_score

def evaluate_projection_quality(projection, verbose=False):
    """
    投影の品質を評価する関数
    """
    if np.sum(projection > 0) == 0:
        return 0.0
    
    # バイナリマスクを作成
    binary_proj = (projection > 0).astype(np.uint8)
    
    # 1. 構造の複雑さ（エッジの量）
    edges = cv2.Canny(binary_proj * 255, 50, 150)
    edge_score = np.sum(edges > 0) / (binary_proj.shape[0] * binary_proj.shape[1])
    
    # 2. 垂直方向の広がり（樹高の印象）
    vertical_spread = 0
    if np.sum(binary_proj) > 0:
        non_zero_rows = np.any(binary_proj > 0, axis=1)
        if np.sum(non_zero_rows) > 0:
            first_row = np.argmax(non_zero_rows)
            last_row = len(non_zero_rows) - 1 - np.argmax(non_zero_rows[::-1])
            vertical_spread = (last_row - first_row) / binary_proj.shape[0]
    
    # 3. 水平方向の分散（枝の広がり）
    horizontal_spread = 0
    if np.sum(binary_proj) > 0:
        non_zero_cols = np.any(binary_proj > 0, axis=0)
        if np.sum(non_zero_cols) > 0:
            first_col = np.argmax(non_zero_cols)
            last_col = len(non_zero_cols) - 1 - np.argmax(non_zero_cols[::-1])
            horizontal_spread = (last_col - first_col) / binary_proj.shape[1]
    
    # 4. 構造の密度バランス
    density = np.sum(binary_proj > 0) / (binary_proj.shape[0] * binary_proj.shape[1])
    density_score = min(density * 10, 1.0)
    
    # 5. アスペクト比（樹木らしい縦長の形状）
    aspect_ratio_score = 0
    if horizontal_spread > 0:
        aspect_ratio = vertical_spread / horizontal_spread
        if 1.5 <= aspect_ratio <= 3.0:
            aspect_ratio_score = 1.0
        else:
            aspect_ratio_score = max(0, 1.0 - abs(aspect_ratio - 2.25) / 2.25)
    
    # 重み付き総合スコア
    total_score = (
        edge_score * 0.3 +           # エッジの複雑さ
        vertical_spread * 0.2 +      # 垂直方向の広がり
        horizontal_spread * 0.2 +    # 水平方向の広がり
        density_score * 0.15 +       # 密度
        aspect_ratio_score * 0.15    # アスペクト比
    )
    
    if verbose:
        print(f"  エッジスコア: {edge_score:.3f}")
        print(f"  垂直広がり: {vertical_spread:.3f}")
        print(f"  水平広がり: {horizontal_spread:.3f}")
        print(f"  密度スコア: {density_score:.3f}")
        print(f"  アスペクト比スコア: {aspect_ratio_score:.3f}")
        print(f"  総合スコア: {total_score:.3f}")
    
    return total_score

def create_optimal_sketch(voxel_data, output_size=(224, 224), save_path="optimal_sketch.png", verbose=True):
    """
    最適角度での改良スケッチを生成
    """
    # 最適角度を探索
    best_angle, best_projection, best_score = find_best_projection_angle(
        voxel_data, num_angles=16, verbose=verbose
    )
    
    # データの値を分析
    unique_vals = np.unique(voxel_data)
    min_val = voxel_data.min()
    max_val = voxel_data.max()
    
    if verbose:
        print(f"[DEBUG] Unique values: {unique_vals}")
        print(f"[DEBUG] Best angle: {np.degrees(best_angle):.1f}度")
        print(f"[DEBUG] Quality score: {best_score:.3f}")
    
    # 最適角度での構造別投影を実行
    trunk_proj, branch_proj, leaf_proj = get_structure_projections(
        voxel_data, best_angle, unique_vals, min_val, max_val
    )
    
    # 画像生成
    color_image = create_enhanced_color_image(trunk_proj, branch_proj, leaf_proj, verbose)
    
    # サイズ調整
    color_image_resized = cv2.resize(color_image, output_size, interpolation=cv2.INTER_CUBIC)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_image = Image.fromarray(color_image_resized)
    pil_image.save(save_path, format='PNG')
    
    if verbose:
        print(f"[INFO] Optimal sketch saved to {save_path}")
        print(f"[INFO] Best angle: {np.degrees(best_angle):.1f} degrees")
    
    return color_image_resized, best_angle

def get_structure_projections(voxel_data, angle, unique_vals, min_val, max_val):
    """
    指定角度での構造別投影を実行
    """
    h, w, d = voxel_data.shape
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    center_x, center_z = w // 2, d // 2
    
    # 構造要素のマスクを作成
    empty_mask = (voxel_data <= min_val)
    
    if len(unique_vals) <= 4:
        sorted_vals = np.sort(unique_vals)
        non_empty_vals = [val for val in sorted_vals if val > min_val or (min_val >= 0 and val >= 1)]
        
        if len(non_empty_vals) >= 3:
            trunk_mask = (voxel_data == non_empty_vals[-1])  # 最高値を幹
            branch_mask = (voxel_data == non_empty_vals[-2])  # 2番目を枝
            leaf_mask = (voxel_data == non_empty_vals[0])    # 最低値を葉
        elif len(non_empty_vals) == 2:
            trunk_mask = (voxel_data == non_empty_vals[-1])
            branch_mask = np.zeros_like(voxel_data, dtype=bool)
            leaf_mask = (voxel_data == non_empty_vals[0])
        else:
            trunk_mask = (voxel_data == max_val) & ~empty_mask
            branch_mask = np.zeros_like(voxel_data, dtype=bool)
            leaf_mask = np.zeros_like(voxel_data, dtype=bool)
    else:
        trunk_mask = (voxel_data >= 0.9) & ~empty_mask
        branch_mask = (voxel_data >= 0.3) & (voxel_data < 0.9) & ~empty_mask
        leaf_mask = (voxel_data >= 0) & (voxel_data < 0.3) & ~empty_mask
    
    # 回転投影
    trunk_proj = np.zeros((h, d))
    branch_proj = np.zeros((h, d))
    leaf_proj = np.zeros((h, d))
    
    for y in range(h):
        for x in range(w):
            for z in range(d):
                # 回転変換
                x_centered = x - center_x
                z_centered = z - center_z
                x_rot = x_centered * cos_a - z_centered * sin_a + center_x
                z_rot = x_centered * sin_a + z_centered * cos_a + center_z
                
                z_proj = int(round(z_rot))
                
                if 0 <= z_proj < d:
                    if trunk_mask[y, x, z]:
                        trunk_proj[y, z_proj] = max(trunk_proj[y, z_proj], voxel_data[y, x, z])
                    elif branch_mask[y, x, z]:
                        branch_proj[y, z_proj] = max(branch_proj[y, z_proj], voxel_data[y, x, z])
                    elif leaf_mask[y, x, z]:
                        leaf_proj[y, z_proj] = max(leaf_proj[y, z_proj], voxel_data[y, x, z])
    
    # 上下反転
    trunk_proj = np.flip(trunk_proj, axis=0)
    branch_proj = np.flip(branch_proj, axis=0)
    leaf_proj = np.flip(leaf_proj, axis=0)
    
    return trunk_proj, branch_proj, leaf_proj

def create_enhanced_color_image(trunk_proj, branch_proj, leaf_proj, verbose=False):
    """
    改良されたカラー画像を生成
    """
    height, width = trunk_proj.shape
    
    # 構造強調処理
    trunk_enhanced = enhance_structure(trunk_proj, kernel_size=(4, 4), iterations=2)
    branch_enhanced = enhance_structure(branch_proj, kernel_size=(2, 2), iterations=1)
    leaf_enhanced = enhance_structure(leaf_proj, kernel_size=(3, 3), iterations=1, use_blur=True)
    
    # 枝を幹より優先表示
    if np.sum(branch_enhanced) > 0 and np.sum(trunk_enhanced) > 0:
        branch_priority_mask = branch_enhanced > 0
        trunk_enhanced = trunk_enhanced & ~branch_priority_mask
    
    # カラー画像作成
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 色の定義
    trunk_color = np.array([101, 67, 33])     # 濃い茶色
    branch_color = np.array([160, 100, 50])   # 明るい茶色
    leaf_color = np.array([50, 150, 50])      # 緑色
    background_color = np.array([245, 245, 245])  # 薄いグレー
    
    # 背景
    color_image[:, :] = background_color
    
    # 構造がある部分は白背景
    structure_mask = (trunk_enhanced > 0) | (branch_enhanced > 0) | (leaf_enhanced > 0)
    color_image[structure_mask] = [255, 255, 255]
    
    # 各構造要素を描画
    leaf_mask_final = leaf_enhanced > 0
    branch_mask_final = branch_enhanced > 0
    trunk_mask_final = trunk_enhanced > 0
    
    color_image[leaf_mask_final] = leaf_color
    color_image[branch_mask_final] = branch_color
    color_image[trunk_mask_final] = trunk_color
    
    # エッジ強調
    edges = create_combined_edges(trunk_enhanced, branch_enhanced, leaf_enhanced)
    color_image[edges > 0] = [0, 0, 0]
    
    if verbose:
        print(f"[DEBUG] Image created: {height}x{width}")
        print(f"[DEBUG] Trunk pixels: {np.sum(trunk_mask_final)}")
        print(f"[DEBUG] Branch pixels: {np.sum(branch_mask_final)}")
        print(f"[DEBUG] Leaf pixels: {np.sum(leaf_mask_final)}")
    
    return color_image

def enhance_structure(projection, kernel_size=(3, 3), iterations=1, use_blur=False):
    """
    構造を強調する
    """
    if np.sum(projection) == 0:
        return projection.astype(np.uint8)
    
    binary = (projection > 0).astype(np.uint8)
    
    if use_blur:
        # 葉の場合はブラーでクラスター感を出す
        blurred = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 1.0)
        enhanced = (blurred > 0.3).astype(np.uint8)
    else:
        # 幹・枝の場合は膨張処理
        kernel = np.ones(kernel_size, np.uint8)
        enhanced = cv2.dilate(binary, kernel, iterations=iterations)
    
    # モルフォロジー演算でノイズ除去
    kernel_clean = np.ones((2, 2), np.uint8)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_clean)
    
    return enhanced

def create_combined_edges(trunk_enhanced, branch_enhanced, leaf_enhanced):
    """
    統合されたエッジを作成
    """
    edges_combined = np.zeros_like(trunk_enhanced, dtype=np.uint8)
    
    if np.sum(trunk_enhanced) > 0:
        edges_trunk = cv2.Canny(trunk_enhanced * 255, 30, 100)
        edges_combined = np.maximum(edges_combined, edges_trunk)
    
    if np.sum(branch_enhanced) > 0:
        edges_branch = cv2.Canny(branch_enhanced * 255, 20, 80)
        edges_combined = np.maximum(edges_combined, edges_branch)
    
    if np.sum(leaf_enhanced) > 0:
        edges_leaf = cv2.Canny(leaf_enhanced * 255, 10, 60)
        edges_combined = np.maximum(edges_combined, edges_leaf)
    
    return edges_combined


if __name__ == "__main__":
    print("=== 最適投影角度スケッチ生成テスト ===")
    
    # テスト用の複雑な樹木データを作成
    def create_complex_tree_voxel(size=64):
        voxel_data = np.zeros((size, size, size))
        
        # 幹（中央に垂直、若干傾斜）
        trunk_x = size // 2
        trunk_y = size // 2
        for z in range(size // 4, size):
            thickness = max(1, 4 - (z - size//4) // 8)
            offset = (z - size//4) // 8
            voxel_data[trunk_x-thickness:trunk_x+thickness+1, 
                      trunk_y-thickness+offset:trunk_y+thickness+1+offset, z] = 2.0
        
        # 主要な枝（複数レベル）
        branch_levels = [size//2, size*2//3, size*3//4]
        for level in branch_levels:
            level_thickness = max(1, 3 - (level - size//2) // 8)
            
            # 東西南北の6方向に枝
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)]
            for dx, dy in directions:
                branch_length = np.random.randint(8, 15)
                for i in range(1, branch_length):
                    x = trunk_x + dx * i
                    y = trunk_y + dy * i
                    if 0 <= x < size and 0 <= y < size:
                        voxel_data[x-level_thickness:x+level_thickness+1, 
                                  y-level_thickness:y+level_thickness+1, 
                                  level:level+2] = 1.5
        
        # 細い枝と葉
        for level in branch_levels:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for radius in range(8, 18, 3):
                    x = int(trunk_x + radius * np.cos(angle))
                    y = int(trunk_y + radius * np.sin(angle))
                    if 0 <= x < size-2 and 0 <= y < size-2:
                        voxel_data[x:x+2, y:y+2, level:level+3] = 1.5
                        
                        # 葉のクラスター
                        for leaf_offset in range(-2, 3):
                            leaf_x = max(0, min(size-3, x + leaf_offset))
                            leaf_y = max(0, min(size-3, y + leaf_offset))
                            leaf_z = max(0, min(size-3, level + leaf_offset))
                            voxel_data[leaf_x:leaf_x+3, leaf_y:leaf_y+3, leaf_z:leaf_z+3] = 1.0
        
        return voxel_data
    
    # テストデータ作成
    test_voxel = create_complex_tree_voxel(64)
    print(f"テストデータのユニーク値: {np.unique(test_voxel)}")
    
    # 出力ディレクトリ作成
    output_dir = "optimal_sketch_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 最適角度でのスケッチ生成
    sketch_path = os.path.join(output_dir, "optimal_angle_sketch.png")
    
    try:
        result_image, best_angle = create_optimal_sketch(
            test_voxel,
            output_size=(256, 256),
            save_path=sketch_path,
            verbose=True
        )
        
        print(f"\n=== テスト完了 ===")
        print(f"最適角度: {np.degrees(best_angle):.1f}度")
        print(f"スケッチ保存先: {sketch_path}")
        
        # 比較用に複数角度でのスケッチも生成
        test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        angle_names = ["0度", "45度", "90度", "135度"]
        
        for angle, name in zip(test_angles, angle_names):
            comp_path = os.path.join(output_dir, f"comparison_{name}.png")
            trunk_proj, branch_proj, leaf_proj = get_structure_projections(
                test_voxel, angle, np.unique(test_voxel), test_voxel.min(), test_voxel.max()
            )
            color_image = create_enhanced_color_image(trunk_proj, branch_proj, leaf_proj)
            color_image_resized = cv2.resize(color_image, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            pil_image = Image.fromarray(color_image_resized)
            pil_image.save(comp_path, format='PNG')
            print(f"比較用スケッチ生成: {comp_path}")
        
        print(f"\n各角度での結果を '{output_dir}' で比較してください")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
