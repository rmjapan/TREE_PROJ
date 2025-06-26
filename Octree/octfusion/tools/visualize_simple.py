import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

"""
GraphVAEの処理過程を可視化するシンプルなスクリプト
重みを読み込まずに、処理フローを説明するためのモックアップを作成します
"""

def create_sphere_points(n_points=1000, radius=0.5):
    """球体の点群を生成"""
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    points = np.stack([x, y, z], axis=1)
    normals = points.copy() / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points, normals

def visualize_point_cloud(points, title="点群データ", save_path=None):
    """点群の可視化"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=2, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 表示範囲を設定
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    
    # グリッドを表示
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"点群可視化を保存: {save_path}")
    
    return fig, ax

def visualize_octree(depth=6, save_path=None):
    """オクトリー構造の可視化（簡略化）"""
    # 深さごとの色を設定
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    fig = plt.figure(figsize=(15, 10))
    
    # 各レベルを可視化
    for i, d in enumerate(range(2, min(depth+1, 6))):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # この深さでのセルサイズ
        cell_size = 1.0 / (2**d)
        
        # セルの中心位置をランダムに生成（実際のオクトリーの非空ノードを模倣）
        n_cells = min(100, 8**d)  # 表示するセル数を制限
        
        # 各レベルでランダムなセルを生成
        centers = np.random.uniform(-0.5 + cell_size/2, 0.5 - cell_size/2, (n_cells, 3))
        
        # セルを描画（シンプルな立方体として）
        for center in centers:
            # 立方体の頂点を計算
            r = cell_size / 2
            corners = np.array([
                [center[0]-r, center[1]-r, center[2]-r],
                [center[0]+r, center[1]-r, center[2]-r],
                [center[0]+r, center[1]+r, center[2]-r],
                [center[0]-r, center[1]+r, center[2]-r],
                [center[0]-r, center[1]-r, center[2]+r],
                [center[0]+r, center[1]-r, center[2]+r],
                [center[0]+r, center[1]+r, center[2]+r],
                [center[0]-r, center[1]+r, center[2]+r],
            ])
            
            # 立方体の辺を描画
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
                (4, 5), (5, 6), (6, 7), (7, 4),  # 上面
                (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直辺
            ]
            
            for start, end in edges:
                ax.plot3D(
                    [corners[start, 0], corners[end, 0]],
                    [corners[start, 1], corners[end, 1]],
                    [corners[start, 2], corners[end, 2]],
                    color=colors[i % len(colors)], linewidth=0.5, alpha=0.3
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'深さ {d} のオクトリー（{n_cells}セル）')
        
        # 表示範囲を設定
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([-0.6, 0.6])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"オクトリー可視化を保存: {save_path}")
    
    return fig

def visualize_latent_space(save_path=None):
    """潜在空間の可視化"""
    # 潜在コードを模倣（4チャンネル、16x16x16の3Dボリューム）
    n_channels = 4
    spatial_dim = 16
    
    # ランダムな潜在コードを生成
    latent_code = np.random.randn(n_channels, spatial_dim, spatial_dim, spatial_dim)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 各チャネルの中央スライスを可視化
    for c in range(n_channels):
        mid_z = spatial_dim // 2
        slice_data = latent_code[c, :, :, mid_z]
        
        im = axes[c].imshow(slice_data, cmap='viridis')
        axes[c].set_title(f'潜在コード チャンネル {c+1} (z={mid_z})')
        plt.colorbar(im, ax=axes[c])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"潜在空間可視化を保存: {save_path}")
    
    return fig

def visualize_sdf_grid(save_path=None):
    """SDFグリッドの可視化"""
    # 64x64x64のSDFグリッドを模倣
    resolution = 64
    
    # 球体のSDFを生成
    x = np.linspace(-0.9, 0.9, resolution)
    y = np.linspace(-0.9, 0.9, resolution)
    z = np.linspace(-0.9, 0.9, resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 球体のSDF値を計算
    radius = 0.5
    sdf = np.sqrt(X**2 + Y**2 + Z**2) - radius
    
    # スライスを可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mid_x = resolution // 2
    mid_y = resolution // 2
    mid_z = resolution // 2
    
    # 各軸に沿った中央スライスを可視化
    im0 = axes[0].imshow(sdf[mid_x, :, :], cmap='viridis')
    axes[0].set_title(f'YZ平面 (X={mid_x})')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(sdf[:, mid_y, :], cmap='viridis')
    axes[1].set_title(f'XZ平面 (Y={mid_y})')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(sdf[:, :, mid_z], cmap='viridis')
    axes[2].set_title(f'XY平面 (Z={mid_z})')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"SDF可視化を保存: {save_path}")
    
    return fig

def visualize_process_diagram(save_path=None):
    """GraphVAEの処理過程の図解を作成"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 処理段階のラベル
    stages = [
        "入力点群",
        "入力オクトリー\n(深さ6)",
        "エンコード",
        "潜在コード\n(16x16x16x3)",
        "デコード",
        "出力オクトリー\n(深さ8)",
        "SDF計算",
        "メッシュ抽出"
    ]
    
    # 矢印と各段階のボックス位置
    y_pos = 0.5
    box_width = 0.8
    box_height = 0.5
    arrow_length = 0.4
    
    for i, stage in enumerate(stages):
        x_pos = i + 0.5
        
        # ボックスを描画
        rect = plt.Rectangle((x_pos - box_width/2, y_pos - box_height/2), 
                             box_width, box_height, 
                             facecolor='skyblue', edgecolor='black',
                             alpha=0.7, zorder=2)
        ax.add_patch(rect)
        
        # テキストを追加
        ax.text(x_pos, y_pos, stage, ha='center', va='center', 
                fontsize=10, fontweight='bold', zorder=3)
        
        # 矢印を追加（最後の要素には矢印なし）
        if i < len(stages) - 1:
            ax.arrow(x_pos + box_width/2, y_pos,
                     arrow_length, 0,
                     head_width=0.1, head_length=0.1,
                     fc='black', ec='black', zorder=1)
    
    # 詳細を追加
    details = [
        "・点群データから3D形状を表現\n・法線ベクトル付き",
        "・階層的な空間分割\n・深さごとにノード構造化",
        "・グラフ畳み込みで特徴抽出\n・ダウンサンプリング",
        "・確率的エンコーディング\n・KL損失で正則化",
        "・アップサンプリングで形状復元\n・各ノードの属性予測",
        "・階層的に形状情報を保持\n・領域分割情報",
        "・MPU関数で任意点のSDF計算\n・符号付き距離場の表現",
        "・Marching Cubesアルゴリズム\n・3Dメッシュの生成"
    ]
    
    for i, detail in enumerate(details):
        x_pos = i + 0.5
        ax.text(x_pos, y_pos - box_height/2 - 0.15, detail,
                ha='center', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))
    
    # ラベルとタイトル
    ax.set_title('GraphVAE 処理過程の概要', fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"処理過程図解を保存: {save_path}")
    
    return fig

def main():
    # 出力ディレクトリを作成
    output_dir = "visualization_results/simple_vis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("GraphVAEの処理過程の可視化を開始します...")
    
    # 処理の流れを図示
    visualize_process_diagram(save_path=os.path.join(output_dir, "01_process_overview.png"))
    
    # 入力点群の可視化
    points, normals = create_sphere_points(n_points=2000)
    visualize_point_cloud(points, "入力点群データ", 
                         save_path=os.path.join(output_dir, "02_input_pointcloud.png"))
    
    # オクトリー構造の可視化
    visualize_octree(depth=6, save_path=os.path.join(output_dir, "03_octree_structure.png"))
    
    # 潜在空間の可視化
    visualize_latent_space(save_path=os.path.join(output_dir, "04_latent_space.png"))
    
    # SDFグリッドの可視化
    visualize_sdf_grid(save_path=os.path.join(output_dir, "05_sdf_grid.png"))
    
    # 処理が成功したメッセージ
    print(f"可視化が完了しました。結果は {output_dir} に保存されています。")
    
    # 処理説明ファイルを作成
    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("GraphVAE処理過程の説明\n")
        f.write("=====================\n\n")
        
        f.write("1. 入力点群処理\n")
        f.write("   - 3D点群データを入力として受け取る\n")
        f.write("   - 点の座標と法線ベクトルを使用\n")
        f.write("   - スケーリングして[-1, 1]の範囲に正規化\n\n")
        
        f.write("2. オクトリー構築\n")
        f.write("   - 点群から階層的なオクトリー構造を作成\n")
        f.write("   - 通常は深さ6程度まで構築\n")
        f.write("   - 形状の詳細を効率的に表現\n\n")
        
        f.write("3. エンコーディング\n")
        f.write("   - グラフ畳み込みネットワークで特徴抽出\n")
        f.write("   - オクトリーの各レベルで特徴を抽出\n")
        f.write("   - 深さd=6からd=2まで段階的に圧縮\n\n")
        
        f.write("4. 潜在表現\n")
        f.write("   - 変分自己符号化器（VAE）で確率的にエンコード\n")
        f.write("   - 平均と分散パラメータでガウス分布を表現\n")
        f.write("   - KLダイバージェンスで正則化\n")
        f.write("   - 潜在コードは通常16x16x16の3D形状とチャンネル数3\n\n")
        
        f.write("5. デコーディング\n")
        f.write("   - 潜在コードからオクトリーを再構築\n")
        f.write("   - 深さ2から開始し、段階的に深さ8まで拡張\n")
        f.write("   - 各ノードの分割確率と特徴値を予測\n\n")
        
        f.write("6. 神経MPU（Neural MPU）\n")
        f.write("   - 再構築されたオクトリーから任意の3D位置のSDF値を計算\n")
        f.write("   - 多レベル分割統一（Multi-level Partition of Unity）手法\n")
        f.write("   - 各ノードの寄与を重み付けして補間\n\n")
        
        f.write("7. メッシュ抽出\n")
        f.write("   - SDFからMarching Cubesアルゴリズムでメッシュを生成\n")
        f.write("   - レベルセット値0の等値面を抽出\n")
        f.write("   - 最終的な3Dメッシュを出力\n\n")
    
    print(f"説明ファイルを作成しました: {os.path.join(output_dir, 'README.txt')}")


if __name__ == "__main__":
    main() 