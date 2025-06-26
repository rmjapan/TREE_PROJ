import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

###############################################################################
# 可視化関連のコード
###############################################################################

def draw_cube(ax, x, y, z, size, color='blue', alpha=0.2):
    # 立方体の8頂点を定義
    r = [0, size]
    points = [
        (x+r[0], y+r[0], z+r[0]),
        (x+r[1], y+r[0], z+r[0]),
        (x+r[1], y+r[1], z+r[0]),
        (x+r[0], y+r[1], z+r[0]),
        (x+r[0], y+r[0], z+r[1]),
        (x+r[1], y+r[0], z+r[1]),
        (x+r[1], y+r[1], z+r[1]),
        (x+r[0], y+r[1], z+r[1]),
    ]
    faces = [
        [points[0], points[1], points[2], points[3]],
        [points[4], points[5], points[6], points[7]],
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[1], points[2], points[6], points[5]],
        [points[4], points[7], points[3], points[0]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='gray', alpha=alpha))

def visualize_octree(node, ax, x=0, y=0, z=0, size=1):
    if node.is_leaf:
        if node.value != -1:  # 空でない葉のみ描画
            draw_cube(ax, x, y, z, size, color='blue', alpha=0.4)
    else:
        half = size / 2
        idx = 0
        for dx in [0, half]:
            for dy in [0, half]:
                for dz in [0, half]:
                    child = node.children[idx]
                    visualize_octree(child, ax, x+dx, y+dy, z+dz, half)
                    idx += 1

def visualize_and_save_octree(tree, filename="octree.png"):
    """オクツリーを可視化して画像として保存する関数"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    visualize_octree(tree, ax)
    plt.savefig(filename)
    plt.close(fig) 
    print(f"オクツリーを{filename}に保存しました")