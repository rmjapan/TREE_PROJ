
import json
import sys
import torch
from Octree.octree import Octree4voxel

# JSONファイルから親キーと深さを読み込む
with open("octree_data/clicked_key.json", "r") as f:
    data = json.load(f)

parent_key = data["parent_key"]
depth = data["depth"]

# Octreeを初期化して子キーを可視化
octree = Octree4voxel(depth=8, full_depth=3)
octree.visualize_child_keys(parent_key, depth)
print(f"Child keys visualization complete for parent key {parent_key} at depth {depth}")
    