#!/bin/bash

# パスの設定
CONFIG_PATH="configs/vae_snet_eval.yaml"  # VAE設定ファイルへのパス
CHECKPOINT_PATH="saved_ckpt/vae-ckpt/vae-shapenet-depth-8.pth"  # VAEチェックポイントへのパス

# サンプルデータディレクトリとログディレクトリを作成
EXAMPLE_DATA_DIR="datasets/example_data"
LOG_DIR="tensorboard_logs/graphvae_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXAMPLE_DATA_DIR
mkdir -p $LOG_DIR

# シンプルな球体の点群を作成
python -c "
import numpy as np
import os

# 1000点の球体の点群を作成
n_points = 1000  # メモリ節約のため点数を減らす
phi = np.random.uniform(0, 2*np.pi, n_points)
theta = np.random.uniform(0, np.pi, n_points)

# 球座標からデカルト座標へ変換
x = 0.5 * np.sin(theta) * np.cos(phi)  # 半径0.5にスケーリング
y = 0.5 * np.sin(theta) * np.sin(phi)
z = 0.5 * np.cos(theta)

points = np.stack([x, y, z], axis=1)
# 法線は中心から外向き
normals = points.copy() / np.linalg.norm(points, axis=1, keepdims=True)

os.makedirs('$EXAMPLE_DATA_DIR', exist_ok=True)
np.savez('$EXAMPLE_DATA_DIR/pointcloud.npz', points=points, normals=normals)
print('${n_points}点の球体点群を作成しました')
"

DATA_PATH="$EXAMPLE_DATA_DIR/pointcloud.npz"

echo "設定ファイル: $CONFIG_PATH"
echo "チェックポイント: $CHECKPOINT_PATH"
echo "データパス: $DATA_PATH"
echo "ログディレクトリ: $LOG_DIR"

# CPUを使用するよう指定してメモリ使用量を削減
echo "CPUを使用して可視化を実行中..."
python tools/visualize_graphvae_tensorboard.py \
    --vq_cfg $CONFIG_PATH \
    --vq_ckpt $CHECKPOINT_PATH \
    --data_path $DATA_PATH \
    --log_dir $LOG_DIR \
    --device cpu

echo "可視化が完了しました。以下のコマンドでTensorBoardを起動できます:"
echo "tensorboard --logdir=$LOG_DIR" 