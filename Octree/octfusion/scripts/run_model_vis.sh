#!/bin/bash

# パスの設定
CONFIG_PATH="configs/vae_snet_eval.yaml"  # VAE設定ファイルへのパス
CHECKPOINT_PATH="saved_ckpt/vae-ckpt/vae-shapenet-depth-8.pth"  # VAEチェックポイントへのパス

# ログディレクトリを作成
LOG_DIR="tensorboard_logs/graphvae_model_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "設定ファイル: $CONFIG_PATH"
echo "チェックポイント: $CHECKPOINT_PATH"
echo "ログディレクトリ: $LOG_DIR"

# CPUを使用してモデル構造のみを可視化
echo "CPUを使用してモデル構造を可視化中..."
python tools/visualize_model_only.py \
    --vq_cfg $CONFIG_PATH \
    --vq_ckpt $CHECKPOINT_PATH \
    --log_dir $LOG_DIR \
    --device cpu

echo "可視化が完了しました。以下のコマンドでTensorBoardを起動できます:"
echo "tensorboard --logdir=$LOG_DIR" 