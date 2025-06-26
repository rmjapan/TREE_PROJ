#!/bin/bash

# TensorBoardログディレクトリを作成
LOG_DIR="tensorboard_logs/network_structure_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# パラメータ設定
DEPTH=6
FULL_DEPTH=2
DEPTH_STOP=4
DEPTH_OUT=6
CODE_CHANNEL=16

echo "GraphVAEのネットワーク構造をTensorBoardで可視化します..."
echo "設定:"
echo "  深さ: $DEPTH"
echo "  フル深さ: $FULL_DEPTH"
echo "  停止深さ: $DEPTH_STOP"
echo "  出力深さ: $DEPTH_OUT"
echo "  コードチャネル: $CODE_CHANNEL"

# 環境変数設定
export PYTHONPATH=$PYTHONPATH:$(pwd)

# スクリプト実行（CPUモードで）
python tools/visualize_network_tensorboard.py \
  --log_dir $LOG_DIR \
  --depth $DEPTH \
  --full_depth $FULL_DEPTH \
  --depth_stop $DEPTH_STOP \
  --depth_out $DEPTH_OUT \
  --code_channel $CODE_CHANNEL

echo "可視化が完了しました。TensorBoardを以下のコマンドで起動してください:"
echo "tensorboard --logdir=$LOG_DIR"

# TensorBoardが既に実行中なら新しいタブでブラウザを開く
tensorboard --logdir=$LOG_DIR &
TB_PID=$!

# 3秒待ってからブラウザでTensorBoardを開く
echo "3秒後にブラウザでTensorBoardを開きます..."
sleep 3

# ブラウザを開く（OSによって異なるコマンド）
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:6006/
elif command -v open &> /dev/null; then
    open http://localhost:6006/
else
    echo "ブラウザを手動で開いて、http://localhost:6006/ にアクセスしてください"
fi

# TensorBoardの終了を待つ（Ctrl+Cで終了）
echo "TensorBoardを終了するには Ctrl+C を押してください"
wait $TB_PID 