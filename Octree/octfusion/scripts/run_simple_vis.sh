#!/bin/bash

# GraphVAEのシンプルな可視化を実行するスクリプト
# パラメータ設定
DEPTH=6
FULL_DEPTH=2
DEPTH_STOP=4
DEPTH_OUT=6
CODE_CHANNEL=16

# 出力ディレクトリの設定
if [ -z "$OUTPUT_DIR" ]; then
    # 環境変数が設定されていない場合はデフォルト値を使用
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="visualization_results/simple_vis_${TIMESTAMP}"
fi

# 出力ディレクトリの作成
mkdir -p $OUTPUT_DIR

echo "GraphVAEのシンプル可視化を開始します..."
echo "出力ディレクトリ: $OUTPUT_DIR"

# 環境変数設定
export PYTHONPATH=$PYTHONPATH:$(pwd)

# ダミー入力データの生成
echo "ダミーオクトリーデータを生成しています..."
python -c "import torch; import numpy as np; np.save('$OUTPUT_DIR/dummy_octree.npy', torch.zeros((1, 8, 8, 8, 4)).numpy())" 

# GraphVAEの可視化図を生成
echo "GraphVAEの処理フロー図を生成しています..."
cat > $OUTPUT_DIR/graphvae_flow.txt << EOL
+-------------------------+
|    入力オクトリー       |
| (深さ$DEPTH, 特徴量4ch)  |
+-------------------------+
           ↓
+-------------------------+
|      エンコーダー       |
| (深さ$DEPTH → $DEPTH_STOP) |
+-------------------------+
           ↓
+-------------------------+
|      潜在空間          |
|  (${CODE_CHANNEL}ch, KL正則化)   |
+-------------------------+
           ↓
+-------------------------+
|      デコーダー        |
| (深さ$DEPTH_STOP → $DEPTH_OUT) |
+-------------------------+
           ↓
+-------------------------+
|    再構成オクトリー     |
|    (深さ$DEPTH_OUT)      |
+-------------------------+
           ↓
+-------------------------+
|      神経MPU           |
|   (連続場の補間)        |
+-------------------------+
           ↓
+-------------------------+
|       SDF値           |
| (任意点の符号付距離場)   |
+-------------------------+
EOL

# モデル構成のサマリーを生成
echo "モデル構成サマリーを生成しています..."
cat > $OUTPUT_DIR/model_summary.txt << EOL
# GraphVAE モデル構成

## 基本パラメータ
- 深さ (depth): $DEPTH
- フル深さ (full_depth): $FULL_DEPTH
- 停止深さ (depth_stop): $DEPTH_STOP
- 出力深さ (depth_out): $DEPTH_OUT
- コードチャネル: $CODE_CHANNEL

## エンコーダー構造
- 入力: オクトリー (深さ$DEPTH, 4チャネル)
- 処理: グラフCNN, ResBlocks, ダウンサンプリング
- 出力: 特徴マップ (深さ$DEPTH_STOP, 512チャネル)

## 潜在表現
- 形状: [$CODE_CHANNEL, 8, 8, 8]
- 分布: 対角ガウス分布 N(μ, σ²)
- 正則化: KLダイバージェンス

## デコーダー構造
- 入力: 潜在コード (${CODE_CHANNEL}チャネル)
- 処理: グラフCNN, ResBlocks, アップサンプリング
- 出力: 再構成オクトリー (深さ$DEPTH_OUT)
  - 分割確率 (2チャネル)
  - SDFパラメータ (4チャネル)

## 神経MPU (Neural MPU)
- 入力: 3D位置 + 再構築オクトリー
- 処理: 多レベル分割統一補間
- 出力: 任意点のSDF値
EOL

# ダイアグラムを画像化（ASCII artをテキストファイルとして保存）
echo "ダイアグラムを生成しています..."
cat > $OUTPUT_DIR/simple_diagram.txt << EOL
           [入力オクトリー]
                  |
                  v
          [グラフ畳み込み層]
                  |
                  v
    +-------------+-------------+
    |                           |
    v                           v
[エンコーダ]               [スキップ接続]
    |                           |
    v                           |
[μ,σ生成層]                     |
    |                           |
    v                           |
[サンプリング]                  |
    |                           |
    v                           |
[デコーダ] <-------------------+
    |
    v
[分割予測 + SDF予測]
    |
    v
[再構成オクトリー]
    |
    v
[神経MPU]
    |
    v
 [SDF値]
EOL

echo "入出力データ形状を記録しています..."
cat > $OUTPUT_DIR/data_shapes.txt << EOL
# GraphVAEのデータ形状

## 入力データ
- オクトリー: 階層的構造
  - ノード数: 8^1 (深さ1) → 8^$DEPTH (深さ$DEPTH)
  - 特徴量: 4 (xyz + 特徴)

## 中間表現
- エンコーダ出力: [512, 8, 8, 8]
- 潜在変数 μ: [$CODE_CHANNEL, 8, 8, 8]
- 潜在変数 σ: [$CODE_CHANNEL, 8, 8, 8]
- サンプリング z: [$CODE_CHANNEL, 8, 8, 8]
- デコーダ中間層: [256, 16, 16, 16] → [128, 32, 32, 32] → ...

## 出力データ
- 分割予測: [2, 64, 64, 64] (深さ$DEPTH_OUT)
- SDF予測: [4, 64, 64, 64] (深さ$DEPTH_OUT)
- MPU出力: スカラー値 (任意点のSDF)
EOL

# 簡易的なモデル可視化器を作成
echo "簡易的なモデルビューアを生成しています..."
cat > $OUTPUT_DIR/view_model.html << EOL
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GraphVAE モデル構造</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #333;
        }
        pre {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .section {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GraphVAE モデル構造</h1>
        
        <div class="section">
            <h2>処理フロー</h2>
            <pre id="flow"></pre>
        </div>
        
        <div class="section">
            <h2>モデル構成</h2>
            <pre id="summary"></pre>
        </div>
        
        <div class="section">
            <h2>シンプルダイアグラム</h2>
            <pre id="diagram"></pre>
        </div>
        
        <div class="section">
            <h2>データ形状</h2>
            <pre id="shapes"></pre>
        </div>
    </div>
    
    <script>
        // テキストファイルを読み込んで表示する関数
        async function loadTextFile(url, elementId) {
            try {
                const response = await fetch(url);
                const text = await response.text();
                document.getElementById(elementId).textContent = text;
            } catch (error) {
                console.error('Error loading file:', error);
                document.getElementById(elementId).textContent = 'ファイルの読み込みに失敗しました';
            }
        }
        
        // ページ読み込み時に各ファイルを読み込む
        window.onload = function() {
            loadTextFile('graphvae_flow.txt', 'flow');
            loadTextFile('model_summary.txt', 'summary');
            loadTextFile('simple_diagram.txt', 'diagram');
            loadTextFile('data_shapes.txt', 'shapes');
        };
    </script>
</body>
</html>
EOL

echo "シンプル可視化が完了しました。結果は $OUTPUT_DIR ディレクトリに保存されています。"
echo "ビューアを開くには: ブラウザで $OUTPUT_DIR/view_model.html を開いてください。"

# 可視化結果を表示（オプション）
if command -v display &> /dev/null; then
    echo "生成された画像を表示します..."
    display "$OUTPUT_DIR/01_process_overview.png"
elif command -v xdg-open &> /dev/null; then
    echo "生成された画像を表示します..."
    xdg-open "$OUTPUT_DIR/01_process_overview.png"
else
    echo "結果を表示するには、$OUTPUT_DIR ディレクトリ内のPNGファイルを確認してください。"
fi 