#!/bin/bash

# GraphVAEの可視化スクリプトを実行するためのメインスクリプト

# カラー出力用の設定
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}GraphVAE可視化スクリプト実行ツール${NC}"
echo "============================================"

# デフォルトパラメータ
DEPTH=6
FULL_DEPTH=2
DEPTH_STOP=4
DEPTH_OUT=6
CODE_CHANNEL=16
LOG_DIR="tensorboard_logs"

# 引数の処理
while [[ $# -gt 0 ]]; do
  case $1 in
    --depth)
      DEPTH="$2"
      shift 2
      ;;
    --full-depth)
      FULL_DEPTH="$2"
      shift 2
      ;;
    --depth-stop)
      DEPTH_STOP="$2"
      shift 2
      ;;
    --depth-out)
      DEPTH_OUT="$2"
      shift 2
      ;;
    --code-channel)
      CODE_CHANNEL="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      echo -e "${YELLOW}警告: 不明な引数 $1${NC}"
      shift
      ;;
  esac
done

# 設定の表示
echo -e "${BLUE}実行設定:${NC}"
echo "- オクトリーの深さ: $DEPTH"
echo "- フル展開の深さ: $FULL_DEPTH"
echo "- エンコーダの停止深さ: $DEPTH_STOP"
echo "- デコーダの出力深さ: $DEPTH_OUT"
echo "- 潜在コードのチャネル数: $CODE_CHANNEL"
echo "- ログディレクトリ: $LOG_DIR"
echo

# 必要なディレクトリの作成
mkdir -p "$LOG_DIR"

# スクリプトのパスを取得 (現在のスクリプトがあるディレクトリ)
SCRIPT_DIR="$(dirname "$0")"

# Pythonスクリプトの実行
echo -e "${GREEN}TensorBoard可視化スクリプトを実行中...${NC}"
python scripts/visualize_network_tensorboard.py \
  --depth "$DEPTH" \
  --full_depth "$FULL_DEPTH" \
  --depth_stop "$DEPTH_STOP" \
  --depth_out "$DEPTH_OUT" \
  --code_channel "$CODE_CHANNEL" \
  --log_dir "$LOG_DIR"

# スクリプト実行結果のチェック
if [ $? -eq 0 ]; then
  echo -e "${GREEN}可視化が正常に完了しました${NC}"
  echo "TensorBoardを起動するには、次のコマンドを実行してください:"
  echo -e "${BLUE}tensorboard --logdir=$LOG_DIR${NC}"
  echo "その後、ブラウザで http://localhost:6006 にアクセスしてください"
else
  echo -e "${YELLOW}可視化スクリプトの実行中にエラーが発生しました${NC}"
fi

# 簡易可視化スクリプトも実行
echo
echo -e "${GREEN}簡易テキスト可視化も実行します...${NC}"

# タイムスタンプ付きの出力ディレクトリを作成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VIS_DIR="visualization_results/simple_vis_${TIMESTAMP}"
mkdir -p "$VIS_DIR"

# CODE_CHANNELの修正
CODE_CHANNEL_FIXED="$CODE_CHANNEL"

# 簡易テキスト可視化ファイルの生成
echo "GraphVAE処理フロー" > "$VIS_DIR/graphvae_flow.txt"
echo "====================" >> "$VIS_DIR/graphvae_flow.txt"
echo "1. 入力: オクトリーデータ (深さ=$DEPTH)" >> "$VIS_DIR/graphvae_flow.txt"
echo "2. エンコーダー: 3D畳み込みネットワーク" >> "$VIS_DIR/graphvae_flow.txt"
echo "3. 潜在空間: ${CODE_CHANNEL_FIXED}チャネル (μとlogvar)" >> "$VIS_DIR/graphvae_flow.txt"
echo "4. サンプリング: z = μ + ε·σ" >> "$VIS_DIR/graphvae_flow.txt"
echo "5. デコーダー: 3D畳み込みネットワーク" >> "$VIS_DIR/graphvae_flow.txt"
echo "6. 出力: 分割予測 + 特徴予測" >> "$VIS_DIR/graphvae_flow.txt"

# モデル概要
echo "GraphVAEモデル概要" > "$VIS_DIR/model_summary.txt"
echo "==================" >> "$VIS_DIR/model_summary.txt"
echo "モデルパラメータ:" >> "$VIS_DIR/model_summary.txt"
echo "- 深さ (depth): $DEPTH" >> "$VIS_DIR/model_summary.txt"
echo "- フル深さ (full_depth): $FULL_DEPTH" >> "$VIS_DIR/model_summary.txt"
echo "- 停止深さ (depth_stop): $DEPTH_STOP" >> "$VIS_DIR/model_summary.txt"
echo "- 出力深さ (depth_out): $DEPTH_OUT" >> "$VIS_DIR/model_summary.txt"
echo "- コードチャネル: $CODE_CHANNEL" >> "$VIS_DIR/model_summary.txt"
echo >> "$VIS_DIR/model_summary.txt"
echo "エンコーダー構造:" >> "$VIS_DIR/model_summary.txt"
echo "- 入力→Conv3D→ReLU→Conv3D→ReLU→...→Conv3D→ReLU" >> "$VIS_DIR/model_summary.txt"
echo "- 最終層: 512チャネル→μ層(${CODE_CHANNEL_FIXED})およびlogvar層(${CODE_CHANNEL_FIXED})" >> "$VIS_DIR/model_summary.txt"
echo >> "$VIS_DIR/model_summary.txt"
echo "潜在表現:" >> "$VIS_DIR/model_summary.txt"
echo "- ${CODE_CHANNEL_FIXED}チャネルの3Dテンソル (8x8x8)" >> "$VIS_DIR/model_summary.txt"
echo "- 再パラメータ化: z = μ + ε·σ (ε ~ N(0,1))" >> "$VIS_DIR/model_summary.txt"
echo >> "$VIS_DIR/model_summary.txt"
echo "デコーダー構造:" >> "$VIS_DIR/model_summary.txt"
echo "- 潜在表現→Conv3D→ReLU→Conv3D→ReLU→...→Conv3D→ReLU" >> "$VIS_DIR/model_summary.txt"
echo "- 最終層: 32チャネル→分割予測(2)および特徴予測(4)" >> "$VIS_DIR/model_summary.txt"

# 簡易ダイアグラム
echo "GraphVAE簡易ダイアグラム" > "$VIS_DIR/simple_diagram.txt"
echo "=======================" >> "$VIS_DIR/simple_diagram.txt"
echo "" >> "$VIS_DIR/simple_diagram.txt"
echo "  入力オクトリー (深さ=$DEPTH)" >> "$VIS_DIR/simple_diagram.txt"
echo "        ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "  ┌───────────────┐" >> "$VIS_DIR/simple_diagram.txt"
echo "  │  エンコーダー  │" >> "$VIS_DIR/simple_diagram.txt"
echo "  │  (Conv3D x5)   │" >> "$VIS_DIR/simple_diagram.txt"
echo "  └───────────────┘" >> "$VIS_DIR/simple_diagram.txt"
echo "        ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "  ┌───────┐   ┌───────┐" >> "$VIS_DIR/simple_diagram.txt"
echo "  │   μ   │   │logvar │" >> "$VIS_DIR/simple_diagram.txt"
echo "  └───────┘   └───────┘" >> "$VIS_DIR/simple_diagram.txt"
echo "       ↓ ↘   ↙ ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "       ↓  ┌─────┐  ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "       ↓  │  z  │  ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "       ↓  └─────┘  ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "        ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "  ┌───────────────┐" >> "$VIS_DIR/simple_diagram.txt"
echo "  │  デコーダー   │" >> "$VIS_DIR/simple_diagram.txt"
echo "  │  (Conv3D x5)  │" >> "$VIS_DIR/simple_diagram.txt"
echo "  └───────────────┘" >> "$VIS_DIR/simple_diagram.txt"
echo "        ↓" >> "$VIS_DIR/simple_diagram.txt"
echo "  ┌───────┐   ┌───────┐" >> "$VIS_DIR/simple_diagram.txt"
echo "  │ 分割  │   │ 特徴  │" >> "$VIS_DIR/simple_diagram.txt"
echo "  │ 予測  │   │ 予測  │" >> "$VIS_DIR/simple_diagram.txt"
echo "  └───────┘   └───────┘" >> "$VIS_DIR/simple_diagram.txt"

# データ形状
echo "GraphVAEデータ形状" > "$VIS_DIR/data_shapes.txt"
echo "=================" >> "$VIS_DIR/data_shapes.txt"
echo "入力オクトリー: [B, 4, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "エンコーダー中間出力:" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv1: [B, 32, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv2: [B, 64, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv3: [B, 128, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv4: [B, 256, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv5: [B, 512, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "μ: [B, $CODE_CHANNEL, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "logvar: [B, $CODE_CHANNEL, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "z: [B, $CODE_CHANNEL, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "デコーダー中間出力:" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv1: [B, 512, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv2: [B, 256, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv3: [B, 128, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv4: [B, 64, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "- Conv5: [B, 32, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "分割予測: [B, 2, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"
echo "特徴予測: [B, 4, 8, 8, 8]" >> "$VIS_DIR/data_shapes.txt"

# シンプルなHTMLビューア
cat > "$VIS_DIR/view_model.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>GraphVAE モデル可視化</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            flex: 1 1 45%;
        }
        pre {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .header {
            background-color: #4285f4;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>GraphVAE モデル可視化</h1>
        <p>生成日時: $(date)</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>処理フロー</h2>
            <pre id="flow"></pre>
        </div>
        
        <div class="card">
            <h2>モデル概要</h2>
            <pre id="summary"></pre>
        </div>
        
        <div class="card">
            <h2>簡易ダイアグラム</h2>
            <pre id="diagram"></pre>
        </div>
        
        <div class="card">
            <h2>データ形状</h2>
            <pre id="shapes"></pre>
        </div>
    </div>

    <script>
        // ファイルを読み込んでプリタグに表示する関数
        async function loadFile(filename, elementId) {
            try {
                const response = await fetch(filename);
                const text = await response.text();
                document.getElementById(elementId).textContent = text;
            } catch (error) {
                console.error('Error loading file:', error);
                document.getElementById(elementId).textContent = 'ファイルの読み込みに失敗しました';
            }
        }
        
        // ファイルの読み込み
        loadFile('graphvae_flow.txt', 'flow');
        loadFile('model_summary.txt', 'summary');
        loadFile('simple_diagram.txt', 'diagram');
        loadFile('data_shapes.txt', 'shapes');
    </script>
</body>
</html>
EOF

echo -e "${GREEN}簡易可視化が完了しました${NC}"
echo "結果は以下のディレクトリに保存されました:"
echo -e "${BLUE}$VIS_DIR${NC}"
echo "HTMLビューアを開くには、以下のファイルをブラウザで開いてください:"
echo -e "${BLUE}$VIS_DIR/view_model.html${NC}"

# 完了メッセージ
echo
echo -e "${GREEN}すべての可視化が完了しました${NC}" 