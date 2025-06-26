import os
import argparse

"""
GraphVAEのネットワーク構造をHTMLファイルとして視覚化するスクリプト
"""

def generate_html_visualization(output_path, depth=6, full_depth=2, depth_stop=4, depth_out=6, code_channel=16):
    """
    GraphVAEネットワーク構造をHTMLファイルとして生成
    """
    
    # HTMLのヘッダー部分
    html_header = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>GraphVAE ネットワーク構造</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #333;
            }
            .module {
                border: 1px solid #ddd;
                margin: 15px 0;
                padding: 15px;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .arrow {
                text-align: center;
                font-size: 24px;
                margin: 10px 0;
                color: #666;
            }
            .params {
                background-color: #e9f7fe;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .network-diagram {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .module-box {
                width: 80%;
                padding: 10px;
                margin: 5px 0;
                border: 1px solid #aaa;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            .encoder {
                background-color: #e1f5fe;
            }
            .latent {
                background-color: #fff9c4;
            }
            .decoder {
                background-color: #e8f5e9;
            }
            .mpu {
                background-color: #f8bbd0;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .description {
                margin: 10px 0;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GraphVAE ネットワーク構造の可視化</h1>
    """
    
    # パラメータ情報
    params_section = f"""
            <div class="params">
                <h2>モデルパラメータ</h2>
                <ul>
                    <li><strong>深さ (depth):</strong> {depth}</li>
                    <li><strong>フル深さ (full_depth):</strong> {full_depth}</li>
                    <li><strong>停止深さ (depth_stop):</strong> {depth_stop}</li>
                    <li><strong>出力深さ (depth_out):</strong> {depth_out}</li>
                    <li><strong>コードチャネル:</strong> {code_channel}</li>
                    <li><strong>入力チャネル:</strong> 4 (x, y, z, 特徴量)</li>
                    <li><strong>出力チャネル:</strong> 4 (SDF, グラディエント x3)</li>
                </ul>
            </div>
    """
    
    # オクトリー説明
    octree_section = """
            <div class="module">
                <h2>オクトリー構造</h2>
                <div class="description">
                    <p>オクトリーは3D空間を階層的に分割する木構造です。各ノードは8つの子ノードに分割され、深さが増すごとに解像度が向上します。</p>
                    <p>GraphVAEでは、深さ6のオクトリーが一般的に使用され、各ノードは空か非空かの情報と、非空ノードには特徴ベクトルが格納されます。</p>
                </div>
                <pre>
深さ1: 8ノード (2x2x2)
深さ2: 64ノード (4x4x4)
深さ3: 512ノード (8x8x8)
深さ4: 4,096ノード (16x16x16)
深さ5: 32,768ノード (32x32x32)
深さ6: 262,144ノード (64x64x64)
                </pre>
            </div>
    """
    
    # 全体的なネットワーク図
    network_diagram = f"""
            <h2>ネットワーク全体構造</h2>
            <div class="network-diagram">
                <div class="module-box encoder">入力: オクトリー (深さ{depth})</div>
                <div class="arrow">↓</div>
                <div class="module-box encoder">エンコーダ (グラフCNN)</div>
                <div class="arrow">↓</div>
                <div class="module-box latent">潜在表現 (VAE, {code_channel}チャネル)</div>
                <div class="arrow">↓</div>
                <div class="module-box decoder">デコーダ (グラフCNN)</div>
                <div class="arrow">↓</div>
                <div class="module-box decoder">出力: 再構成オクトリー (深さ{depth_out})</div>
                <div class="arrow">↓</div>
                <div class="module-box mpu">神経MPU (Neural MPU)</div>
                <div class="arrow">↓</div>
                <div class="module-box mpu">出力: SDF値</div>
            </div>
    """
    
    # エンコーダセクション
    encoder_section = f"""
            <div class="module encoder">
                <h2>エンコーダ構造</h2>
                <div class="description">
                    <p>エンコーダはオクトリー構造をグラフCNNで処理し、特徴を抽出します。深さ{depth}から{depth_stop}まで段階的にダウンサンプリングします。</p>
                </div>
                <pre>
入力オクトリー (深さ{depth})
↓
グラフ畳み込み (Conv1) - 4 → 32チャネル
↓
グラフResBlocks (深さ{depth}) - 複数のResidual Blockを適用
↓
グラフダウンサンプル - 32 → 64チャネル
↓
グラフResBlocks (深さ{depth-1})
↓
グラフダウンサンプル - 64 → 128チャネル
↓
グラフResBlocks (深さ{depth-2})
↓
グラフダウンサンプル - 128 → 256チャネル
↓
... (繰り返し) ...
↓
グラフResBlocks (深さ{depth_stop})
↓
正規化 + GELU非線形変換
↓
1x1畳み込み (平均と分散パラメータ生成) - 512 → {code_channel*2}チャネル
                </pre>
            </div>
    """
    
    # 潜在表現セクション
    latent_section = f"""
            <div class="module latent">
                <h2>潜在表現 (VAE)</h2>
                <div class="description">
                    <p>変分自己符号化器 (VAE) として実装され、潜在分布からサンプリングします。KLダイバージェンスで正則化されています。</p>
                </div>
                <pre>
潜在分布: ガウス分布 N(μ, σ²)
潜在コード形状: [{code_channel}, 8, 8, 8] (3D空間構造を維持)
再パラメータ化トリック: z = μ + σ * ε (ε ~ N(0, 1))
KL損失: KL(N(μ, σ²) || N(0, 1))
                </pre>
            </div>
    """
    
    # デコーダセクション
    decoder_section = f"""
            <div class="module decoder">
                <h2>デコーダ構造</h2>
                <div class="description">
                    <p>デコーダは潜在コードから階層的にオクトリーを再構築します。深さ{depth_stop}から{depth_out}まで段階的にアップサンプリングします。</p>
                </div>
                <pre>
潜在コード [{code_channel}, 8, 8, 8]
↓
1x1畳み込み (チャネル数復元) - {code_channel} → 512チャネル
↓
デコーダーMid Block (ResBlocks)
↓
グラフResBlocks (深さ{depth_stop})
↓
グラフアップサンプル - 512 → 256チャネル
↓
グラフResBlocks (深さ{depth_stop+1})
↓
グラフアップサンプル - 256 → 128チャネル
↓
... (繰り返し) ...
↓
グラフResBlocks (深さ{depth_out})
↓
予測レイヤー:
  - 分割確率: 各ノードの分割確率を予測 (2チャネル)
  - SDFパラメータ: SDF値のパラメータを予測 (4チャネル)
                </pre>
            </div>
    """
    
    # 神経MPUセクション
    mpu_section = """
            <div class="module mpu">
                <h2>神経MPU (Neural MPU)</h2>
                <div class="description">
                    <p>多レベル分割統一 (MPU) アプローチを使用して、再構築されたオクトリーから任意の3D位置のSDF値を計算します。</p>
                </div>
                <pre>
入力: 3D位置 + 再構築オクトリー + 特徴量
↓
位置のオクトリーノード所属を特定
↓
各レベルのノード特徴量を抽出
↓
多レベル分割統一補間:
  - 各深さのノードからSDF値を予測
  - ブレンディング重みで補間
↓
出力: 指定位置のSDF値
                </pre>
            </div>
    """
    
    # 使用方法セクション
    usage_section = """
            <div class="module">
                <h2>GraphVAEの利用方法</h2>
                <div class="description">
                    <p>GraphVAEは3D形状の生成と操作に使用できます。主な用途には以下があります：</p>
                </div>
                <ul>
                    <li><strong>形状圧縮:</strong> 複雑な3Dメッシュをコンパクトな潜在コードに圧縮</li>
                    <li><strong>形状補間:</strong> 異なる形状間をスムーズに補間</li>
                    <li><strong>形状生成:</strong> 新しい3D形状の生成</li>
                    <li><strong>条件付き生成:</strong> テキストや画像からの3D形状生成</li>
                </ul>
            </div>
    """
    
    # HTMLフッター
    html_footer = """
        </div>
    </body>
    </html>
    """
    
    # 完全なHTMLを構築
    full_html = (
        html_header +
        params_section +
        network_diagram +
        octree_section +
        encoder_section +
        latent_section +
        decoder_section +
        mpu_section +
        usage_section +
        html_footer
    )
    
    # HTMLファイルに書き込み
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTMLビジュアライゼーションを保存しました: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='GraphVAEネットワーク構造をHTMLファイルとして視覚化')
    parser.add_argument('--output_path', type=str, default='visualization_results/graphvae_structure.html', 
                        help='出力HTMLファイルのパス')
    parser.add_argument('--depth', type=int, default=6, 
                        help='オクトリーの最大深さ')
    parser.add_argument('--full_depth', type=int, default=2, 
                        help='完全オクトリーの深さ')
    parser.add_argument('--depth_stop', type=int, default=4, 
                        help='エンコード停止深さ')
    parser.add_argument('--depth_out', type=int, default=6, 
                        help='デコード出力深さ')
    parser.add_argument('--code_channel', type=int, default=16, 
                        help='潜在コードのチャネル数')
    args = parser.parse_args()
    
    # 出力ディレクトリが存在することを確認
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # HTML生成を実行
    html_path = generate_html_visualization(
        output_path=args.output_path,
        depth=args.depth,
        full_depth=args.full_depth,
        depth_stop=args.depth_stop,
        depth_out=args.depth_out,
        code_channel=args.code_channel
    )
    
    print(f"HTML可視化が完了しました。以下のファイルをブラウザで開いてください:")
    print(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main() 