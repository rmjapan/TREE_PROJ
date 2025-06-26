import os
import sys
import argparse
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from ocnn.octree import Octree, Points
import numpy as np

# GraphVAEモデルをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.networks.dualoctree_networks.graph_vae import GraphVAE

def create_sample_points(n_points=1000):
    """球面上のサンプル点を生成"""
    points = []
    normals = []
    
    for _ in range(n_points):
        # 球面上の点をサンプリング
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        # 球面座標から直交座標に変換
        x = 0.5 * np.sin(phi) * np.cos(theta)
        y = 0.5 * np.sin(phi) * np.sin(theta)
        z = 0.5 * np.cos(phi)
        
        points.append([x, y, z])
        normals.append([x, y, z])  # 法線は点の位置と同じ（単位球面の場合）
    
    return torch.tensor(points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)

def visualize_graphvae(args):
    """GraphVAEモデルをTensorBoardで可視化"""
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデルの初期化
    model = GraphVAE(
        depth=args.depth,
        channel_in=4,  # 入力チャネル数（x,y,z + 特徴量）
        nout=2,        # 出力チャネル数（分割予測）
        full_depth=args.full_depth,
        depth_stop=args.depth_stop,
        depth_out=args.depth_out,
        code_channel=args.code_channel,
        embed_dim=3    # 埋め込み次元
    ).to(device)
    
    print("モデルを初期化しました")
    
    # TensorBoardのログディレクトリを設定
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f'graphvae_{timestamp}')
    writer = SummaryWriter(log_dir)
    print(f"ログディレクトリ: {log_dir}")
    
    # モデルのパラメータを記録
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'parameters/{name}', param, global_step=0)
    
    # モデルのアーキテクチャ情報を記録
    writer.add_text('model/architecture', str(model), global_step=0)
    writer.add_text('model/parameters', f"""
    - depth: {args.depth}
    - full_depth: {args.full_depth}
    - depth_stop: {args.depth_stop}
    - depth_out: {args.depth_out}
    - code_channel: {args.code_channel}
    - channel_in: 4
    - nout: 2
    - embed_dim: 3
    """, global_step=0)
    
    # サンプル入力の生成とモデルグラフの記録
    try:
        # サンプル点群の生成
        points, normals = create_sample_points()
        points = points.to(device)
        normals = normals.to(device)
        
        # Points オブジェクトの作成
        points_obj = Points(points=points, normals=normals)
        
        # Octreeの構築
        octree = Octree(args.depth, args.full_depth, batch_size=1, device=device)
        octree.build_octree(points_obj)
        
        # サンプル位置の生成
        sample_pos = torch.rand(1000, 4, device=device)  # [N, 4] 形式（バッチインデックス付き）
        
        # モデルグラフの記録
        writer.add_graph(model, (octree, None, sample_pos))
        print("モデルグラフを記録しました")
        
    except Exception as e:
        writer.add_text('model/graph_error', f"モデルグラフの記録に失敗: {str(e)}", global_step=0)
        print(f"警告: モデルグラフの記録に失敗しました: {e}")
    
    # ハイパーパラメータの記録
    hparams = {
        'depth': args.depth,
        'full_depth': args.full_depth,
        'depth_stop': args.depth_stop,
        'depth_out': args.depth_out,
        'code_channel': args.code_channel,
    }
    writer.add_hparams(hparams, {'hparam/model_initialized': 1})
    
    writer.close()
    print("\n可視化が完了しました")
    print(f"TensorBoardを起動するには: tensorboard --logdir={args.log_dir}")
    print("その後、ブラウザで http://localhost:6006 にアクセスしてください")

def main():
    parser = argparse.ArgumentParser(description='GraphVAEモデルをTensorBoardで可視化')
    parser.add_argument('--depth', type=int, default=6, help='オクトリーの深さ')
    parser.add_argument('--full_depth', type=int, default=2, help='フル展開の深さ')
    parser.add_argument('--depth_stop', type=int, default=4, help='エンコーダの停止深さ')
    parser.add_argument('--depth_out', type=int, default=6, help='デコーダの出力深さ')
    parser.add_argument('--code_channel', type=int, default=16, help='潜在コードのチャネル数')
    parser.add_argument('--log_dir', type=str, default='tensorboard_logs', help='TensorBoardログディレクトリ')
    
    args = parser.parse_args()
    
    # ログディレクトリの作成
    os.makedirs(args.log_dir, exist_ok=True)
    
    visualize_graphvae(args)

if __name__ == '__main__':
    main() 