import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GraphVAEのモデル構造のみインポート（重みは不要）
from models.networks.dualoctree_networks.graph_vae import GraphVAE

# ダミーのOctreeとPointsクラスを定義（実際にデータを作成せずに構造だけを表現）
class DummyPoints:
    def __init__(self, batch_size=1):
        self.points = torch.zeros((batch_size, 3), dtype=torch.float32)
        self.normals = torch.zeros((batch_size, 3), dtype=torch.float32)

class DummyOctree:
    def __init__(self, depth=6, full_depth=2, batch_size=1, device='cpu'):
        self.depth = depth
        self.full_depth = full_depth
        self.batch_size = batch_size
        self.device = device
        
    def to(self, device):
        # デバイスを設定するメソッド（ダミー）
        self.device = device
        return self
        
    def nnum(self, depth):
        # 各深さのノード数（ダミー）
        return 8 ** depth
        
    def build_octree(self, points):
        # ダミーのビルドメソッド
        pass
        
    def nempty_mask(self, depth):
        # 非空ノードのマスク（ダミー）
        return torch.ones(8 ** depth, dtype=torch.bool)

# TensorBoardのためのネットワーク構造のみを表現したGraphVAEクラス
class TensorboardGraphVAE(GraphVAE):
    def __init__(self, depth=6, full_depth=2, depth_stop=4, depth_out=6, code_channel=16, device='cpu'):
        # パラメータを初期化
        channel_in = 4  # 入力チャネル数
        nout = 4        # 出力チャネル数
        super().__init__(
            depth=depth, 
            channel_in=channel_in, 
            nout=nout, 
            full_depth=full_depth,
            depth_stop=depth_stop,
            depth_out=depth_out,
            code_channel=code_channel
        )
        self.to(device)
        
    def forward(self, octree_in, octree_out=None, pos=None):
        # TensorBoardのグラフ可視化用の単純化したforward
        # 実際の計算は行わず、構造だけを表現
        
        # ダミーのDualOctreeを使用（実際には使用しない）
        from models.networks.dualoctree_networks.dual_octree import DualOctree
        
        # エンコーダー部分のダミー処理
        posterior = self.octree_encoder(octree_in, None)
        z = posterior.sample()
        
        # デコーダー部分のダミー処理（簡略化）
        logits = {}
        reg_voxs = {}
        
        for d in range(self.depth_stop, self.depth_out + 1):
            # ダミーの出力値を生成
            n_nodes = 8 ** d
            logits[d] = torch.zeros((n_nodes, 2), device=self.device)
            reg_voxs[d] = torch.zeros((n_nodes, 4), device=self.device)
        
        output = {
            'logits': logits,
            'reg_voxs': reg_voxs,
            'octree_out': octree_in,
            'kl_loss': torch.tensor(0.0, device=self.device),
            'code_max': z.max(),
            'code_min': z.min()
        }
        
        return output


def create_network_visualization(log_dir, depth=6, full_depth=2, depth_stop=4, 
                                depth_out=6, code_channel=16, device='cpu'):
    """TensorBoardを使用してGraphVAEのネットワーク構造を可視化する"""
    
    # TensorBoardのログディレクトリを作成
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"ネットワーク構造の可視化を開始します...")
    print(f"  深さ: {depth}")
    print(f"  フル深さ: {full_depth}")
    print(f"  停止深さ: {depth_stop}")
    print(f"  出力深さ: {depth_out}")
    print(f"  コードチャネル: {code_channel}")
    
    # モデルを作成
    model = TensorboardGraphVAE(
        depth=depth,
        full_depth=full_depth,
        depth_stop=depth_stop,
        depth_out=depth_out,
        code_channel=code_channel,
        device=device
    )
    
    # モデルをCPUに設定
    model.eval()
    
    # ダミーの入力を作成
    dummy_octree = DummyOctree(depth=depth, full_depth=full_depth, device=device)
    dummy_points = DummyPoints()
    
    # モデルのサマリーをテキストとして追加
    model_str = str(model)
    writer.add_text('model/architecture', model_str, global_step=0)
    
    # ネットワークの構成情報を追加
    writer.add_text('model/config', f"""
    深さ (depth): {depth}
    フル深さ (full_depth): {full_depth}
    停止深さ (depth_stop): {depth_stop}
    出力深さ (depth_out): {depth_out}
    コードチャネル: {code_channel}
    入力チャネル: {model.channel_in}
    出力チャネル: {model.nout}
    埋め込み次元: {model.embed_dim}
    """, global_step=0)
    
    # モデルのハイパーパラメータを追加
    hparams = {
        'depth': depth,
        'full_depth': full_depth,
        'depth_stop': depth_stop,
        'depth_out': depth_out,
        'code_channel': code_channel,
        'channel_in': model.channel_in,
        'nout': model.nout,
        'embed_dim': model.embed_dim,
        'bottleneck': model.bottleneck,
        'resblk_num': model.resblk_num
    }
    writer.add_hparams(hparams, {'hparam/model_initialized': 1})
    
    # グラフ構造を可視化
    try:
        # ダミー入力を使ってモデルのグラフを可視化
        dummy_input = (dummy_octree, None, torch.zeros((100, 4)))
        writer.add_graph(model, dummy_input)
        print("モデルグラフの追加に成功しました")
    except Exception as e:
        print(f"モデルグラフの追加中にエラーが発生しました: {e}")
        # エラーメッセージを保存
        writer.add_text('errors/graph_visualization', str(e), global_step=0)
    
    # モデルの層の構造を可視化
    for name, module in model.named_modules():
        if name:  # 空の名前をスキップ
            # モジュールの種類と形状を記録
            writer.add_text(f'modules/{name}', str(module), global_step=0)
    
    # モデルの各部分の構造を図式化（カスタム図）
    # エンコーダ部分
    encoder_structure = """
    入力オクトリー (深さ6)
    ↓
    グラフ畳み込み (Conv1)
    ↓
    グラフResBlocks (深さ6)
    ↓
    グラフダウンサンプル
    ↓
    グラフResBlocks (深さ5)
    ↓
    ... (繰り返し) ...
    ↓
    グラフResBlocks (深さ{depth_stop})
    ↓
    正規化 + 非線形変換
    ↓
    1x1畳み込み (平均と分散パラメータ生成)
    ↓
    潜在コード (z) サンプリング
    """.format(depth_stop=depth_stop)
    
    # デコーダ部分
    decoder_structure = """
    潜在コード (z)
    ↓
    1x1畳み込み (チャネル数復元)
    ↓
    デコーダーMid Block (ResBlocks)
    ↓
    グラフResBlocks (深さ{depth_stop})
    ↓
    グラフアップサンプル
    ↓
    グラフResBlocks (深さ{depth_stop+1})
    ↓
    ... (繰り返し) ...
    ↓
    グラフResBlocks (深さ{depth_out})
    ↓
    予測層 (分割確率 + SDFパラメータ)
    """.format(depth_stop=depth_stop, depth_out=depth_out)
    
    # 神経MPU部分
    mpu_structure = """
    再構築オクトリー + 特徴量
    ↓
    ノード特徴量の抽出
    ↓
    再帰的にノードの寄与を計算
    ↓
    多レベル分割統一 (MPU) 補間
    ↓
    任意点のSDF値
    """
    
    writer.add_text('model_structure/encoder', encoder_structure, global_step=0)
    writer.add_text('model_structure/decoder', decoder_structure, global_step=0)
    writer.add_text('model_structure/neural_mpu', mpu_structure, global_step=0)
    
    # TensorBoardログを完了
    writer.close()
    
    print(f"ネットワーク構造の可視化が完了しました")
    print(f"TensorBoardを実行するには: tensorboard --logdir={log_dir}")
    
    return log_dir

def main():
    parser = argparse.ArgumentParser(description='GraphVAEのネットワーク構造をTensorBoardで可視化')
    parser.add_argument('--log_dir', type=str, default='tensorboard_logs/network_structure', 
                        help='TensorBoardログディレクトリ')
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
    
    # タイムスタンプ付きのディレクトリを作成
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.log_dir}_{timestamp}"
    
    # ネットワーク構造の可視化を実行
    create_network_visualization(
        log_dir=log_dir,
        depth=args.depth,
        full_depth=args.full_depth,
        depth_stop=args.depth_stop,
        depth_out=args.depth_out,
        code_channel=args.code_channel,
        device='cpu'  # CPUのみを使用
    )

if __name__ == "__main__":
    main() 