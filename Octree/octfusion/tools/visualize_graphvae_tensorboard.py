import os
import sys
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks.graph_vae import GraphVAE
from models.networks.dualoctree_networks.dual_octree import DualOctree
from models.model_utils import load_dualoctree


def load_example_data(data_path, point_scale=1.0):
    """点群データを読み込み、Octreeに変換する"""
    # .npzファイルから点群データを読み込む
    raw = np.load(data_path)
    points = raw['points']
    normals = raw['normals']
    
    # 点群を[-1, 1]の範囲にスケーリング
    points = points / point_scale
    
    # Pointsオブジェクトを作成
    points_obj = Points(
        points=torch.from_numpy(points).float(), 
        normals=torch.from_numpy(normals).float()
    )
    points_obj.clip(min=-1, max=1)
    
    # 点群からOctreeを作成
    depth = 6  # Octfusionで使用される標準的な深さ
    full_depth = 2  # GraphVAEのためのfull_depth
    octree = Octree(depth, full_depth)
    octree.build_octree(points_obj)
    
    return octree, points_obj


def visualize_graphvae_tensorboard(model, octree_in, log_dir, device="cpu"):
    """TensorBoardを使ってGraphVAEモデルを可視化する"""
    # TensorBoardのSummaryWriterを初期化
    writer = SummaryWriter(log_dir)
    
    model.eval()
    
    # 入力をデバイスに移動
    octree_in = octree_in.to(device)
    
    # 入力OctreeからDualOctreeを作成
    doctree_in = DualOctree(octree=octree_in)
    doctree_in.post_processing_for_docnn()
    
    # モデルアーキテクチャ情報をTensorBoardに追加
    writer.add_text('model/architecture', str(model), global_step=0)
    writer.add_text('model/depth', f"depth: {model.depth}, full_depth: {model.full_depth}, depth_stop: {model.depth_stop}, depth_out: {model.depth_out}", global_step=0)
    
    # モデルパラメータのヒストグラムをTensorBoardに追加
    for name, param in model.named_parameters():
        writer.add_histogram(f'parameters/{name}', param.clone().cpu().data.numpy(), global_step=0)
    
    # 入力Octreeをエンコードして潜在分布を取得
    print("入力Octreeをエンコード中...")
    posterior = model.octree_encoder(octree_in, doctree_in)
    
    # 潜在分布からサンプリング
    z = posterior.sample()
    
    # 潜在コードの統計情報をTensorBoardに追加
    writer.add_histogram('latent/z', z.clone().cpu().data.numpy(), global_step=0)
    writer.add_scalar('latent/min', z.min().item(), global_step=0)
    writer.add_scalar('latent/max', z.max().item(), global_step=0)
    writer.add_scalar('latent/mean', z.mean().item(), global_step=0)
    writer.add_scalar('latent/std', z.std().item(), global_step=0)
    writer.add_scalar('latent/kl_divergence', posterior.kl().mean().item(), global_step=0)
    
    # 潜在コードの2Dスライスを画像としてTensorBoardに追加
    if len(z.shape) >= 4:  # 3D latent code
        for c in range(min(4, z.shape[1])):  # 最初の4チャンネルのみ表示
            writer.add_image(f'latent/channel_{c}', 
                             z[0, c:c+1].clone().cpu(), 
                             global_step=0, 
                             dataformats='CHW')
    
    # デコード用の出力Octreeを作成
    update_octree = True
    octree_out = model.create_full_octree(octree_in)
    octree_out.depth = model.full_depth
    
    # Octreeをdepth_stopまで成長させる
    for d in range(model.full_depth, model.depth_stop):
        label = octree_in.nempty_mask(d).long()
        octree_out.octree_split(label, d)
        octree_out.octree_grow(d + 1)
        octree_out.depth += 1
    
    # 出力用のDualOctreeを作成
    doctree_out = DualOctree(octree_out)
    doctree_out.post_processing_for_docnn()
    
    try:
        # 潜在コードをデコード
        print("潜在コードをデコード中...")
        out = model.octree_decoder(z, doctree_out, update_octree=update_octree)
        logits, reg_voxs, octree_out = out
        
        # デコード結果の統計情報をTensorBoardに追加
        for d in range(model.depth_stop, model.depth_out + 1):
            if d in logits:
                writer.add_histogram(f'decode/logits_depth_{d}', 
                                     logits[d].clone().cpu().data.numpy(), 
                                     global_step=0)
            
            if d in reg_voxs:
                writer.add_histogram(f'decode/reg_voxs_depth_{d}', 
                                     reg_voxs[d].clone().cpu().data.numpy(), 
                                     global_step=0)
        
        # 各深さのノード数をスカラーとしてTensorBoardに追加
        for d in range(octree_out.depth + 1):
            try:
                mask = octree_out.nempty_mask(d)
                if mask is not None:
                    n_nodes = mask.sum().item()
                    writer.add_scalar(f'octree/nodes_at_depth_{d}', n_nodes, global_step=0)
            except Exception as e:
                print(f"深さ{d}のノード数を取得できませんでした: {e}")
    
    except Exception as e:
        print(f"デコード中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    writer.close()
    print(f"TensorBoard可視化が完了しました。`tensorboard --logdir={log_dir}`で結果を確認できます。")


def main():
    parser = argparse.ArgumentParser(description='TensorBoardを使ってGraphVAEモデルを可視化')
    parser.add_argument('--vq_cfg', type=str, required=True, help='VAEの設定ファイルへのパス')
    parser.add_argument('--vq_ckpt', type=str, required=True, help='VAEのチェックポイントへのパス')
    parser.add_argument('--data_path', type=str, required=True, help='サンプルデータへのパス (.npzファイル)')
    parser.add_argument('--log_dir', type=str, default='tensorboard_logs/graphvae', help='TensorBoardログディレクトリ')
    parser.add_argument('--device', type=str, default='cpu', help='デバイス (cuda または cpu)')
    args = parser.parse_args()
    
    # ログディレクトリを作成
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 設定ファイルを読み込む
    print(f"{args.vq_cfg}から設定を読み込んでいます...")
    try:
        vq_conf = OmegaConf.load(args.vq_cfg)
        print("設定の読み込みに成功しました")
    except Exception as e:
        print(f"設定の読み込み中にエラーが発生しました: {e}")
        return
    
    # モデルを読み込む
    print(f"{args.vq_ckpt}からGraphVAEモデルを読み込んでいます...")
    try:
        model = load_dualoctree(conf=vq_conf, ckpt=args.vq_ckpt)
        model.to(args.device)
        print("モデルの読み込みに成功しました")
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        return
    
    # モデルの構成情報を表示
    print("\nGraphVAEモデルの構成:")
    print(f"  深さ: {model.depth}")
    print(f"  Full深さ: {model.full_depth}")
    print(f"  停止深さ: {model.depth_stop}")
    print(f"  出力深さ: {model.depth_out}")
    print(f"  コードチャンネル: {model.code_channel}")
    
    # サンプルデータを読み込む
    print(f"{args.data_path}からサンプルデータを読み込んでいます...")
    try:
        point_scale = vq_conf.data.test.point_scale if 'point_scale' in vq_conf.data.test else 1.0
        octree_in, points_obj = load_example_data(args.data_path, point_scale)
        print(f"データの読み込みに成功しました: {len(points_obj.points)}点")
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        return
    
    # GraphVAEの処理をTensorBoardで可視化
    print("\nGraphVAEの可視化を開始します...")
    visualize_graphvae_tensorboard(model, octree_in, args.log_dir, args.device)


if __name__ == "__main__":
    main() 