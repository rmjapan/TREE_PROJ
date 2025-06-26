import os
import sys
import argparse
from omegaconf import OmegaConf

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_utils import load_dualoctree
from models.networks.dualoctree_networks.graph_vae import visualize_model


def main():
    parser = argparse.ArgumentParser(description='GraphVAEモデルのアーキテクチャを可視化')
    parser.add_argument('--vq_cfg', type=str, required=True, help='VAEの設定ファイルへのパス')
    parser.add_argument('--vq_ckpt', type=str, required=True, help='VAEのチェックポイントへのパス')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoardログディレクトリ')
    parser.add_argument('--device', type=str, default='cpu', help='デバイス (cuda または cpu)')
    parser.add_argument('--sample_input', action='store_true', default=False, 
                        help='モデルグラフの可視化のためにサンプル入力を生成するかどうか（計算負荷が高い）')
    args = parser.parse_args()
    
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
    
    # モデルのTensorBoard可視化を実行
    print("\nTensorBoardによるモデル可視化を開始します...")
    log_dir = visualize_model(model, args.log_dir, args.sample_input)
    
    print(f"\n可視化が完了しました。以下のコマンドでTensorBoardを起動できます:")
    print(f"tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main() 