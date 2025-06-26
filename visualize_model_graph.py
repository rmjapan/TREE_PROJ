from model.svsae import SVSAE, AutoencoderConfig
import torch
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import os

def visualize_model_graph():
    # モデル設定とインスタンス化
    config = AutoencoderConfig()
    model = SVSAE(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # 入力テンソル (バッチサイズ=1, チャンネル=1, 高さ=幅=深さ=256)
    x = torch.randn(1, 1, 256, 256, 256, requires_grad=True)
    x = x.to("cuda" if torch.cuda.is_available() else "cpu")

    # モデル出力
    y = model(x)

    # torchvizで可視化
    dot = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    dot.render('model_graph', format='png')

    # tensorboardで可視化
    os.makedirs("runs/svsae", exist_ok=True)
    writer = SummaryWriter("runs/svsae")
    writer.add_graph(model, x)
    writer.close()

    print("計算グラフを生成しました。以下のコマンドで確認できます：")
    print("tensorboard --logdir=runs/svsae")

if __name__ == "__main__":
    visualize_model_graph() 