import numpy as np
import pyvista as pv
import os
from pathlib import Path
import time
from multiprocessing import Process
import torch
# PyVistaのバックエンドを設定
pv.start_xvfb()
def visualize_with_timeout(data, timeout=10):
    """指定されたデータを可視化し、タイムアウト後にプロセスを終了する関数"""
    process = Process(target=visualize_voxel_data_4_modelnet, args=(data,))
    process.start()
    
    start_time: float = time.time()
    while time.time() - start_time < timeout:
        if not process.is_alive():
            break
        time.sleep(0.1)
    
    process.terminate()
def visualize_with_timeout4voxel(data, timeout=10, title=None):
    """指定されたデータを可視化し、タイムアウト後にプロセスを終了する関数"""
    process = Process(target=visualize_voxel_data, args=(data,), kwargs={'title': title})
    process.start()
    
    start_time: float = time.time()
    while time.time() - start_time < timeout:
        if not process.is_alive():
            break
        time.sleep(0.1)
    
    process.terminate()
def visualize_voxel_data(voxel_data, title=None):
    import plotly.graph_objects as go
    # ボクセルが存在する位置を判定
    if type(voxel_data) == torch.Tensor:
        voxel_data = voxel_data.numpy()
    filled_positions = np.argwhere(voxel_data != -1)
    x, y, z = filled_positions.T
    print(len(filled_positions))

    # 空ではないボクセルの色を設定
    colors = []
    for pos in filled_positions:
        value = voxel_data[tuple(pos)]
        if value == 1:
            colors.append('brown')
        elif value == 0.5:
            colors.append('yellow')
        elif value == 0:
            colors.append('green')

    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        )
    ))
    
    # タイトルがある場合は設定
    layout_dict = dict(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    
    if title is not None:
        layout_dict['title'] = title
        
    fig.update_layout(**layout_dict)
    fig.show()

def visualize_voxel_data_4_modelnet(voxel_data):
    import plotly.graph_objects as go
    # ボクセルが存在する位置を判定
    #データタイプがtorch.Tensorの場合はnumpyに変換
    if isinstance(voxel_data, torch.Tensor):
        voxel_data = voxel_data.numpy()
    filled_positions = np.argwhere(voxel_data != -1)
    x, y, z = filled_positions.T
    print(len(filled_positions))

    # 空ではないボクセルの色を設定
    colors = []
    for pos in filled_positions:
        value = voxel_data[tuple(pos)]
        if value != 0:
            colors.append('brown')
        elif value == 0:
            #空の場合は透明にする
            colors.append('rgba(0,0,0,0)')

    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        )
    ))
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ))
    fig.show()

def visualize_and_save_volume(volume_data, filename, cmap="viridis"):
            """
            ボリュームデータを可視化して画像として保存する関数
            
            Args:
                volume_data: 可視化するボリュームデータ (torch.Tensor)
                filename: 保存するファイルのパス
                cmap: カラーマップ (デフォルト: "viridis")
            """
            # 保存先ディレクトリが存在することを確認
            save_dir = os.path.dirname(filename)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                pl = pv.Plotter(off_screen=True)
                # データを適切な形式に変換
                volume_np = volume_data[0].cpu().detach().numpy().squeeze()
                
                # ヒートマップとして可視化
                pl.add_volume(volume_np, cmap=cmap, scalar_bar_args={'title': 'Value'})
                pl.show(screenshot=filename)
                print(f"Volume visualization saved to {filename}")
            except Exception as e:
                print(f"Error in visualization: {e}")
                # エラーが発生した場合は、代替の可視化方法を試みる
                try:
                    import matplotlib.pyplot as plt
                    # 3Dデータの中央スライスを2D画像として保存
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    volume_np = volume_data[0].cpu().detach().numpy().squeeze()
                    
                    mid_x = volume_np.shape[0] // 2
                    mid_y = volume_np.shape[1] // 2
                    mid_z = volume_np.shape[2] // 2

                    # ヒートマップとして表示し、カラーバーを追加
                    im0 = axes[0].imshow(volume_np[mid_x, :, :], cmap=cmap)
                    axes[0].set_title('ZY plane')
                    fig.colorbar(im0, ax=axes[0], label='Value')
                    
                    im1 = axes[1].imshow(volume_np[:, mid_y, :], cmap=cmap)
                    axes[1].set_title('XZ plane')
                    fig.colorbar(im1, ax=axes[1], label='Value')
                    
                    im2 = axes[2].imshow(volume_np[:, :, mid_z], cmap=cmap)
                    axes[2].set_title('XY plane')
                    fig.colorbar(im2, ax=axes[2], label='Value')
                    
                    # 数値の最小値と最大値を表示
                    min_val = volume_np.min()
                    max_val = volume_np.max()
                    fig.suptitle(f'Min: {min_val:.3f}, Max: {max_val:.3f}')
                    
                    plt.tight_layout()
                    plt.savefig(filename)
                    plt.close()
                    print(f"Fallback 2D visualization saved to {filename}")
                except Exception as e2:
                    print(f"Fallback visualization also failed: {e2}")
