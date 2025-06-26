import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import networkx as nx
import plotly.graph_objects as go
import json
import os

# 方向関係の可視化用定数
DIRECTION_COLORS = {
    0: '#e41a1c',  # 前
    1: '#377eb8',  # 後ろ
    2: '#4daf4a',  # 左
    3: '#ff7f00',  # 右
    4: '#984ea3',  # 上
    5: '#00bfc4',  # 下
}

def plot_octant_connections(vi, vj, rel_dir, child_vi, child_vj, dir_table, remap, ncum_d):
    """
    vi, vjとその子ノード間の接続関係を可視化
    
    Parameters:
        vi, vj: 親ノードのインデックス
        rel_dir: 相対方向
        child_vi, child_vj: 子ノードのリスト
        dir_table: 方向テーブル
        remap: 方向リマップテーブル
        ncum_d: 累積ノード数オフセット
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 親ノードの位置（中心に配置）
    vi_pos = np.array([0, 0, 0])
    vj_pos = np.array([4, 0, 0])  # vjは少し離す
    
    # 親ノードを描画
    ax.scatter(*vi_pos, color='red', s=100, label=f'vi ({vi})')
    ax.scatter(*vj_pos, color='blue', s=100, label=f'vj ({vj})')
    
    # 親ノード間の接続を描画
    ax.plot([vi_pos[0], vj_pos[0]], [vi_pos[1], vj_pos[1]], [vi_pos[2], vj_pos[2]], 
            'k-', linewidth=1, alpha=0.5)
    
    # 方向によって配置を調整
    directions = {
        0: np.array([0, 1, 0]),    # 前
        1: np.array([0, -1, 0]),   # 後ろ
        2: np.array([-1, 0, 0]),   # 左
        3: np.array([1, 0, 0]),    # 右
        4: np.array([0, 0, 1]),    # 上
        5: np.array([0, 0, -1]),   # 下
    }
    
    # 子ノードの位置を計算（オクタントの位置）
    child_positions = {}
    for i in range(8):
        # 3ビット（8方向）の位置エンコーディング
        x_offset = 1 if (i & 1) else -1
        y_offset = 1 if (i & 2) else -1
        z_offset = 1 if (i & 4) else -1
        
        # viの子ノード位置
        child_vi_pos = vi_pos + np.array([x_offset, y_offset, z_offset])
        child_positions[child_vi[i]] = child_vi_pos
        
        # vjの子ノード位置
        child_vj_pos = vj_pos + np.array([x_offset, y_offset, z_offset])
        child_positions[child_vj[i]] = child_vj_pos
    
    # 子ノードを描画
    for i, (child_idx, pos) in enumerate(child_positions.items()):
        if i < 8:  # viの子ノード
            ax.scatter(*pos, color='lightcoral', s=50, alpha=0.7)
            ax.text(*pos, f'{child_idx}', fontsize=8)
        else:      # vjの子ノード
            ax.scatter(*pos, color='lightblue', s=50, alpha=0.7)
            ax.text(*pos, f'{child_idx}', fontsize=8)
    
    # 変換後の接続関係を計算
    row_o2 = child_vi.unsqueeze(1) * 8 + dir_table[rel_dir, :]
    row_o2 = row_o2.view(-1) + ncum_d
    
    rel_dir_col = remap[rel_dir]
    col_o2 = child_vj.unsqueeze(1) * 8 + dir_table[rel_dir_col, :]
    col_o2 = col_o2.view(-1) + ncum_d
    
    # 子ノード間の接続を描画
    for r, c in zip(row_o2, col_o2):
        if r.item() in child_positions and c.item() in child_positions:
            r_pos = child_positions[r.item()]
            c_pos = child_positions[c.item()]
            ax.plot([r_pos[0], c_pos[0]], [r_pos[1], c_pos[1]], [r_pos[2], c_pos[2]], 
                   'g-', linewidth=0.5, alpha=0.5)
    
    # 方向情報をテキストで表示
    ax.text(2, 2, 2, f'相対方向: {rel_dir.item()}', fontsize=12)
    ax.text(2, 2, 1.5, f'リマップ方向: {rel_dir_col.item()}', fontsize=12)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('オクトリーノード接続関係の可視化')
    plt.legend()
    
    return fig

def visualize_dual_octree_debug(model, depth, vi, vj, save_dir='debug_vis'):
    """
    デュアルオクトリーの接続計算をデバッグするための可視化関数
    
    Parameters:
        model: デュアルオクトリーのモデル
        depth: 現在の深さ
        vi, vj: 調査したいノードインデックス
        save_dir: 保存ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ノードの相対方向を計算
    rel_dir = model.relative_dir(vi, vj, depth - 1, rescale=False)
    
    # 子ノードを取得
    child_vi = model.child[vi]
    child_vj = model.child[vj]
    
    # 累積ノード数を取得
    ncum_d = model.ncum[depth] if hasattr(model, 'ncum') else 0
    
    # Matplotlib で静的可視化
    fig = plot_octant_connections(
        vi, vj, rel_dir, child_vi, child_vj, 
        model.dir_table, model.remap, ncum_d
    )
    plt.savefig(f'{save_dir}/octree_debug_{vi}_{vj}_depth{depth}.png')
    plt.close(fig)
    
    # Plotly で動的可視化 (インタラクティブバージョン)
    visualize_dual_octree_plotly(
        vi, vj, rel_dir, child_vi, child_vj, 
        model.dir_table, model.remap, ncum_d,
        f'{save_dir}/octree_debug_{vi}_{vj}_depth{depth}_interactive.html'
    )
    
    print(f"可視化ファイルを保存しました: {save_dir}/octree_debug_{vi}_{vj}_depth{depth}.png")
    print(f"インタラクティブ可視化: {save_dir}/octree_debug_{vi}_{vj}_depth{depth}_interactive.html")

def visualize_dual_octree_plotly(vi, vj, rel_dir, child_vi, child_vj, dir_table, remap, ncum_d, output_file):
    """
    Plotlyを使用したインタラクティブな可視化
    """
    # Plotly用のデータ構造に変換
    nodes = []
    edges = []
    
    # 親ノードの位置
    vi_pos = np.array([0, 0, 0])
    vj_pos = np.array([4, 0, 0])
    
    # 親ノードを追加
    nodes.append({
        'id': int(vi), 
        'x': float(vi_pos[0]), 
        'y': float(vi_pos[1]), 
        'z': float(vi_pos[2]),
        'color': 'red',
        'size': 15,
        'label': f'vi ({int(vi)})'
    })
    
    nodes.append({
        'id': int(vj), 
        'x': float(vj_pos[0]), 
        'y': float(vj_pos[1]), 
        'z': float(vj_pos[2]),
        'color': 'blue',
        'size': 15,
        'label': f'vj ({int(vj)})'
    })
    
    # 親ノード間のエッジ
    edges.append({
        'source': int(vi),
        'target': int(vj),
        'color': 'black',
        'width': 2
    })
    
    # 子ノードの位置と接続
    child_positions = {}
    
    # 子ノードの位置を計算
    for i in range(8):
        # 3ビットの位置エンコーディング
        x_offset = 1 if (i & 1) else -1
        y_offset = 1 if (i & 2) else -1
        z_offset = 1 if (i & 4) else -1
        
        # viの子ノード
        child_vi_idx = int(child_vi[i])
        child_vi_pos = vi_pos + np.array([x_offset, y_offset, z_offset])
        child_positions[child_vi_idx] = child_vi_pos
        
        nodes.append({
            'id': child_vi_idx,
            'x': float(child_vi_pos[0]),
            'y': float(child_vi_pos[1]),
            'z': float(child_vi_pos[2]),
            'color': 'lightcoral',
            'size': 10,
            'label': f'child_vi_{i} ({child_vi_idx})'
        })
        
        # 親ノードとの接続
        edges.append({
            'source': int(vi),
            'target': child_vi_idx,
            'color': 'rgba(255,0,0,0.3)',
            'width': 1
        })
        
        # vjの子ノード
        child_vj_idx = int(child_vj[i])
        child_vj_pos = vj_pos + np.array([x_offset, y_offset, z_offset])
        child_positions[child_vj_idx] = child_vj_pos
        
        nodes.append({
            'id': child_vj_idx,
            'x': float(child_vj_pos[0]),
            'y': float(child_vj_pos[1]),
            'z': float(child_vj_pos[2]),
            'color': 'lightblue',
            'size': 10,
            'label': f'child_vj_{i} ({child_vj_idx})'
        })
        
        # 親ノードとの接続
        edges.append({
            'source': int(vj),
            'target': child_vj_idx,
            'color': 'rgba(0,0,255,0.3)',
            'width': 1
        })
    
    # 変換後の接続関係を計算
    row_o2 = child_vi.unsqueeze(1) * 8 + dir_table[rel_dir, :]
    row_o2 = row_o2.view(-1) + ncum_d
    
    rel_dir_col = remap[rel_dir]
    col_o2 = child_vj.unsqueeze(1) * 8 + dir_table[rel_dir_col, :]
    col_o2 = col_o2.view(-1) + ncum_d
    
    # 子ノード間の接続を追加
    for r, c in zip(row_o2, col_o2):
        r_int = int(r)
        c_int = int(c)
        if r_int in child_positions and c_int in child_positions:
            edges.append({
                'source': r_int,
                'target': c_int,
                'color': 'rgba(0,255,0,0.7)',
                'width': 1.5,
                'dash': 'dash'
            })
    
    # 可視化データ
    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_z = [node['z'] for node in nodes]
    node_color = [node['color'] for node in nodes]
    node_size = [node['size'] for node in nodes]
    node_text = [node['label'] for node in nodes]
    
    # エッジの線を作成
    edge_x = []
    edge_y = []
    edge_z = []
    edge_color = []
    edge_width = []
    
    for edge in edges:
        source_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['source'])
        target_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['target'])
        
        x0, y0, z0 = nodes[source_idx]['x'], nodes[source_idx]['y'], nodes[source_idx]['z']
        x1, y1, z1 = nodes[target_idx]['x'], nodes[target_idx]['y'], nodes[target_idx]['z']
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_color.extend([edge['color'], edge['color'], edge['color']])
        edge_width.extend([edge['width'], edge['width'], edge['width']])
    
    # ノードトレース
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=0.5, color='rgb(50,50,50)')
        ),
        textposition='top center',
        hoverinfo='text'
    )
    
    # エッジトレース
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(
            color=edge_color,
            width=edge_width
        ),
        hoverinfo='none'
    )
    
    # 情報テキスト
    rel_dir_int = int(rel_dir)
    rel_dir_col_int = int(rel_dir_col)
    
    # フィギュア作成
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # レイアウト設定
    fig.update_layout(
        title=f'デュアルオクトリー接続関係 (vi={int(vi)}, vj={int(vj)}, rel_dir={rel_dir_int}→{rel_dir_col_int})',
        scene=dict(
            xaxis=dict(showbackground=True, showticklabels=True, title='X'),
            yaxis=dict(showbackground=True, showticklabels=True, title='Y'),
            zaxis=dict(showbackground=True, showticklabels=True, title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
        annotations=[
            dict(
                x=0.5, y=0.95, xref='paper', yref='paper',
                text=f'相対方向: {rel_dir_int} → リマップ方向: {rel_dir_col_int}',
                showarrow=False, font=dict(size=14)
            )
        ]
    )
    
    # HTMLに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fig.to_html(include_plotlyjs='cdn', full_html=True))
    
    return output_file

# テスト用の関数
def test_debug_visualization():
    """
    テスト用のシミュレーションデータで可視化をテスト
    """
    class MockDualOctree:
        def __init__(self):
            self.child = torch.tensor([
                [8, 9, 10, 11, 12, 13, 14, 15],   # vi の子ノード
                [16, 17, 18, 19, 20, 21, 22, 23]  # vj の子ノード
            ])
            self.dir_table = torch.tensor([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [0, 1, 4, 5],
                [2, 3, 6, 7]
            ])
            self.remap = torch.tensor([1, 0, 3, 2, 5, 4])
            self.ncum = torch.tensor([0, 8, 24, 64])
        
        def relative_dir(self, vi, vj, depth, rescale=False):
            # 方向をシミュレート (実際は相対位置から計算)
            return torch.tensor(2)  # 左方向と仮定
    
    # モックオブジェクトを作成
    mock_model = MockDualOctree()
    
    # 可視化を実行
    visualize_dual_octree_debug(mock_model, 2, torch.tensor(0), torch.tensor(1), save_dir='debug_test')
    
    print("テスト可視化が完了しました。debug_testディレクトリを確認してください。")

if __name__ == "__main__":
    # テスト実行
    test_debug_visualization() 