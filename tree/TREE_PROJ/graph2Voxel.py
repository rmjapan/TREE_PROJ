import numpy as np


def AABB(edge,DG):
    start_node = DG.nodes[edge[0]]['node']
    end_node = DG.nodes[edge[1]]['node']
    start_pos = np.array([start_node.pos.x, start_node.pos.y, start_node.pos.z])
    end_pos = np.array([end_node.pos.x, end_node.pos.y, end_node.pos.z])
    thickness = DG.edges[edge]['edge'].thickness
    margin=0
    
    x_min = min(start_pos[0], end_pos[0])
    x_max = max(start_pos[0], end_pos[0])
    y_min = min(start_pos[1], end_pos[1])
    y_max = max(start_pos[1], end_pos[1])
    z_min = min(start_pos[2], end_pos[2])
    z_max = max(start_pos[2], end_pos[2])


    
    
    X_min=x_min-thickness/2-margin
    X_max=x_max+thickness/2+margin
    Y_min=y_min-thickness/2-margin
    Y_max=y_max+thickness/2+margin
    Z_min=z_min-thickness/2-margin
    Z_max=z_max+thickness/2+margin
    return X_min,X_max,Y_min,Y_max,Z_min,Z_max
    
def initialize_voxel_data(DG, H=32, W=32, D=32):
        # ボクセルデータを初期化
        voxel_data = np.full((H, W, D), -1, dtype=float)
        
        # ノードの座標を取得(x, y, z)
        x_vals = [DG.nodes[node]['node'].pos.x for node in DG.nodes()]
        y_vals = [DG.nodes[node]['node'].pos.y for node in DG.nodes()]
        z_vals = [DG.nodes[node]['node'].pos.z for node in DG.nodes()]
        # ノードの座標の最小値と最大値を取得
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)
        
        # 配列の中心を計算
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        
        # ボクセルスケールとサイズを計算
        max_length = max(x_max - x_min, y_max - y_min, z_max - z_min)
        margin = 5  # 余裕を持たせる
        voxel_scale = max_length + margin #立方体の一辺の長さ  
        voxel_size = voxel_scale / H #ボクセルの一辺の長さ
        print(voxel_size)
        voxel_xmin = x_center - voxel_scale / 2 #X軸の最小値
        voxel_ymin = y_center - voxel_scale / 2 #Y軸の最小値
        voxel_zmin = z_center - voxel_scale / 2 #Z軸の最小値
        
        # 座標軸の値を計算
        x_coords = voxel_xmin + np.arange(H) * voxel_size
        y_coords = voxel_ymin + np.arange(W) * voxel_size
        z_coords = voxel_zmin + np.arange(D) * voxel_size
        
        # メッシュグリッドを作成
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        #これで、ijkでアクセスできるようになる
        voxel_positions = np.stack((X, Y, Z), axis=-1)
        
        return voxel_data, voxel_positions, x_coords, y_coords, z_coords, voxel_size

from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull, Delaunay

def create_voxel_data(DG, H=32, W=32, D=32):

    # ボクセルデータを初期化
    voxel_data, voxel_positions, x_coords, y_coords, z_coords, voxel_size = initialize_voxel_data(DG, H, W, D)
    
    
    for edge in DG.edges():
        start_node = DG.nodes[edge[0]]['node']#エッジの始点
        end_node = DG.nodes[edge[1]]['node']#エッジの終点
        thickness = DG.edges[edge]['edge'].thickness#エッジの太さ
        #幹の場合はエッジの太さを＋ボクセルサイズ
        if start_node.attr == 1 and end_node.attr == 1:
            thickness += voxel_size
        
        #エッジ内に含まれる可能性のあるボクセルの範囲を数値で計算（AABB）
        X_min,X_max,Y_min,Y_max,Z_min,Z_max=AABB(edge,DG)
        # エッジの範囲をIndexで取得
        i_min = np.searchsorted(x_coords, X_min, side='left')
        i_max = np.searchsorted(x_coords, X_max, side='right') - 1
        j_min = np.searchsorted(y_coords, Y_min, side='left')
        j_max = np.searchsorted(y_coords, Y_max, side='right') - 1
        k_min = np.searchsorted(z_coords, Z_min, side='left')
        k_max = np.searchsorted(z_coords, Z_max, side='right') - 1

        
        
        #エッジが含まれるボクセルの範囲を取得
        pos_array=voxel_positions[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1, :]
        
        #始点と終点をnumpy配列に変換
        start_pos=np.array([DG.nodes[edge[0]]['node'].pos.x, DG.nodes[edge[0]]['node'].pos.y, DG.nodes[edge[0]]['node'].pos.z])
        end_pos=np.array([DG.nodes[edge[1]]['node'].pos.x, DG.nodes[edge[1]]['node'].pos.y, DG.nodes[edge[1]]['node'].pos.z])
    
      
        #エッジのベクトルを計算
        edge_vector=end_pos-start_pos
        
        #エッジのノルムの二乗を計算
        edge_norm_square=np.dot(edge_vector,edge_vector)
        
        #始点から各ボクセルの位置までのベクトルを全て計算
        start_to_pos=pos_array-start_pos
       
        #proj_length(始点から各ボクセルの位置までのベクトルとエッジのベクトルの内積/エッジのノルムの二乗)
        if edge_norm_square >1e-10:
            proj_length = np.einsum('ijkl,l->ijk', start_to_pos, edge_vector) / edge_norm_square
        else:
            proj_length = np.zeros_like(start_to_pos[..., 0])
            
        
        
        #proj_pointの計算(Ni, Nj, Nk, 3)pos_arrayがedge_vecに正射影された点の座標
        proj_point = start_pos + proj_length[..., np.newaxis] * edge_vector  # shapeは[Ni, Nj, Nk, 3]
        #正射影点から各ボクセルの位置までのベクトルを計算
        diff = pos_array - proj_point
        #正射影点から各ボクセルの位置までのベクトルのノルムを計算
        distance = np.linalg.norm(diff, axis=-1)  # shapeは[Ni, Nj, Nk]

        
        min_pos = np.minimum(start_pos, end_pos)
        max_pos = np.maximum(start_pos, end_pos)
        # 正射影点がエッジの範囲内にあるかどうかを判定（マージンを考慮してvoxel_size分だけ拡張）
        within_edge = np.all((proj_point >= min_pos-voxel_size*1) & (proj_point <= max_pos+voxel_size*1), axis=-1) 
        # マスクを作成(エッジの範囲内かつエッジの太さの半分以下にあるボクセル+空ボクセル→True)
        
        mask = within_edge & (distance <= thickness / 2) 
        
        
            
        

        # 属性に応じた値の割り当て
        value =0.5  # デフォルトは枝(元：２)
        #始点が幹、終点が幹の場合→幹
        if start_node.attr == 1 and end_node.attr == 1:
            value = 1
        #始点が枝、終点が葉の場合→葉
        elif start_node.attr == 0.5 and end_node.attr == 0:
            value = 0
        
        #幹>枝>葉の優先順位（既に幹がある場合は枝や葉は無視）
        if value == 1:  # trunk overrides everything
            pass
        elif value == 0.5:  # branch overrides leaves
            mask &= (voxel_data[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] != 1)
            # pass    
        elif value == 0:  # leaf only fills empty
            mask &= (voxel_data[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] != 1)
            # mask &= (voxel_data[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] != 0.5)
            
            
        

        voxel_data[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1][mask] = value
        
    return voxel_data