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
    
    # dammy_root_nodeを検索
    dammy_root_node = None
    for node in DG.nodes():
        if DG.nodes[node]['node'].dammy_root_flag == True:
            dammy_root_node = DG.nodes[node]['node']
            break
    
    if dammy_root_node is not None:
        x_center = dammy_root_node.pos.x
        y_ground = dammy_root_node.pos.y  # Y軸は地面として扱う
        z_center = dammy_root_node.pos.z
        print(f"Center set to dammy_root_node: X={x_center}, Y_ground={y_ground}, Z={z_center}")
        
        # dammy_root_nodeから各方向への最大距離を計算
        x_vals = [DG.nodes[node]['node'].pos.x for node in DG.nodes()]
        y_vals = [DG.nodes[node]['node'].pos.y for node in DG.nodes()]
        z_vals = [DG.nodes[node]['node'].pos.z for node in DG.nodes()]
        
        x_min_dist = x_center - min(x_vals)
        x_max_dist = max(x_vals) - x_center
        y_max_dist = max(y_vals) - y_ground  # Y軸は地面から上方向のみ
        z_min_dist = z_center - min(z_vals)
        z_max_dist = max(z_vals) - z_center
        
        print(f"x_min_dist: {x_min_dist}, x_max_dist: {x_max_dist}, y_max_dist: {y_max_dist}, z_min_dist: {z_min_dist}, z_max_dist: {z_max_dist}")
        
        # 立方体にするため最大距離を採用
        max_distance = max(x_min_dist, x_max_dist, y_max_dist, z_min_dist, z_max_dist)
        margin = 5
        voxel_scale = max_distance * 2 + margin
        
        # X,Z軸は中心に、Y軸は地面から開始
        x_center_final = x_center
        z_center_final = z_center
        y_start = y_ground  # Y軸は地面から開始
        
    else:
        # 従来の方法と同じ処理
        x_vals = [DG.nodes[node]['node'].pos.x for node in DG.nodes()]
        y_vals = [DG.nodes[node]['node'].pos.y for node in DG.nodes()]
        z_vals = [DG.nodes[node]['node'].pos.z for node in DG.nodes()]
        
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)
        
        x_center_final = (x_max + x_min) / 2
        z_center_final = (z_max + z_min) / 2
        y_start = y_min  # Y軸は最小値から開始
        
        max_length = max(x_max - x_min, y_max - y_min, z_max - z_min)
        margin = 5
        voxel_scale = max_length + margin
    
    # 共通の処理
    voxel_size = voxel_scale / H
    print(f"voxel_scale: {voxel_scale}, voxel_size: {voxel_size}")
    
    # X,Z軸は中心配置、Y軸は地面から開始
    voxel_xmin = x_center_final - voxel_scale / 2
    voxel_ymin = y_start-2  # Y軸は地面から開始
    voxel_zmin = z_center_final - voxel_scale / 2
    
    # 座標軸の値を計算
    x_coords = voxel_xmin + np.arange(H) * voxel_size
    y_coords = voxel_ymin + np.arange(W) * voxel_size
    z_coords = voxel_zmin + np.arange(D) * voxel_size
    
    # メッシュグリッドを作成
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    voxel_positions = np.stack((X, Y, Z), axis=-1)
    
    return voxel_data, voxel_positions, x_coords, y_coords, z_coords, voxel_size
# def initialize_voxel_data(DG, H=32, W=32, D=32):
#         # ボクセルデータを初期化
#         voxel_data = np.full((H, W, D), -1, dtype=float)
        
#         # ノードの座標を取得(x, y, z)
#         x_vals = [DG.nodes[node]['node'].pos.x for node in DG.nodes()]
#         y_vals = [DG.nodes[node]['node'].pos.y for node in DG.nodes()]
#         z_vals = [DG.nodes[node]['node'].pos.z for node in DG.nodes()]
#         # ノードの座標の最小値と最大値を取得
#         x_min, x_max = min(x_vals), max(x_vals)
#         y_min, y_max = min(y_vals), max(y_vals)
#         z_min, z_max = min(z_vals), max(z_vals)
#         print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, z_min: {z_min}, z_max: {z_max}")
        
#         # 配列の中心を計算
#         x_center = (x_max + x_min) / 2
#         y_center = (y_max + y_min) / 2
#         z_center = (z_max + z_min) / 2
        
#         # ボクセルスケールとサイズを計算
#         max_length = max(x_max - x_min, y_max - y_min, z_max - z_min)
#         margin = 5  # 余裕を持たせる
#         voxel_scale = max_length + margin #立方体の一辺の長さ  
#         voxel_size = voxel_scale / H #ボクセルの一辺の長さ
#         print(voxel_size)
#         voxel_xmin = x_center - voxel_scale / 2 #X軸の最小値
#         voxel_ymin = y_center - voxel_scale / 2 #Y軸の最小値
#         voxel_zmin = z_center - voxel_scale / 2 #Z軸の最小値
        
#         # 座標軸の値を計算
#         x_coords = voxel_xmin + np.arange(H) * voxel_size
#         y_coords = voxel_ymin + np.arange(W) * voxel_size
#         z_coords = voxel_zmin + np.arange(D) * voxel_size
        
#         # メッシュグリッドを作成
#         X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
#         #これで、ijkでアクセスできるようになる
#         voxel_positions = np.stack((X, Y, Z), axis=-1)
        
#         return voxel_data, voxel_positions, x_coords, y_coords, z_coords, voxel_size

from regex import X
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull, Delaunay
def draw_sphere(center, radius, voxel_positions, voxel_data, value,x_coords,y_coords,z_coords):
    """
    center: 球の中心（x,y,z）
    radius: 球の半径
    voxel_positions: 4次元配列（H, W, D, 3）
    voxel_data: ボクセル値配列（H, W, D）
    value: 割り当てたい値（1:幹など）
    """
    X_min,X_max=center[0]-radius,center[0]+radius
    Y_min,Y_max=center[1]-radius,center[1]+radius
    Z_min,Z_max=center[2]-radius,center[2]+radius
    i_min = np.searchsorted(x_coords, X_min, side='left')
    i_max = np.searchsorted(x_coords, X_max, side='right') - 1
    j_min = np.searchsorted(y_coords, Y_min, side='left')
    j_max = np.searchsorted(y_coords, Y_max, side='right') - 1
    k_min = np.searchsorted(z_coords, Z_min, side='left')
    k_max = np.searchsorted(z_coords, Z_max, side='right') - 1
    diff = voxel_positions[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1, :] - center  # 全ボクセルと中心の差分
    dist = np.linalg.norm(diff, axis=-1)
    mask = dist <= radius
    voxel_data[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1][mask] = np.maximum(voxel_data[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1][mask], value)  #     優先順位を保った上書き
import numpy as np

def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R

def align_DG_to_Y(DG):
    # 幹部分を取得
    trunk_positions = np.array([
        [DG.nodes[node]['node'].pos.x,
         DG.nodes[node]['node'].pos.y,
         DG.nodes[node]['node'].pos.z]
        for node in DG.nodes() if DG.nodes[node]['node'].attr == 1
    ])

    # 幹のroot（Z最小）とtip（Z最大）を取得

    root_idx_z = np.argmin(trunk_positions[:, 2])
    tip_idx_z = np.argmax(trunk_positions[:, 2])

    root_idx_y = np.argmin(trunk_positions[:, 1])
    tip_idx_y = np.argmax(trunk_positions[:, 1])

    root_idx_x = np.argmin(trunk_positions[:, 0])
    tip_idx_x = np.argmax(trunk_positions[:, 0])

    root_pos_z = trunk_positions[root_idx_z]
    tip_pos_z = trunk_positions[tip_idx_z]

    root_pos_y = trunk_positions[root_idx_y]
    tip_pos_y = trunk_positions[tip_idx_y]

    root_pos_x = trunk_positions[root_idx_x]
    tip_pos_x = trunk_positions[tip_idx_x]
    
    # print(f"root_pos_z: {root_pos_z}, tip_pos_z: {tip_pos_z}"
    #       f"\nroot_pos_y: {root_pos_y}, tip_pos_y: {tip_pos_y}"
    #       f"\nroot_pos_x: {root_pos_x}, tip_pos_x: {tip_pos_x}")

    # 幹ベクトルを求める
    trunk_norm_z = np.linalg.norm(tip_pos_z - root_pos_z)
    trunk_norm_y = np.linalg.norm(tip_pos_y - root_pos_y)
    trunk_norm_x = np.linalg.norm(tip_pos_x - root_pos_x)

    trunk_norms = [trunk_norm_x, trunk_norm_y, trunk_norm_z]
    idx = np.argmax(trunk_norms)
    trunk_vectors = [tip_pos_x - root_pos_x, tip_pos_y - root_pos_y, tip_pos_z - root_pos_z]
    axis_names = ["X軸", "Y軸", "Z軸"]
    # print(f"{axis_names[idx]}に沿った幹")
    trunk_vector = trunk_vectors[idx]

    print(f"trunk_vector: {trunk_vector}")
    
    if trunk_vector[1] < 0:
        trunk_vector = -trunk_vector
    # 幹ベクトルをx軸に揃える回転行列
    rotation_mat = rotation_matrix_from_vectors(trunk_vector, np.array([0, -2, 0]))  # Z軸に揃える

    # DGの全ノード座標に回転適用
    for node in DG.nodes():
        pos = DG.nodes[node]['node'].pos
        pos_arr = np.array([pos.x, pos.y, pos.z])
        rotated_pos = pos_arr @ rotation_mat.T

        # 回転後座標を上書き
        pos.x, pos.y, pos.z = rotated_pos.tolist()

    return DG  # 回転後DG



def create_voxel_data(DG, H=32, W=32, D=32):

    # ボクセルデータを初期化
    DG=align_DG_to_Y(DG)#座標をY軸に沿った幹に揃える
    voxel_data, voxel_positions, x_coords, y_coords, z_coords, voxel_size = initialize_voxel_data(DG, H, W, D)
    
    
    for edge in DG.edges():
        if DG.nodes[edge[0]]['node'].root_flag==True:
            continue
        start_node = DG.nodes[edge[0]]['node']#エッジの始点
        end_node = DG.nodes[edge[1]]['node']#エッジの終点
        thickness = DG.edges[edge]['edge'].thickness#エッジの太さ
        #幹の場合はエッジの太さを＋ボクセルサイズ
        if start_node.attr == 1 and end_node.attr == 1:
            thickness += voxel_size*1
        if start_node.attr == 0.5 and end_node.attr == 0.5:
            thickness += voxel_size*1
        if start_node.attr == 0.5 and end_node.attr == 0:
            thickness += voxel_size*1
        
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
        # 球の半径 = thickness / 2
        radius = DG.edges[edge]['edge'].thickness / 2
        value=0.5
        if start_node.attr == 1 and end_node.attr == 1:
            value=1
        elif start_node.attr == 0.5 and end_node.attr == 0:
            value=0

        # 球を両端に追加
        if DG.nodes[edge[0]]['node'].dammy_root_flag==False:
            draw_sphere(start_pos, radius, voxel_positions, voxel_data, value,x_coords,y_coords,z_coords)
            draw_sphere(end_pos, radius, voxel_positions, voxel_data, value,x_coords,y_coords,z_coords)

    
      
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
        # mask = within_edge & (distance <= thickness) #試し
        
        
            
        

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
    
    # trunk_positions = np.argwhere(voxel_data == 1)
    # root_idx=np.argmin(trunk_positions[:,1])
    # tip_idx=np.argmax(trunk_positions[:,1])
    # root_pos=trunk_positions[root_idx]
    # tip_pos=trunk_positions[tip_idx]
    # print(root_pos,tip_pos)

    # trank_vector=tip_pos-root_pos
    # trank_vector=trank_vector/np.linalg.norm(trank_vector)

    # Y_axis=np.array([0,1,0])
    # v=np.cross(Y_axis,trank_vector)
    # c=np.dot(Y_axis,trank_vector)
    # s=np.linalg.norm(v)
    # vx=np.array([
    #     [0,-v[2],v[1]],
    #     [v[2],0,-v[0]],
    #     [-v[1],v[0],0]
    # ])
    # rotation_matrix=np.eye(3)+vx+vx@vx*((1-c)/(s**2))
    
    # # voxel_positions: (H, W, D, 3) 形状 (各ボクセルの位置座標)
    # H, W, D,_ = voxel_positions.shape

    # # (H*W*D, 3) にフラット化して回転行列を適用
    # voxel_pos_flat = voxel_positions.reshape(-1, 3)

    # # 回転行列適用
    # rotated_voxel_pos_flat = voxel_pos_flat @ rotation_matrix.T

    # # 元の形状に戻す
    # rotated_voxel_positions = rotated_voxel_pos_flat.reshape(H, W, D, 3)

    # # 回転後のボクセルデータを初期化
    # rotated_voxel_data = np.full((H, W, D), -1, dtype=float)

    # # 回転後のボクセルデータに値を割り当て
    # filled_positions = np.argwhere(voxel_data != -1)
    # for pos in filled_positions:
    #     rotated_pos=rotated_voxel_positions[pos[0],pos[1],pos[2]]
    #     voxel_idx=int((rotated_pos[0]-x_coords[0])/voxel_size)
    #     voxel_idy=int((rotated_pos[1]-y_coords[0])/voxel_size)
    #     voxel_idz=int((rotated_pos[2]-z_coords[0])/voxel_size)
    #     rotated_voxel_data[voxel_idx,voxel_idy,voxel_idz]=voxel_data[pos[0],pos[1],pos[2]]

    return voxel_data