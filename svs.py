
import numpy as np  
import networkx as nx
import math
import os
from typing import List, Tuple
from estimate_thickness import make_davinch_tree
os.environ["OMP_NUM_THREADS"] = "2"
import matplotlib.pyplot as plt
from matplotlib import cm, figure
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix, save_npz, load_npz
from save_file import save_npzForm
from graph2Voxel import create_voxel_data
from filecount import last_file_num
from visualize_func import visualize_voxel_data,visualize_with_timeout4voxel
import random
class Edge():
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c
        self.thickness=1
        self.attr=1
        
class Pos():
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
class Node():
    def __init__(self,pos,attr,root_flag=False,dammy_root_flag=False):
        self.pos=pos
        self.attr=attr
        self.strmatrix=np.identity(4)
        self.root_flag=root_flag
        self.dammy_root_flag=dammy_root_flag
def cal_attr(depth):
    if depth==0:#幹
        return 1
    elif depth==-1:#葉
        return 0
    else:
        return 0.5 
def extract_value(cmd):
    start = cmd.find('(')
    end = cmd.find(')')
    if start != -1 and end != -1 and end > start:
        val_str = cmd[start+1:end]
        return float(val_str)
    return 0.0
def process_stack2(stack,depth):
    """
 Module Turtle Command
F(d) Move Forward by the distance d
+(α ) Turn Left by α
−(α ) Turn Right by α
\(α ) Roll Left by α
/(α ) Roll Right by α
&(α ) Pitch Down by α
^(α ) Pitch Up by α
    """
    # print(f"stack={stack}")
    rot_x=0
    rot_y=0
    rot_z=0
    L= 0.0
    for i in range(len(stack)):
        cmd=stack[i]
        angle=extract_value(cmd)
        #X軸周りの回転
        if cmd[0]=="&":
            rot_x+=angle
        elif cmd[0]=="^":
            rot_x-=angle
        #Y軸周りの回転
        elif cmd[0]=="+":
            rot_y+=angle#Turn Left
        elif cmd[0]=="-":
            rot_y-=angle#Turn Right
        #Z軸周りの回転
        elif cmd[0]=="\\":
            rot_z+=angle#Roll Left
        elif cmd[0]=="/":
            rot_z-=angle#Roll Right
        elif cmd[0]=="F":
            L=angle

    rot_x_rad=rot_x
    rot_y_rad=rot_y
    rot_z_rad=rot_z
    # 回転行列定義
    R_x = np.array([
        [1, 0,              0,             0],
        [0, math.cos(rot_x_rad), math.sin(rot_x_rad), 0],
        [0, -math.sin(rot_x_rad),  math.cos(rot_x_rad), 0],
        [0, 0,              0,             1]
    ])

    R_y = np.array([
        [math.cos(rot_y_rad), 0, -math.sin(rot_y_rad), 0],
        [0,                   1, 0,                  0],
        [math.sin(rot_y_rad),0, math.cos(rot_y_rad),0],
        [0,                   0, 0,                  1]
    ])

    R_z = np.array([
        [math.cos(rot_z_rad),  math.sin(rot_z_rad), 0, 0],
        [-math.sin(rot_z_rad),  math.cos(rot_z_rad), 0, 0],
        [0,                    0,                   1, 0],
        [0,                    0,                   0, 1]
    ])
    #平行移動行列
    T = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,L],
        [0,0,0,1]
    ])

    M=T@R_z@R_y@R_x
    return M
def cal_pos2(stack,parent_pos,strmatrix,depth):
    #stackの中身を処理する
    
    M=process_stack2(stack,depth)
    strmatrix=strmatrix@M
    
    #座標変換
    
    new_x_axis=strmatrix[0:3,0]
    new_y_axis=strmatrix[0:3,1]
    new_z_axis=strmatrix[0:3,2]
    
    #直交しているか確認
    
    # x_dot_y=np.dot(new_x_axis,new_y_axis)
    # x_dot_z=np.dot(new_x_axis,new_z_axis)
    # y_dot_z=np.dot(new_y_axis,new_z_axis)
    
    # # if x_dot_y!=0 or x_dot_z!=0 or y_dot_z!=0:
    # #     print("直交していない")
    # #     print(f"x_dot_y={x_dot_y},x_dot_z={x_dot_z},y_dot_z={y_dot_z}")
    # # else:
    # #     print("直交している")
    
    p_pos=np.array([0,0,0,1])#最初の頂点
    
    new_pos=strmatrix@p_pos
 
    return Pos(float(new_pos[0]),float(new_pos[1]),float(new_pos[2])),strmatrix
def make_node(depth,stack,parent_pos,strmatrix):
    node=Node(0,0)
    
    #属性:depthで割り当てる
    attr=cal_attr(depth)
    node.attr=attr
    
    #位置の計算:
    pos,strmatrix=cal_pos2(stack,parent_pos,strmatrix,depth)
    node.pos=pos
    node.strmatrix=strmatrix
 
    return node, strmatrix
def make_edge(DG,current_index,index):
    #辺の向きを計算
    a=DG.nodes[index]["node"].pos.x-DG.nodes[current_index]["node"].pos.x
    b=DG.nodes[index]["node"].pos.y-DG.nodes[current_index]["node"].pos.y
    c=DG.nodes[index]["node"].pos.z-DG.nodes[current_index]["node"].pos.z
    edge=Edge(a,b,c)
    return edge

DAMMY_ROOT_FLAG=True
def make_svs(l_list,depth,current_index,index,DG,stmatrix,tree_type:str):
    global DAMMY_ROOT_FLAG
    #深さ（枝、幹を分ける存在）
    stack=[]
    #print(f"len(l_list)={len(l_list)}")
    while current_index<len(l_list) and index<len(l_list):
        #print(f"current_index={current_index}")
    
        if l_list[index][0]=="F":
            #print(f"新しいノードを作成する:親{current_index}->子{index+1}")
            DG.add_node(index+1)
            DG.add_edge(current_index,index+1)
            stack.append(l_list[index])#Fをスタックに追加
            
            ##頂点属性を設定
        
            new_node,stmatrix=make_node(depth,stack,DG.nodes[current_index]["node"].pos,stmatrix)
            DG.nodes[index+1]["node"]=new_node
            
            ##辺属性を設定
            new_edge=make_edge(DG,current_index,index+1)
            DG.edges[(current_index,index+1)]["edge"]=new_edge
            #新しいノードをCurrentNodeにする
            current_index=index+1
            stack.clear()
            plot_flag=False
            if plot_flag and index%10==0:
                plot_graph(DG)
                #plot_graph_and_strmatrix(DG)
            if DAMMY_ROOT_FLAG==True:
                DG.nodes[index+1]["node"].dammy_root_flag=True
                DAMMY_ROOT_FLAG=False
            #plot_graph(DG)
        elif l_list[index][0]=="[":
            # print("分岐開始")
            index=make_svs(l_list,depth+1,current_index,index+1,DG,stmatrix,tree_type)
            # print(f"分岐終了")
  
        elif l_list[index][0]=="]":
  
            parent_list=list(DG.predecessors(current_index))
            parent_id=parent_list[0]
      
            parent_pos_x=DG.nodes[parent_id]["node"].pos.x
            parent_pos_y=DG.nodes[parent_id]["node"].pos.y
            parent_pos_z=DG.nodes[parent_id]["node"].pos.z
            
            
            #葉の位置を計
            pos_x=DG.nodes[current_index]["node"].pos.x
            pos_y=DG.nodes[current_index]["node"].pos.y
            pos_z=DG.nodes[current_index]["node"].pos.z
            
            parent_pos=np.array([parent_pos_x,parent_pos_y,parent_pos_z])
            pos=np.array([pos_x,pos_y,pos_z])
            r=np.linalg.norm(parent_pos-pos)
            
            if r<0.2:
                if tree_type=="MapleClustered" or tree_type=="PineClustered":
                    leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*10
                    leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*10
                    leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*10
                else:
                    leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*200
                    leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*200
                    leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*200
            else:
                leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*1.5
                leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*1.5
                leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*1.5
                
            #新しいコードVer２
            #posからleaf_posの角度を０とする.
            # 前提：pos_x, pos_y, pos_z, leaf_pos_x, ... はすでに定義されている
            pos = np.stack([pos_x, pos_y, pos_z])
            leaf_pos = np.stack([leaf_pos_x, leaf_pos_y, leaf_pos_z])
            r = np.linalg.norm(pos - leaf_pos)+0.01
            zero_dir = (leaf_pos - pos) / r
            point_num = 1

            # サンプリング半径（体積均一にするため立方根を使う）
            radii = r * np.cbrt(np.random.rand(point_num))

            # -60度〜60度のランダム角（ラジアン）
            angles = np.radians(np.random.uniform(-15, 15, point_num))

            # 基準ベクトル zero_dir を z軸に一致させる回転を求める
            z_axis = np.array([0.0, 0.0, 1.0])
            v = np.cross(z_axis, zero_dir)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, zero_dir)
            if s < 1e-8:
                R_align = np.eye(3) if c > 0 else -np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
                R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

            samples = []
            for radius, theta in zip(radii, angles):
                # z軸周りに回転
                rot_z = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0, 0, 1]
                ])
                # z軸 → zero_dir に回転したあと点を回して、元の中心に移動
                direction = R_align @ rot_z @ z_axis
                sample = pos + radius * direction
                samples.append(sample)

            samples = np.array(samples)

            for i,sample in enumerate(samples):
                
                # 現在のposとsample（葉の位置）の中間点を計算
                branch_pos = (pos + sample) / 2
                
                # ノードのキーを設定
                branch_key = index + len(l_list) + i + 1
                leaf_key = branch_key*2
                
                # ノードを追加
                DG.add_node(branch_key)
                DG.add_node(leaf_key)
                
                # エッジを追加（current_index → branch_key → leaf_key）
                DG.add_edge(current_index, branch_key)
                DG.add_edge(branch_key, leaf_key)
                
                # 枝ノードを作成（属性は0.5=枝）
                branch_node = Node(Pos(branch_pos[0], branch_pos[1], branch_pos[2]), 0.5)
                DG.nodes[branch_key]["node"] = branch_node
                
                # 葉ノードを作成（属性は0=葉）
                leaf_node = Node(Pos(sample[0], sample[1], sample[2]), 0)
                DG.nodes[leaf_key]["node"] = leaf_node
                
                # エッジを作成
                branch_edge = make_edge(DG, current_index, branch_key)
                DG.edges[(current_index, branch_key)]["edge"] = branch_edge
                
                leaf_edge = make_edge(DG, branch_key, leaf_key)
                DG.edges[(branch_key, leaf_key)]["edge"] = leaf_edge

            return index

            
            
            
            
            
            return index
        else:
            # print(f"コマンドをスタックに追加:{l_list[index]}"z
            stack.append(l_list[index])
        index+=1
    return index



def voxel_distribution(voxel_data):
    voxel_data=voxel_data.flatten()
    element_num=len(voxel_data)
    #0,1,2,3の数をカウント
    result_blank=voxel_data[voxel_data==-1].size
    result_trunk=voxel_data[voxel_data==1].size
    result_branch=voxel_data[voxel_data==0.5].size
    result_leaf=voxel_data[voxel_data==0].size
    
    print(f"result_blank:{result_blank/element_num*100}%")#99.9%
    print(f"result_trunk:{result_trunk/element_num*100}%")#0.01%
    print(f"result_branch:{result_branch/element_num*100}%")#0.001%
    print(f"result_leaf:{result_leaf/element_num*100}%")#0.0001%
    

def resize_with_padding(img, target_size):
    import cv2
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded


def find_best_projection_angle(voxel_data, num_angles=8, verbose=False):
    """
    Y軸周りの回転で最も印象的な投影角度を見つける関数
    
    Args:
        voxel_data: (H, W, D) ndarray, ボクセルデータ
        num_angles: int, テストする角度数
        verbose: bool, デバッグ出力
    
    Returns:
        tuple: (best_angle, best_projection, projection_score)
    """
    import numpy as np
    from scipy.ndimage import rotate
    
    # 空白部分を除外したマスクを作成
    min_val = voxel_data.min()
    structure_mask = voxel_data > min_val
    
    best_score = 0
    best_angle = 0
    best_projection = None
    
    # 360度を等分した角度でテスト
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    for i, angle in enumerate(angles):
        # Y軸周りの回転でボクセルデータを回転
        # scipy.ndimage.rotateを使用（軸の順序に注意）
        # axes=(2, 0) はZ軸とX軸の平面での回転 = Y軸周りの回転
        rotated_voxel = rotate(
            voxel_data, 
            angle=np.degrees(angle), 
            axes=(0, 2),  # Z-X平面での回転（Y軸周り）
            reshape=False,  # 形状を保持
            order=0,  # 最近傍補間（ボクセルデータに適している）
            mode='constant',  # 境界の処理
            cval=min_val  # 境界外の値
        )
    
        
        # X軸方向に投影（正面から見た図）
        projection = np.max(rotated_voxel, axis=0)  # X軸方向に投影
        
        # 上下反転
        projection = np.flip(projection, axis=0)
        
        # 投影画像の確認（デバッグ用）
        if verbose:
            import cv2
            print("投影画像の保存: projection.png")
            cv2.imwrite("projection.png", (projection * 255).astype(np.uint8))

        # 投影の品質を評価
        score = evaluate_projection_quality(projection, verbose and i == 0)
        
        if verbose:
            print(f"角度 {np.degrees(angle):.1f}度: スコア = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_angle = angle
            best_projection = projection.copy()
    
    if verbose:
        print(f"最適角度: {np.degrees(best_angle):.1f}度 (スコア: {best_score:.3f})")
    
    return best_angle, best_projection, best_score

def evaluate_projection_quality(projection, verbose=False):
    """
    投影の品質を評価する関数
    幹の可視性、水平方向の広がり、分岐の見やすさを重視
    
    Args:
        projection: 2D投影画像
        verbose: bool, デバッグ出力
    
    Returns:
        float: 品質スコア（高いほど良い）
    """
    import cv2
    import numpy as np
    
    if np.sum(projection > 0) == 0:
        return 0.0
    
    # バイナリマスクを作成
    binary_proj = (projection > 0).astype(np.uint8)
    
    # 1. 幹の可視性（枝による被りを考慮した可視性）
    trunk_visibility = 0
    if np.sum(binary_proj) > 0:
        # 画像の中央部分（幹がある可能性が高い領域）に重点を置く
        height, width = binary_proj.shape
        center_region = binary_proj[:, width//4:3*width//4]  # 中央50%の幅
        
        if np.sum(center_region) > 0:
            # 中央領域での幹の存在感を評価（垂直連続性は緩く評価）
            central_presence = 0
            for col in range(center_region.shape[1]):
                col_pixels = center_region[:, col]
                if np.sum(col_pixels) > 0:
                    # 各列でのピクセル密度を評価（連続性よりも存在感重視）
                    column_density = np.sum(col_pixels) / len(col_pixels)
                    central_presence += column_density
            
            # 中央領域の列数で平均化
            if center_region.shape[1] > 0:
                trunk_visibility = central_presence / center_region.shape[1]
        
        # 全体領域での幹の露出度もチェック
        # 下部から上部へのピクセル密度の変化を評価（幹は下部が太い）
        row_densities = np.sum(binary_proj, axis=1) / width
        if len(row_densities) > 0:
            # 下部3分の1の密度
            lower_third = row_densities[2*height//3:]
            # 上部3分の1の密度
            upper_third = row_densities[:height//3]
            
            if len(lower_third) > 0 and len(upper_third) > 0:
                lower_density = np.mean(lower_third)
                upper_density = np.mean(upper_third)
                
                # 幹らしい形状（下部が太く上部が細い）を評価
                trunk_shape_score = min(lower_density / (upper_density + 0.01), 2.0) / 2.0
                
                # 中央存在感（80%）と幹らしい形状（20%）を組み合わせ
                trunk_visibility = trunk_visibility * 0.8 + trunk_shape_score * 0.2
    
    # 2. 水平方向の広がり（枝の展開）
    horizontal_spread = 0
    if np.sum(binary_proj) > 0:
        non_zero_cols = np.any(binary_proj > 0, axis=0)
        if np.sum(non_zero_cols) > 0:
            first_col = np.argmax(non_zero_cols)
            last_col = len(non_zero_cols) - 1 - np.argmax(non_zero_cols[::-1])
            horizontal_spread = (last_col - first_col) / binary_proj.shape[1]
    
    # 3. 分岐の見やすさ（エッジの複雑さと分散）
    branch_visibility = 0
    if np.sum(binary_proj) > 0:
        # エッジ検出で分岐構造を評価
        edges = cv2.Canny(binary_proj * 255, 30, 120)
        edge_density = np.sum(edges > 0) / (binary_proj.shape[0] * binary_proj.shape[1])
        
        # 水平方向のエッジ分散（分岐の広がり具合）
        if np.sum(edges) > 0:
            edge_positions = np.where(edges > 0)
            if len(edge_positions[1]) > 0:
                horizontal_variance = np.var(edge_positions[1]) / (binary_proj.shape[1] ** 2)
                branch_visibility = edge_density * 0.7 + horizontal_variance * 0.3
            else:
                branch_visibility = edge_density
    
    # 4. 構造の密度バランス（適度な密度を評価）
    density = np.sum(binary_proj > 0) / (binary_proj.shape[0] * binary_proj.shape[1])
    density_score = min(density * 8, 1.0)  # 適度な密度を評価
    
    # 5. 垂直分布の均等性（上下に構造が分散している）
    vertical_distribution = 0
    if np.sum(binary_proj) > 0:
        row_sums = np.sum(binary_proj, axis=1)
        non_zero_rows = row_sums > 0
        if np.sum(non_zero_rows) > 0:
            # 上部、中部、下部に構造があるかチェック
            height = binary_proj.shape[0]
            upper_third = np.sum(non_zero_rows[:height//3]) > 0
            middle_third = np.sum(non_zero_rows[height//3:2*height//3]) > 0
            lower_third = np.sum(non_zero_rows[2*height//3:]) > 0
            vertical_distribution = (upper_third + middle_third + lower_third) / 3.0
    
    # 重み付き総合スコア
    total_score = (
        trunk_visibility * 0 +         # 幹の可視性（最重要）
        horizontal_spread * 0.5 +        # 水平方向の広がり
        branch_visibility * 0.25 +        # 分岐の見やすさ
        density_score * 0 +            # 密度バランス
        vertical_distribution * 0      # 垂直分布
    )
    
    if verbose:
        print(f"  幹の可視性: {trunk_visibility:.3f}")
        print(f"  水平広がり: {horizontal_spread:.3f}")
        print(f"  分岐の見やすさ: {branch_visibility:.3f}")
        print(f"  密度スコア: {density_score:.3f}")
        print(f"  垂直分布: {vertical_distribution:.3f}")
        print(f"  総合スコア: {total_score:.3f}")
    
    return total_score

def rotate_voxel_data_y_axis(voxel_data, angle, verbose=False):
    """
    ボクセルデータをY軸周りに実際に回転させる関数
    
    Args:
        voxel_data: (H, W, D) ndarray, 回転させるボクセルデータ
        angle: float, 回転角度（ラジアン）
        verbose: bool, デバッグ出力
    
    Returns:
        numpy.ndarray: 回転後のボクセルデータ
    """
    import numpy as np
    from scipy.ndimage import rotate
    
    if verbose:
        print(f"[INFO] ボクセルデータをY軸周りに{np.degrees(angle):.1f}度回転中...")
    
    # Y軸周りの回転のため、X-Z平面で回転
    # scipy.ndimage.rotateを使用（軸の順序に注意）
    # axes=(2, 0) はZ軸とX軸の平面での回転 = Y軸周りの回転
    rotated_voxel = rotate(
        voxel_data, 
        angle=np.degrees(angle), 
        axes=(2, 0),  # Z-X平面での回転（Y軸周り）
        reshape=False,  # 形状を保持
        order=0,  # 最近傍補間（ボクセルデータに適している）
        mode='constant',  # 境界の処理
        cval=voxel_data.min()  # 境界外の値
    )
    
    if verbose:
        print(f"[INFO] 回転完了: {voxel_data.shape} -> {rotated_voxel.shape}")
    
    return rotated_voxel

def get_rotated_projections(voxel_data, angle, unique_vals, min_val, max_val, verbose=False):
    """
    指定角度でボクセルデータを実際に回転させ、構造要素ごとの投影を返す
    """
    import numpy as np
    
    # ボクセルデータを実際に回転
    rotated_voxel = rotate_voxel_data_y_axis(voxel_data, angle, verbose)
    
    h, w, d = rotated_voxel.shape
    
    # 構造要素のマスクを作成
    # 幹=1, 枝=0.5, 葉=0, 空白=-1 の定義に基づく
    if len(unique_vals) <= 2:
        # 2値データの場合（空白=-1と構造=0または1）
        trunk_mask = (rotated_voxel == 1)
        branch_mask = np.zeros_like(rotated_voxel, dtype=bool)
        leaf_mask = (rotated_voxel == 0)
    elif len(unique_vals) <= 4:
        # 3-4値データの場合（空白=-1, 葉=0, 枝=0.5, 幹=1）
        trunk_mask = (rotated_voxel == 1)
        branch_mask = (rotated_voxel == 0.5)
        leaf_mask = (rotated_voxel == 0)
    else:
        # 連続値データの場合
        trunk_mask = (rotated_voxel >= 0.9) & (rotated_voxel <= 1.0)
        branch_mask = (rotated_voxel >= 0.3) & (rotated_voxel < 0.9)
        leaf_mask = (rotated_voxel >= 0) & (rotated_voxel < 0.3)
    
    # 回転後のボクセルデータから直接投影を作成（X方向に投影）
    # 注意: 葉の値が0なので、マスクを直接使用して投影
    trunk_proj = np.max(trunk_mask.astype(float), axis=1)  # X軸方向に投影
    branch_proj = np.max(branch_mask.astype(float), axis=1)
    leaf_proj = np.max(leaf_mask.astype(float), axis=1)

    return trunk_proj, branch_proj, leaf_proj, rotated_voxel


def make_enhanced_sketch_multicolor(
    voxel_data,
    output_size=(224, 224),
    save_path="enhanced_sketch.png",
    verbose=True,
    auto_angle=True,
    fixed_angle=None
):
    """
    マルチカラーでより詳細なスケッチ画像を生成する関数
    最適な投影角度を自動選択し、ボクセルデータを実際に回転させる
    
    Args:
        voxel_data: (H, W, D) ndarray, ボクセルデータ
        output_size: tuple, 出力画像サイズ (width, height)
        save_path: str, 保存先ファイル名
        verbose: bool, ログ出力
        auto_angle: bool, 最適角度を自動選択するか
        fixed_angle: float, 固定角度（ラジアン、auto_angle=Falseの場合）
    
    Returns:
        tuple: (color_image_resized, rotated_voxel_data, best_angle)
    """
    import numpy as np
    import cv2
    from PIL import Image
    import os
    
    # データの値を確認
    unique_vals = np.unique(voxel_data)
    
    # --- 1. 最適な投影角度を選択 ---
    if auto_angle:
        if verbose:
            print("[INFO] 最適な投影角度を探索中...")
        best_angle, best_projection_raw, best_score = find_best_projection_angle(
            voxel_data, num_angles=16, verbose=verbose
        )
        if verbose:
            print(f"[INFO] 選択された角度: {np.degrees(best_angle):.1f}度")
    else:
        # 固定角度を使用
        angle = fixed_angle if fixed_angle is not None else 0.0
        best_angle = angle
        if verbose:
            print(f"[INFO] 固定角度を使用: {np.degrees(angle):.1f}度")
    
    # --- 2. ボクセルデータを最適角度で実際に回転 ---
    rotated_voxel = rotate_voxel_data_y_axis(voxel_data, best_angle, verbose)
    
    # --- 3. 回転後のボクセルデータから構造要素を分離 ---
    min_val = rotated_voxel.min()
    max_val = rotated_voxel.max()
    
    # 構造要素のマスクを作成
    # 幹=1, 枝=0.5, 葉=0, 空白=-1 の定義に基づく
    if len(unique_vals) <= 2:
        # 2値データの場合（空白=-1と構造=0または1）
        trunk_mask = (rotated_voxel == 1)
        branch_mask = np.zeros_like(rotated_voxel, dtype=bool)
        leaf_mask = (rotated_voxel == 0)
    elif len(unique_vals) <= 4:
        # 3-4値データの場合（空白=-1, 葉=0, 枝=0.5, 幹=1）
        trunk_mask = (rotated_voxel == 1)
        branch_mask = (rotated_voxel == 0.5)
        leaf_mask = (rotated_voxel == 0)
    else:
        # 連続値データの場合
        trunk_mask = (rotated_voxel >= 0.9) & (rotated_voxel <= 1.0)
        branch_mask = (rotated_voxel >= 0.3) & (rotated_voxel < 0.9)
        leaf_mask = (rotated_voxel >= 0) & (rotated_voxel < 0.3)
    
    # --- 4. 正面投影（X軸方向に投影） ---
    # 注意: 葉の値が0なので、マスクを直接使用して投影
    trunk_proj = np.max(trunk_mask.astype(float), axis=0)
    branch_proj = np.max(branch_mask.astype(float), axis=0) 
    leaf_proj = np.max(leaf_mask.astype(float), axis=0)
    
    # 上下反転
    trunk_proj = np.flip(trunk_proj, axis=0)
    branch_proj = np.flip(branch_proj, axis=0)
    leaf_proj = np.flip(leaf_proj, axis=0)
    
    # --- 5. 構造強調処理 ---
    # 幹の強調
    if np.sum(trunk_proj) > 0:
        trunk_binary = trunk_proj > 0
        kernel = np.ones((3, 3), np.uint8)
        trunk_enhanced = cv2.morphologyEx(trunk_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        kernel_dilate = np.ones((4, 4), np.uint8)
        trunk_enhanced = cv2.dilate(trunk_enhanced, kernel_dilate, iterations=2)
    else:
        trunk_enhanced = trunk_proj.astype(np.uint8)
    
    # 枝の強調（分岐構造を強調、太さは変更しない）
    if np.sum(branch_proj) > 0:
        branch_binary = branch_proj > 0
        branch_enhanced = branch_binary.astype(np.uint8)  # 元の太さを保持
        
        # 1. Y字型分岐を検出・強調
        # 分岐検出（MORPH_HIT_MISSを使わない安全な方法）
        # まず枝を軽く細くして骨格に近い状態にする
        kernel_thin = np.ones((2, 2), np.uint8)
        branch_thinned = cv2.erode(branch_enhanced, kernel_thin, iterations=1)
        
        # 分岐点検出：近隣8ピクセルでの接続数を数える
        kernel_detect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(branch_thinned.astype(np.float32), -1, kernel_detect.astype(np.float32))
        
        # 3つ以上の接続がある点を分岐点とする
        branch_junctions = ((neighbor_count >= 3) & (branch_thinned > 0)).astype(np.uint8)
        
        # 2. 分岐点のみを軽く強調（太くしない）
        kernel_junction = np.ones((2, 2), np.uint8)  # 小さなカーネルで最小限の強調
        branch_junction_enhanced = cv2.dilate(branch_junctions, kernel_junction, iterations=1)
        
        # 3. 枝の連続性を保持（太さは変えない）
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 小さなカーネル
        branch_connected = cv2.morphologyEx(branch_enhanced, cv2.MORPH_CLOSE, kernel_connect)
        
        # 4. 最終的な枝の強調画像を作成（元の太さ + 分岐点強調のみ）
        branch_enhanced = branch_connected | branch_junction_enhanced
        
        # 5. 軽いノイズ除去のみ
        kernel_clean = np.ones((2, 2), np.uint8)
        branch_enhanced = cv2.morphologyEx(branch_enhanced, cv2.MORPH_OPEN, kernel_clean)
            
    else:
        branch_enhanced = branch_proj.astype(np.uint8)
    
    # 葉の強調（クラスター状に自然化）
    if np.sum(leaf_proj) > 0:
        leaf_binary = leaf_proj > 0
        
        # 1. 小さな葉の粒を統合してクラスター化
        kernel_cluster = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        leaf_clustered = cv2.morphologyEx(leaf_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel_cluster)
        
        # 2. ガウシアンブラーで自然な広がりを作成
        leaf_blurred = cv2.GaussianBlur(leaf_clustered.astype(np.float32), (7, 7), 1.5)
        leaf_soft = (leaf_blurred > 0.2).astype(np.uint8)  # より低い閾値で広がりを保持
        
        # 3. 距離変換を使ってクラスターの中心部を強調
        dist_transform = cv2.distanceTransform(leaf_soft, cv2.DIST_L2, 5)
        dist_normalized = dist_transform / (np.max(dist_transform) + 1e-6)
        
        # 4. クラスターの中心部とエッジ部を分離
        cluster_core = (dist_normalized > 0.4).astype(np.uint8)  # 中心部
        cluster_edge = ((dist_normalized > 0.15) & (dist_normalized <= 0.4)).astype(np.uint8)  # エッジ部
        
        # 5. エッジ部をより薄く表現するために膨張処理
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cluster_edge_expanded = cv2.dilate(cluster_edge, kernel_edge, iterations=1)
        
        # 6. 最終的な葉のマスクを作成（中心部 + エッジ部）
        leaf_enhanced = cluster_core | cluster_edge_expanded
        
        # 7. 孤立した小さなピクセルを除去
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        leaf_enhanced = cv2.morphologyEx(leaf_enhanced, cv2.MORPH_OPEN, kernel_clean)
        
        # 8. 最終的なクラスター形状の調整
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        leaf_enhanced = cv2.morphologyEx(leaf_enhanced, cv2.MORPH_CLOSE, kernel_final)
            
    else:
        leaf_enhanced = leaf_proj.astype(np.uint8)
    
    # --- 6. 構造要素の優先度調整 ---
    # 優先度: 幹 > 葉 > 枝
    
    # 枝が最低優先度なので、葉がある場所では枝を非表示
    if np.sum(branch_enhanced) > 0 and np.sum(leaf_enhanced) > 0:
        leaf_priority_mask = leaf_enhanced > 0
        branch_enhanced = branch_enhanced & ~leaf_priority_mask
    
    # 枝が最低優先度なので、幹がある場所では枝を非表示
    if np.sum(branch_enhanced) > 0 and np.sum(trunk_enhanced) > 0:
        trunk_priority_mask = trunk_enhanced > 0
        branch_enhanced = branch_enhanced & ~trunk_priority_mask
    
    # 葉が幹より低い優先度なので、幹がある場所では葉を非表示
    if np.sum(leaf_enhanced) > 0 and np.sum(trunk_enhanced) > 0:
        trunk_priority_mask = trunk_enhanced > 0
        leaf_enhanced = leaf_enhanced & ~trunk_priority_mask
    
    # --- 7. カラー合成 ---
    height, width = trunk_proj.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 色の定義
    trunk_color = np.array([101, 67, 33])     # 濃い茶色（幹）
    branch_color = np.array([160, 100, 50])   # 明るい茶色（枝）  
    leaf_color = np.array([50, 150, 50])      # 緑色（葉）
    background_color = np.array([240, 240, 240])  # 薄いグレー（背景）
    
    # 背景設定
    color_image[:, :] = background_color
    
    # 構造のマスクを作成
    structure_mask = (trunk_enhanced > 0) | (branch_enhanced > 0) | (leaf_enhanced > 0)
    
    # 各要素を重ね合わせ
    leaf_mask_final = leaf_enhanced > 0
    branch_mask_final = branch_enhanced > 0
    trunk_mask_final = trunk_enhanced > 0
    
    for i in range(3):
        # 構造がある部分を白背景に
        color_image[structure_mask, i] = 255
        
        # 各構造要素を描画
        color_image[leaf_mask_final, i] = leaf_color[i]
        color_image[branch_mask_final, i] = branch_color[i]  
        color_image[trunk_mask_final, i] = trunk_color[i]
    
    # --- 8. エッジ強調で構造を明確に ---
    edges_combined = np.zeros((height, width), dtype=np.uint8)
    
    if np.sum(trunk_enhanced) > 0:
        edges_trunk = cv2.Canny(trunk_enhanced * 255, 30, 100)
        edges_combined = np.maximum(edges_combined, edges_trunk)
    
    if np.sum(branch_enhanced) > 0:
        # 枝の分岐構造を強調するエッジ検出
        edges_branch = cv2.Canny(branch_enhanced * 255, 15, 70)  # より低い閾値で細かい分岐も検出
        
        # Sobelフィルタで分岐の方向性を強調
        sobel_x = cv2.Sobel(branch_enhanced.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(branch_enhanced.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_edges = (sobel_magnitude > 0.3).astype(np.uint8) * 255
        
        # Cannyエッジとソーベルエッジを結合
        edges_branch_combined = np.maximum(edges_branch, sobel_edges)
        edges_combined = np.maximum(edges_combined, edges_branch_combined)
    
    if np.sum(leaf_enhanced) > 0:
        # 葉のソフトなエッジ検出（低いしきい値で自然な境界線を）
        edges_leaf = cv2.Canny((leaf_enhanced * 255).astype(np.uint8), 5, 30)
        
        # 葉のエッジをさらにソフトにするためのガウシアンフィルタ
        edges_leaf = cv2.GaussianBlur(edges_leaf.astype(np.float32), (3, 3), 0.5)
        edges_leaf = (edges_leaf > 10).astype(np.uint8) * 255
        
        edges_combined = np.maximum(edges_combined, edges_leaf)
    
    # エッジを黒線として追加
    edge_mask = edges_combined > 0
    color_image[edge_mask] = [0, 0, 0]
    
    # --- 9. サイズ調整と保存 ---
    color_image_resized = cv2.resize(color_image, output_size, interpolation=cv2.INTER_CUBIC)
    
    # 保存先ディレクトリの作成（ディレクトリが存在する場合のみ）
    save_dir = os.path.dirname(save_path)
    if save_dir:  # 空文字列でない場合のみディレクトリを作成
        os.makedirs(save_dir, exist_ok=True)
    
    # PIL Imageとして保存
    pil_image = Image.fromarray(color_image_resized)
    pil_image.save(save_path, format='PNG')
    
    if verbose:
        print(f"[INFO] Enhanced multicolor sketch saved to {save_path}")
        print(f"[INFO] Image size: {pil_image.size}")
        print(f"[INFO] Best angle used: {np.degrees(best_angle):.1f} degrees")
        print(f"[INFO] Voxel data rotated by {np.degrees(best_angle):.1f} degrees")
    
    return color_image_resized, rotated_voxel, best_angle


def initialize_tree() -> Tuple[nx.DiGraph, np.ndarray]:
    """
    ルートノードと変換行列の初期化を行う。

    Returns:
        Tuple[nx.DiGraph, np.ndarray]: 初期化済みのグラフと4x4の単位行列
    """
    pos = Pos(0, 0, 0)
    root = Node(pos, 1, True)
    graph = nx.DiGraph()
    graph.add_node(1)
    graph.nodes[1]["node"] = root
    transform_matrix = np.identity(4)
    return graph, transform_matrix

def read_lstring_file(file_path: str) -> List[str]:
    """
    lstringファイルを読み込み、行ごとのリストを返す。

    Args:
        file_path (str): 読み込むファイルのパス

    Returns:
        List[str]: ファイル内の各行のリスト
    """
    with open(file_path, "r") as file:
        return file.readlines()

def delete_short_branches(DG, min_length=4):
    """短い枝を削除する補助関数"""
    # 実装省略 - 元の実装を使用
    return DG

def prune_branches(DG, prune_last_n=3, min_depth_to_prune=6, root=None, verbose=True):
    """
    指定した深さより深い葉に対して、末尾nエッジを剪定する。
    """
    # 実装省略 - 元の実装を保持
    return DG





def prune_branches(DG, prune_last_n=3, min_depth_to_prune=6, root=None, verbose=True):
    """
    指定した深さより深い葉に対して、末尾nエッジを剪定する。
    - prune_last_n: 末尾から消すエッジ数（例:3なら葉から遡って3つ分消す）
    - min_depth_to_prune: これより深い葉のみ剪定（例:6なら深さ6以上の葉のみ）
    - root: Noneなら自動で入次数0のノード
    """

    # root自動検出（入次数0）
    # DG=delete_branches(DG)
    DG=delete_short_branches(DG,min_length=4)
    if root is None:
        roots = [n for n in DG.nodes if DG.in_degree(n)==0]
        if len(roots)==0:
            raise ValueError("Rootが見つからない")
        root = roots[0]

    
    leaves = [n for n in DG.nodes if DG.out_degree(n) == 0]
    print(f"length of leaves:{len(leaves)}")
    depths = nx.single_source_shortest_path_length(DG, root)
    max_depth = max(depths.values())
    edges_to_remove = set()

    # for leaf in leaves:
    #     if leaf not in depths:  # disconnected leaf
    #         continue
    #     path = nx.shortest_path(DG, source=root, target=leaf)
    #     if len(path) > min_depth_to_prune:
    #         # 末尾nエッジを消す
    #         for i in range(1, min(prune_last_n+1, len(path))):
    #             edges_to_remove.add((path[-(i+1)], path[-i]))
    

    G_pruned = DG.copy()
    G_pruned.remove_edges_from(edges_to_remove)
    # 出自も入次数も0のノードを削除
    nodes_to_remove = [n for n in G_pruned.nodes if G_pruned.out_degree(n)==0 and G_pruned.in_degree(n)==0]
    G_pruned.remove_nodes_from(nodes_to_remove)

    return G_pruned
def adjust_file_indices(
    tree_category: str,
    saved_file_path: str,
    total_count: int = 30000
) -> Tuple[int, int, int]:
    """
    ファイルインデックスを調整する関数。
    
    Args:
        tree_category (str): ツリー種別のフォルダ名（例："AcaciaClustered"）
        saved_file_path (str): 保存済みファイルのパス
        total_count (int, optional): 処理するファイル数の上限。デフォルトは30000。
    
    Returns:
        Tuple[int, int]: 
            - 開始保存ファイルインデックス
            - 新規作成ファイル数
    """
    # 保存済みファイルのパスから現在のファイルリストを取得
    current_file_list = os.listdir(saved_file_path)
    current_file_list = [f for f in current_file_list if f.startswith(tree_category) and f.endswith('.png')]
    # 現在のファイル番号を抽出
    current_file_number_list = [f.split('_')[-1].split('.')[0] for f in current_file_list]
    # 開始保存ファイルインデックスを計算(最大の番号+1)
    start_savefile_index = max(map(int, current_file_number_list), default=0) + 1
    # 現在のファイル数を取得
    saved_file_length = len(current_file_list)
    print(f"Saved files for {tree_category}: {saved_file_length} files")
    # 新規作成ファイル数を計算
    new_created_file_num = total_count - saved_file_length

    return start_savefile_index, new_created_file_num
def process_tree_category(
    tree_folder: str,
    file_prefix: str,
    input_base_dir: str,
    output_sketch_dir: str,
    output_dir: str,
    total_count: int,
    make_svs_dataset: bool,
    make_sketch_dataset: bool,
    visualize_flag: bool,
    voxel_dims: Tuple[int, int, int] = (64, 64, 64),
    convex_hull_flag: bool = False,
    uwagaki_flag: bool = False,
    trunk_thick: int = 1,
    branch_thick: int = 1,
    leaf_thick: int = 1,

):
    """
    指定したツリー種別のデータ処理を行う。
    
    入力ファイルは以下のパス形式を想定している：
        {input_base_dir}/{tree_folder}/{tree_folder}/{file_prefix}_{i}.lstring
    出力はoutput_dirへ保存する。
    
    Args:
        tree_folder (str): ツリー種別のフォルダ名（例："AcaciaClustered"）
        file_prefix (str): ファイル名のプレフィックス（例："Acacia"）
        input_base_dir (str): l-stringファイルのベースディレクトリ
        output_dir (str): データセット保存用ディレクトリ
        total_count (int): 処理するファイル数の上限
        make_svs_dataset (bool): npz形式でデータセットを保存するか否か
        make_sketch_dataset (bool): スケッチ用データセットを生成するか否か
        visualize_flag (bool): 可視化処理を行うか否か
        voxel_dims (Tuple[int, int, int], optional): ボクセル生成時の各次元のサイズ。デフォルトは(256, 256, 256)。
        start_file_index (int): 開始ファイルインデックス（連続番号管理用）
        current_num:既に処理したファイル数（樹種別）（Int)
    
    Returns:
        int: 次の開始ファイルインデックス
    """
    # ファイルインデックスの管理（全種類を通して連続番号にする）

    start_savefile_index,new_created_file_num = adjust_file_indices(
        tree_category=file_prefix,
        saved_file_path=output_sketch_dir,
        total_count=total_count
    )

  
    if new_created_file_num <= 0:
        print(f"Skipping {tree_folder} as no new files to process.")
        return
    from tqdm import tqdm
    for i  in tqdm(range(new_created_file_num), desc=f"Processing {tree_folder}"):
        # 入力ファイルパスの組み立て（元のlstringファイルのインデックス）
        # ファイル名は1から開始するため+1する
        lstring_index = i + start_savefile_index
        file_path = os.path.join(input_base_dir, tree_folder, tree_folder, f"{file_prefix}_{lstring_index}.lstring")
    
        # 出力ファイルのインデックス（データセット全体で連続番号）
        output_index = start_savefile_index + i
        
        print(f"Processing lstring file: {file_path} -> output index: {output_index}")
        lstring_lines = read_lstring_file(file_path)

        # グラフと変換行列の初期化
        graph, transform_matrix = initialize_tree()

        
    
        # SVS生成とツリー構築
        make_svs(lstring_lines, 0, 1, 0, graph, transform_matrix,tree_type=tree_folder)
        global DAMMY_ROOT_FLAG
        DAMMY_ROOT_FLAG = True  # ダミールートノードを使用するフラグを設定
        make_davinch_tree(graph, (1, 2))


        # ボクセルデータの生成（パラメータで指定された厚さを使用）
        voxel_data = create_voxel_data(graph, *voxel_dims, trunk_thick=trunk_thick, branch_thick=branch_thick, leaf_thick=leaf_thick)

        if make_sketch_dataset:
            # スケッチ画像の生成（最適角度選択機能付き・ボクセル回転対応）
            sketch_path = os.path.join(output_sketch_dir, f"{file_prefix}_{output_index}.png")
            
            # 改良版スケッチ生成を使用（ボクセル回転機能付き）
            try:
                # 新しい関数を使用（ボクセルデータ自体も回転される）
                result_image, rotated_voxel, best_angle = make_enhanced_sketch_multicolor(
                    voxel_data,
                    output_size=(224, 224),  # 出力画像サイズ
                    save_path=sketch_path,
                    verbose=False,
                    auto_angle=True  # 最適角度を自動選択
                )
                #print(f"スケッチ画像を保存: {sketch_path} (最適角度: {np.degrees(best_angle):.1f}度)")
                #print(f"ボクセルデータもY軸周りに{np.degrees(best_angle):.1f}度回転しました")
                
                # 回転後のボクセルデータを使用（必要に応じて元のvoxel_dataを置き換え）
                voxel_data = rotated_voxel
                
            except Exception as e:
                #print(f"改良版スケッチ生成エラー: {e}")
                # フォールバック: 従来の関数を使用（回転なし）
                try:
                    result_image, _, _ = make_enhanced_sketch_multicolor(
                        voxel_data,
                        output_size=(256, 256),
                        save_path=sketch_path,
                        verbose=True,
                        auto_angle=False,  # 固定角度を使用
                        fixed_angle=0.0
                    )
                    #print(f"スケッチ画像を保存（フォールバック）: {sketch_path}")
                except Exception as e2:
                    #print(f"スケッチ生成に失敗しました: {e2}")
                    continue
        
        
        
        #print(f"make_voxel_data: {output_index}")

        # 必要に応じてデータセットの保存（npz形式）
        if make_svs_dataset:
            # 単一のボクセルファイルとして保存（スケッチとペアになるように）
            os.makedirs(output_dir, exist_ok=True)

            # 元の save_npzForm 関数を使用
            voxel_file_path = save_npzForm(voxel_data, output_dir, output_index,file_prefix)

         
            # print(f"ボクセルファイルを保存: {voxel_file_path}")
            
            # XYZ形式での保存
            from edit_xyz import convert_npz_to_xyz
            output_xyz_dir = os.path.join(output_dir, "xyz")
            # print(f"voxel_file_path={voxel_file_path}, output_xyz_dir={output_xyz_dir}")
            convert_npz_to_xyz(
                voxel_file_path,
                output_xyz_dir,
            )
        # 可視化処理（Acaciaはvoxelデータ、その他はグラフを描画）
        if visualize_flag:
            # print(f"Visualizing file index: {output_index}")
            visualize_with_timeout4voxel(voxel_data,timeout=25)
            # plot_trunk_and_mainskelton_graph(graph)
            # plot_graph(graph)
    


def main() -> None:
    # 全体で利用するパラメータ設定
    total_files_per_category: int = 2  # 各木種類ごとのファイル数

    # 入力・出力のベースディレクトリ
    input_base_dir: str = "/mnt/nas/rmjapan2000/tree/l-strings"
    output_base_dir_svs: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/svs_LDM_v1"
    output_base_dir_svs_sketch: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/sketch_LDM_v1"

    # ディレクトリの作成
    os.makedirs(output_base_dir_svs, exist_ok=True)
    os.makedirs(output_base_dir_svs_sketch, exist_ok=True)
    
    voxel_dim = 256
    
    # 全木種類の処理（連続したファイル番号で管理）
    current_index = len(os.listdir(output_base_dir_svs_sketch))  # 現在のスケッチファイル数から開始インデックスを決定
    if current_index == 0:
        current_index = 1  # 最初のファイルは1から開始するため、0の場合は1に設定

    # 木の種類とそのパラメータの定義
    tree_categories = [
        {
            "tree_folder": "AcaciaClustered",
            "file_prefix": "Acacia",
            "trunk_thick": 3,
            "branch_thick": 2,
            "leaf_thick": 100
        },
        {
            "tree_folder": "BirchClustered", 
            "file_prefix": "Birch",
            "trunk_thick": 3,
            "branch_thick": 2,
            "leaf_thick": 1
        },
        {
            "tree_folder": "MapleClustered",
            "file_prefix": "Maple", 
            "trunk_thick": 1,
            "branch_thick": 1,
            "leaf_thick": 1
        },
        {
            "tree_folder": "OakClustered",
            "file_prefix": "Oak",
            "trunk_thick": 1,
            "branch_thick": 1,
            "leaf_thick": 1
        },
        {
            "tree_folder": "PineClustered",
            "file_prefix": "Pine",
            "trunk_thick": 1,
            "branch_thick": 1,
            "leaf_thick": 1
        }
    ]

    # 各木種類を処理
    for category in tree_categories:
        print(f"\n{'='*50}")
        print(f"Processing {category['tree_folder']} ({category['file_prefix']})")
        print(f"Starting from index: {current_index}") 
        print(f"{'='*50}")

        current_output_file_list = os.listdir(output_base_dir_svs_sketch)
        # 現在のTreeのファイル数をカウント
        current_processed_num = sum(1 for f in current_output_file_list if f.startswith(category["file_prefix"]) and f.endswith(".png"))
        print(f"現在の処理済みファイル数: {current_processed_num}個")
        process_tree_category(
            tree_folder=category["tree_folder"],
            file_prefix=category["file_prefix"],
            input_base_dir=input_base_dir,
            output_sketch_dir=output_base_dir_svs_sketch,
            output_dir=output_base_dir_svs,
            total_count=total_files_per_category,
            make_svs_dataset=True,
            make_sketch_dataset=True,
            visualize_flag=False,
            voxel_dims=(voxel_dim, voxel_dim, voxel_dim),
            convex_hull_flag=False,
            uwagaki_flag=False,
            trunk_thick=category["trunk_thick"],
            branch_thick=category["branch_thick"],
            leaf_thick=category["leaf_thick"],
        )
    
    print(f"\n{'='*50}")
    print(f"データセット作成完了!")
    print(f"総ファイル数: {current_index}")
    print(f"各種類: {total_files_per_category}個")
    print(f"総種類数: {len(tree_categories)}種類")
    print(f"ボクセルファイル: {output_base_dir_svs}")
    print(f"スケッチファイル: {output_base_dir_svs_sketch}")
    print(f"{'='*50}")
    
    # メタデータファイルの作成
    metadata_file = os.path.join(output_base_dir_svs, "dataset_metadata.txt")
    with open(metadata_file, "w") as f:
        f.write("Tree Dataset Metadata\n")
        f.write("=====================\n\n")
        f.write(f"Total files: {current_index}\n")
        f.write(f"Files per category: {total_files_per_category}\n")
        f.write(f"Total categories: {len(tree_categories)}\n\n")
        f.write("File naming convention:\n")
        f.write("- Voxel files: voxel_XXXXXX.npz\n")
        f.write("- Sketch files: sketch_XXXXXX.png\n")
        f.write("  (where XXXXXX is a 6-digit zero-padded index)\n\n")
        f.write("Category ranges:\n")
        
        current_start = 0
        for i, category in enumerate(tree_categories):
            end_index = current_start + total_files_per_category - 1
            f.write(f"{category['tree_folder']} ({category['file_prefix']}): {current_start:06d} - {end_index:06d}\n")
            current_start += total_files_per_category
    
    print(f"メタデータファイルを作成: {metadata_file}")

if __name__ == "__main__":
    # svs_path="/mnt/nas/rmjapan2000/tree/data_dir/train/svs_cgvi/svs_1499.npz"
    # svs_path="/mnt/nas/rmjapan2000/tree/data_dir/train/svd_0.2/svs_1.npz"
    # svs_data=np.load(svs_path)
    # from utils import npz2dense
    # svs_data=npz2dense(svs_data,256,256,256)
    # print(svs_data.shape)
    # # make_sketch_v2(svs_data)
    main()
