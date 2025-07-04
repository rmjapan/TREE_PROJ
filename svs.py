
import numpy as np  
import networkx as nx
import math
import os
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
    def __init__(self,pos,attr):
        self.pos=pos
        self.attr=attr
        self.strmatrix=np.identity(4)

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
    L=0
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
            L+=angle
    #度からラジアンに変換
    depth+=1
    depth_factor=np.random.randint(10, 31)

    depth=depth*depth_factor

    rot_x_rad=math.radians(rot_x)*depth
    rot_y_rad=math.radians(rot_y)*depth
    rot_z_rad=math.radians(rot_z)*depth
    # print(f"rot_x={rot_x},rot_y={rot_y},rot_z={rot_z},L={L}")
    

    # 回転行列定義
    R_x = np.array([
        [1, 0,              0,             0],
        [0, math.cos(rot_x_rad), -math.sin(rot_x_rad), 0],
        [0, math.sin(rot_x_rad),  math.cos(rot_x_rad), 0],
        [0, 0,              0,             1]
    ])

    R_y = np.array([
        [math.cos(rot_y_rad), 0, math.sin(rot_y_rad), 0],
        [0,                   1, 0,                  0],
        [-math.sin(rot_y_rad),0, math.cos(rot_y_rad),0],
        [0,                   0, 0,                  1]
    ])

    R_z = np.array([
        [math.cos(rot_z_rad), -math.sin(rot_z_rad), 0, 0],
        [math.sin(rot_z_rad),  math.cos(rot_z_rad), 0, 0],
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
def make_svs(l_list,depth,current_index,index,DG,stmatrix):
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
            
            #plot_graph(DG)
        elif l_list[index][0]=="[":
            # print("分岐開始")
            index=make_svs(l_list,depth+1,current_index,index+1,DG,stmatrix)
            # print(f"分岐終了")
  
        elif l_list[index][0]=="]":
            # print("葉に到達")（元のコード）
            # DG.nodes[current_index]["node"].attr=0
            # print(f"current_index={current_index}")
            # print(f"index={index}")
            # print(f"current_index+1={current_index+1}")
            # print(f"index+1={index+1}")
            
            
            #あたらしいコード
            # DG.add_node(index+1)
            # DG.add_edge(current_index,index+1)
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
                r
                leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*200
                leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*200
                leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*200
            else:
                leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*1.5
                leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*1.5
                leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*1.5
                
            # #Nodeの作成
            # new_node=Node(Pos(leaf_pos_x,leaf_pos_y,leaf_pos_z),0)
            # DG.nodes[index+1]["node"]=new_node
            # new_edge=make_edge(DG,current_index,index+1)
            # DG.edges[(current_index,index+1)]["edge"]=new_edge
            
            #新しいコードVer２
            #posからleaf_posの角度を０とする.
            # 前提：pos_x, pos_y, pos_z, leaf_pos_x, ... はすでに定義されている
            pos = np.stack([pos_x, pos_y, pos_z])
            leaf_pos = np.stack([leaf_pos_x, leaf_pos_y, leaf_pos_z])
            r = np.linalg.norm(pos - leaf_pos)+0.01
            zero_dir = (leaf_pos - pos) / r
            point_num = 4

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

# センチメートルからポイントに変換する関数
def cm_to_pt(linewidth_cm, dpi=100):
    return linewidth_cm * dpi / 2.54
def plot_trunk_and_mainskelton_graph(DG):
    # Extract node positions
    node_positions = {node: (DG.nodes[node]['node'].pos.x, 
                             DG.nodes[node]['node'].pos.y, 
                             DG.nodes[node]['node'].pos.z) 
                      for node in DG.nodes}
    attr_positions = {node: DG.nodes[node]['node'].attr
                      for node in DG.nodes}
    edge_thickness = {(u, v): DG[u][v]['edge'].thickness
                      for u, v in DG.edges()}
    # Create a 3D plot
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # for node, (x, y, z) in node_positions.items():  
    #     if  attr_positions[node]==1:#Trunk(幹＋深さ１にある枝)
    #         ax.scatter(x, y, z, c='brown', marker='o',s=10)
      
    
    # Plot the edges
    for (u, v) in DG.edges():
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        z = [node_positions[u][2], node_positions[v][2]]
        thickness = edge_thickness[(u, v)]
        thickness = cm_to_pt(thickness)
        if attr_positions[u] == 1 and attr_positions[v] == 1:
            ax.plot(x, y, z, c="#a65628", linewidth=thickness)
        # elif attr_positions[u] == 0.5 and attr_positions[v] == 0:
        #     ax.plot(x, y, z, c="green", linewidth=thickness)
        # else:
        #     ax.plot(x, y, z, c='#a65628', linewidth=thickness)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.view_init(elev=0, azim=90)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
def plot_graph(DG):
    # Extract node positions
    node_positions = {node: (DG.nodes[node]['node'].pos.x, 
                             DG.nodes[node]['node'].pos.y, 
                             DG.nodes[node]['node'].pos.z) 
                      for node in DG.nodes}
    attr_positions = {node: DG.nodes[node]['node'].attr
                      for node in DG.nodes}
    edge_thickness = {(u, v): DG[u][v]['edge'].thickness
                      for u, v in DG.edges()}
    # Create a 3D plot
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    for node, (x, y, z) in node_positions.items():  
        if attr_positions[node] ==0:
            #葉
            ax.scatter(x, y, z, c='green', marker='o',s=10)
        elif attr_positions[node]==1:#枝
            ax.scatter(x, y, z, c='brown', marker='o',s=10)
        else:
            ax.scatter(x, y, z, c='yellow', marker='o',s=10)
    
    # Plot the edges
    for (u, v) in DG.edges():
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        z = [node_positions[u][2], node_positions[v][2]]
        thickness = edge_thickness[(u, v)]
        thickness = cm_to_pt(thickness)
        if attr_positions[u] == 1 and attr_positions[v] == 1:
            ax.plot(x, y, z, c="#a65628", linewidth=thickness)
        elif attr_positions[u] == 0.5 and attr_positions[v] == 0:
            ax.plot(x, y, z, c="green", linewidth=thickness)
        else:
            ax.plot(x, y, z, c='#a65628', linewidth=thickness)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.view_init(elev=0, azim=90)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()


# 凸包と枝をMixして描画する関数
def mix_branches_and_leaves(plotter, DG, clusterd_leaf_points, num_cluster):
    # 凸包（葉）を描画
    colors = cm.rainbow(np.linspace(0, 1, num_cluster))
    for i in range(num_cluster):
        leaf_points = np.array(clusterd_leaf_points[i])
        leaf_hull = ConvexHull(leaf_points)
        leaf_mesh = []
        for simplex in leaf_hull.simplices:
            simplex = np.append(simplex, simplex[0])
            leaf_mesh.append(leaf_points[simplex])
        

        # PyVista用に四面体分割
        leaf_mesh=np.array(leaf_mesh)
        leaf_mesh=leaf_mesh.reshape(-1,3)
        
        
        leaf_mesh = pv.PolyData(leaf_mesh).delaunay_3d()

        # 葉をプロット
        plotter.add_mesh(leaf_mesh, color="green", opacity=1.0)

    # 枝を描画
    node_positions = {node: (DG.nodes[node]['node'].pos.x,
                             DG.nodes[node]['node'].pos.y,
                             DG.nodes[node]['node'].pos.z)
                      for node in DG.nodes}
    edge_thickness = {(u, v): DG[u][v]['edge'].thickness for u, v in DG.edges()}
    attr = {node: DG.nodes[node]['node'].attr for node in DG.nodes}

    for (u, v) in DG.edges():
        start = node_positions[u]
        end = node_positions[v]
        thickness = edge_thickness[(u, v)]
        
        line = pv.Line(start, end, resolution=10)
        if attr[u] == 1 and attr[v] == 1:
            plotter.add_mesh(line, color="#a65628", line_width=cm_to_pt(thickness))


#
def convex_hull_leaf_and_branch(DG):

    leaf_points=[]
    branch_points=[]
    leaf_and_brunch_points=[]
    for node in DG.nodes():
        if DG.nodes[node]['node'].attr==0:
            pos=DG.nodes[node]['node'].pos
            np_pos=np.array([pos.x,pos.y,pos.z])
            leaf_points.append(np_pos)
            
        elif DG.nodes[node]['node'].attr==1:#Trunk
            pos=DG.nodes[node]['node'].pos
            np_pos=np.array([pos.x,pos.y,pos.z])
            branch_points.append(np_pos)
    leaf_and_brunch_points=leaf_points+branch_points
    # Convert lists to NumPy arrays
    leaf_points = np.array(leaf_points)
    branch_points = np.array(branch_points)
    leaf_and_brunch_points = np.array(leaf_and_brunch_points)
    #凸包を求める3次元
    # Removed unused variables 'branch_hull' and 'leaf_and_branch_hull'
    
    leaf_hull = ConvexHull(leaf_points)
    branch_hull = ConvexHull(branch_points)
    
    leaf_and_branch_hull=ConvexHull(leaf_and_brunch_points)
    
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(111, projection='3d')
    #葉の凸包を描画
    #凸包を描画
    for simplex in leaf_hull.simplices:
        simplex = np.append(simplex, simplex[0])
        ax.plot(leaf_points[simplex, 0], leaf_points[simplex, 1], leaf_points[simplex, 2], 'g-')
    for simplex in branch_hull.simplices:
        simplex = np.append(simplex, simplex[0])
        #ax.plot(branch_points[simplex, 0], branch_points[simplex, 1], branch_points[simplex, 2], 'b-')
        
    for simplex in leaf_and_branch_hull.simplices:
        simplex = np.append(simplex, simplex[0])
        #ax.plot(leaf_and_brunch_points[simplex, 0], leaf_and_brunch_points[simplex, 1], leaf_and_brunch_points[simplex, 2], 'g-')
    for edge in DG.edges():
        #numpy配列に変換
        start_pos=np.array([DG.nodes[edge[0]]['node'].pos.x,DG.nodes[edge[0]]['node'].pos.y,DG.nodes[edge[0]]['node'].pos.z])
        end_pos=np.array([DG.nodes[edge[1]]['node'].pos.x,DG.nodes[edge[1]]['node'].pos.y,DG.nodes[edge[1]]['node'].pos.z])
        #幹の描画
        ax.plot([start_pos[0],end_pos[0]],[start_pos[1],end_pos[1]],[start_pos[2],end_pos[2]],c="black")
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
def convex_hull_pca_leaf(DG,pca_points,num_cluster,labels_3d):
    figure = plt.figure(figsize=(10, 10))
    ax=figure.add_subplot(111, projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, num_cluster))
    for i in range(num_cluster):
        leaf_points=pca_points[labels_3d==i]
        leaf_hull = ConvexHull(leaf_points)
        for simplex in leaf_hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(leaf_points[simplex, 0], leaf_points[simplex, 1], c=colors[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
def colors(n):
    ret = []
    r = int("0x100000", 16)
    g = int("0x001000", 16)
    b = int("0x000010", 16)
    for i in range(n):
        r = (r + 8372226) % 16777216
        g = (g + 8372226) % 16777216
        b = (b + 8372226) % 16777216
        ret.append("#" + hex(r)[2:] + hex(g)[2:] + hex(b)[2:])
    return ret
def branch_and_trunk_cluster(DG):
    branch_and_trunk_points=[]
    for node in DG.nodes():
        if DG.nodes[node]['node'].attr==1 or DG.nodes[node]['node'].attr==0.5:
            pos=DG.nodes[node]['node'].pos
            np_pos=np.array([pos.x,pos.y,pos.z])
            branch_and_trunk_points.append(np_pos)  
    branch_and_trunk_points=np.array(branch_and_trunk_points)
    n_clusters = 1
    # Step 1: Apply K-means clustering directly to the original 3D data
    kmeans_3d = KMeans(n_clusters=n_clusters)
    labels_3d = kmeans_3d.fit_predict(branch_and_trunk_points)
    pca_after_clustering = PCA(n_components=2)
    point_cloud_2d_after_clustering = pca_after_clustering.fit_transform(branch_and_trunk_points)
    
    # Step 3: Visualize the results
    fig = plt.figure(figsize=(12, 6))
    
    # 3D Plot of the original point cloud with clustering labels
    ax1 = fig.add_subplot(121, projection='3d')
    scatter3d_clustered = ax1.scatter(  branch_and_trunk_points[:, 0], branch_and_trunk_points[:, 1], branch_and_trunk_points[:, 2], c=labels_3d, cmap='viridis',s=20)
    ax1.set_title("Original 3D Point Cloud with Clustering")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    
    # PCA-reduced 2D plot after clustering
    ax2 = fig.add_subplot(122)
    scatter2d_clustered = ax2.scatter(point_cloud_2d_after_clustering[:, 0], point_cloud_2d_after_clustering[:, 1],
                                    c=labels_3d, cmap='viridis', s=20)
    print(f"labels_3d={labels_3d}")
    print(f"len(labels_3d)={len(labels_3d)}")
    ax2.set_title("PCA-Reduced 2D Representation After Clustering")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    plt.colorbar(scatter3d_clustered, ax=ax1, label='Cluster')
    plt.colorbar(scatter2d_clustered, ax=ax2, label='Cluster')
    plt.tight_layout()
    plt.show()


def leaf_cluster(DG):
    leaf_points=[]
    for node in DG.nodes():
        if DG.nodes[node]['node'].attr==0:
            pos=DG.nodes[node]['node'].pos
            np_pos=np.array([pos.x,pos.y,pos.z])
            leaf_points.append(np_pos)
    leaf_points=np.array(leaf_points)
    n_clusters =5
    # Step 1: Apply K-means clustering directly to the original 3D data
    kmeans_3d = KMeans(n_clusters=n_clusters)
    labels_3d = kmeans_3d.fit_predict(leaf_points)
    pca_after_clustering = PCA(n_components=2)
    point_cloud_2d_after_clustering = pca_after_clustering.fit_transform(leaf_points)
    
    # # Step 3: Visualize the results
    # fig = plt.figure(figsize=(12, 6))

    # # 3D Plot of the original point cloud with clustering labels
    # ax1 = fig.add_subplot(121, projection='3d')
    # scatter3d_clustered = ax1.scatter(  leaf_points[:, 0], leaf_points[:, 1], leaf_points[:, 2], c=labels_3d, cmap='viridis',s=20)
    # ax1.set_title("Original 3D Point Cloud with Clustering")
    # ax1.set_xlabel("X-axis")
    # ax1.set_ylabel("Y-axis")
    # ax1.set_zlabel("Z-axis")

    # # PCA-reduced 2D plot after clustering
    # ax2 = fig.add_subplot(122)
    # scatter2d_clustered = ax2.scatter(point_cloud_2d_after_clustering[:, 0], point_cloud_2d_after_clustering[:, 1],
    #                                 c=labels_3d, cmap='viridis', s=20)
    # print(f"labels_3d={labels_3d}")
    # print(f"len(labels_3d)={len(labels_3d)}")
    # ax2.set_title("PCA-Reduced 2D Representation After Clustering")
    # ax2.set_xlabel("Principal Component 1")
    # ax2.set_ylabel("Principal Component 2")

    # plt.colorbar(scatter3d_clustered, ax=ax1, label='Cluster')
    # plt.colorbar(scatter2d_clustered, ax=ax2, label='Cluster')

    # plt.tight_layout()
    # plt.show()
    
    #葉の分類  
    clustred_leaf_points=[[] for i in range(n_clusters)]
    for i in range(len(leaf_points)):
        clustred_leaf_points[labels_3d[i]].append(leaf_points[i])
    return clustred_leaf_points,point_cloud_2d_after_clustering,labels_3d

    


def plot_graph_and_strmatrix(DG):
    #nodeが持つ座標系（strmatrix）を使って、座標点＋座標軸を描画する
    # Extract node positions
    node_positions = {node: (DG.nodes[node]['node'].pos.x,
                                DG.nodes[node]['node'].pos.y,
                                DG.nodes[node]['node'].pos.z)
                        for node in DG.nodes}
    attr_positions = {node: DG.nodes[node]['node'].attr
                        for node in DG.nodes}
    edge_thickness = {(u, v): DG[u][v]['edge'].thickness
                        for u, v in DG.edges()}
    # Create a 3D plot
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the nodes
    for node, (x, y, z) in node_positions.items():
        if attr_positions[node] == 0:
            #葉
            ax.scatter(x, y, z, c='green', marker='o',s=10)
        else:
            ax.scatter(x, y, z, c='brown', marker='o',s=10)
            
        #座標軸を描画（Strmatrixを使って）
        #nodeの座標
        pos=np.array([x,y,z])
        #座標軸の長さ
     
        stmatrix=DG.nodes[node]["node"].strmatrix
        #x軸
        x_axis=stmatrix[0:3,0]
        y_axis=stmatrix[0:3,1]
        z_axis=stmatrix[0:3,2]
        x_axis_length=1
        y_axis_length=1
        z_axis_length=1
        x_axis_end=pos+x_axis_length*x_axis
        y_axis_end=pos+y_axis_length*y_axis
        z_axis_end=pos+z_axis_length*z_axis
        ax.plot([pos[0],x_axis_end[0]],[pos[1],x_axis_end[1]],[pos[2],x_axis_end[2]],c="red")
        ax.plot([pos[0],y_axis_end[0]],[pos[1],y_axis_end[1]],[pos[2],y_axis_end[2]],c="blue")
        ax.plot([pos[0],z_axis_end[0]],[pos[1],z_axis_end[1]],[pos[2],z_axis_end[2]],c="green")
        
    # Plot the edges
    for (u, v) in DG.edges():
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        z = [node_positions[u][2], node_positions[v][2]]
        thickness = edge_thickness[(u, v)]
        if attr_positions[u] == 1 and attr_positions[v] == 1:
            ax.plot(x, y, z, c="#a65628", linewidth=thickness)
        elif attr_positions[u] == 0.5 and attr_positions[v] == 0:
            ax.plot(x, y, z, c="green", linewidth=thickness)
        else:
            ax.plot(x, y, z, c='#a65628', linewidth=thickness)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.view_init(elev=0, azim=90)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

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
    
    
    
    
    
def make_svs(l_list,depth,current_index,index,DG,stmatrix):
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
            
            #plot_graph(DG)
        elif l_list[index][0]=="[":
            # print("分岐開始")
            index=make_svs(l_list,depth+1,current_index,index+1,DG,stmatrix)
            # print(f"分岐終了")
  
        elif l_list[index][0]=="]":
            # print("葉に到達")（元のコード）
            # DG.nodes[current_index]["node"].attr=0
            # print(f"current_index={current_index}")
            # print(f"index={index}")
            # print(f"current_index+1={current_index+1}")
            # print(f"index+1={index+1}")
            
            
            #あたらしいコード
            # DG.add_node(index+1)
            # DG.add_edge(current_index,index+1)
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
                leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*200
                leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*200
                leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*200
            else:
                leaf_pos_x=parent_pos_x+(pos_x-parent_pos_x)*1.5
                leaf_pos_y=parent_pos_y+(pos_y-parent_pos_y)*1.5
                leaf_pos_z=parent_pos_z+(pos_z-parent_pos_z)*1.5
                
            # #Nodeの作成
            # new_node=Node(Pos(leaf_pos_x,leaf_pos_y,leaf_pos_z),0)
            # DG.nodes[index+1]["node"]=new_node
            # new_edge=make_edge(DG,current_index,index+1)
            # DG.edges[(current_index,index+1)]["edge"]=new_edge
            
            #新しいコードVer２
            #posからleaf_posの角度を０とする.
            # 前提：pos_x, pos_y, pos_z, leaf_pos_x, ... はすでに定義されている
            pos = np.stack([pos_x, pos_y, pos_z])
            leaf_pos = np.stack([leaf_pos_x, leaf_pos_y, leaf_pos_z])
            r = np.linalg.norm(pos - leaf_pos)+0.01
            zero_dir = (leaf_pos - pos) / r
            point_num = 4

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
def make_sketch_canny_edge(
    voxel_data,
    output_size=(224, 224),
    upscale_size=(224, 224),
    save_path="sketch_xy_canny.png",
    verbose=True
):
    """
    Cannyエッジ検出を使用してスケッチ画像を生成する関数。

    Args:
        voxel_data: (H, W, D) ndarray, -1=空白, 0=葉, 0.5=枝, 1=幹
        output_size: tuple, 出力画像サイズ (width, height)
        upscale_size: tuple, アップスケールサイズ (width, height)
        save_path: str, 保存先ファイル名
        verbose: bool, ログ出力

    Returns:
        None
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    # --- 1. セマンティック濃淡マップ ---
    dense_voxel = np.copy(voxel_data)
    dense_voxel[dense_voxel == -1] = 0.0
    dense_voxel[dense_voxel == 0] = 0.2
    dense_voxel[dense_voxel == 0.5] = 0.5
    dense_voxel[dense_voxel == 1] = 1.0

    # --- 2. Y軸投影 & 画像化 ---
    projection_xy = np.flip(np.max(dense_voxel, axis=2), axis=0)
    img = np.clip(projection_xy * 255, 0, 255).astype(np.uint8)

    # --- 3. アップスケール ---
    img_upscaled = cv2.resize(img, upscale_size, interpolation=cv2.INTER_CUBIC)

    # --- 4. Cannyエッジ画像（白黒2値化） ---
    canny_image = cv2.Canny(img_upscaled, 100, 200)
    # 0/255の2値画像に変換
    binary_image = (canny_image > 0).astype(np.uint8) * 255

    if verbose:
        print(f"[INFO] Canny image shape: {canny_image.shape}")

    # 保存先ディレクトリの作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # PIL Imageとして保存（サイズを明示的に指定）
    binary_pil = Image.fromarray(binary_image)
    binary_pil = binary_pil.resize(output_size, Image.Resampling.LANCZOS)
    binary_pil.save(save_path, format='PNG')

    if verbose:
        print(f"[INFO] Binary (black and white) Canny edge sketch saved to {save_path}")
        print(f"[INFO] Saved image size: {binary_pil.size}")

import networkx as nx


def delete_short_branches(DG, min_length=4, root=None):
    if root is None:
        # 入次数が0のノードをrootとみなす
        roots = [n for n in DG.nodes if DG.in_degree(n)==0]
        if len(roots)==0:
            raise ValueError("No root found")
        root = roots[0]

    leaves = [n for n in DG.nodes if DG.out_degree(n) == 0]
    edges_to_remove = []

    for leaf in leaves:
        try:
            path = nx.shortest_path(DG, source=root, target=leaf)
        except nx.NetworkXNoPath:
            continue
        # 枝の長さがmin_length未満なら、その枝を削除
        if len(path) < min_length:
            # 葉に接続するエッジを消す（またはパス全体を消すことも可）
            edges_to_remove.append((path[-2], path[-1]))
    DG.remove_edges_from(edges_to_remove)
    return DG

def delete_branches(DG):
    #枝狩り処理
    leaves=[n for n in DG.nodes if DG.out_degree(n)==0]
    print(f"length of leaves:{len(leaves)}")
    #枝の数を半分にする
    edges_to_remove=[]
    for edge in DG.edges:
        if edge[1] in leaves:
            edges_to_remove.append(edge)
    DG.remove_edges_from(edges_to_remove)
    return DG



import networkx as nx
import random
def remove_subtree(G, node):
    # まず、このノードのすべての子ノード（successors）を取得
    children = list(G.successors(node))
    # 子ノードそれぞれについて再帰的にこの関数を呼ぶ
    for child in children:
        remove_subtree(G, child)
    # すべての子を消した後、自分自身も消す
    G.remove_node(node)
def dfs_prune(DG, node, max_branch=2, seed=42):
    random.seed(seed)
    children = list(DG.successors(node))
    # 分岐が多い場合だけ pruning
    node_depth=nx.shortest_path_length(DG,source=1,target=node)
    if node_depth>5:
        #0か１のどちらかをランダムに選択
        max_branch=np.random.randint(1,32)
        if max_branch!=1:
            max_branch=2
    if node_depth>10:
        max_branch=np.random.randint(1,16)
        if max_branch!=1:
            max_branch=2
    if node_depth>15:
        max_branch=np.random.randint(0,12)
        if max_branch!=0:
            max_branch=1
    if node_depth>20:
        max_branch=np.random.randint(0,8)
        if max_branch!=0:
            max_branch=1
    if node_depth>25:
        max_branch=np.random.randint(0,4)
        if max_branch!=0:
            max_branch=1
    if node_depth>30:
        max_branch=np.random.randint(0,2)
        if max_branch!=0:
            max_branch=1
    if len(children) > max_branch:
        keep = random.sample(children, max_branch)
        prune = set(children) - set(keep)
        for child in prune:
            remove_subtree(DG,child)
        children = keep  # 残した枝のみ探索

    for child in children:
        dfs_prune(DG, child, max_branch, seed)
def pre_process_make_sketch(projection, upscale_size=(224, 224)):
    import cv2
    projection_img = np.clip(projection * 255, 0, 255).astype(np.uint8)
    img_upscale = cv2.resize(projection_img, upscale_size, interpolation=cv2.INTER_CUBIC)
    return img_upscale
def prune_branches_dfs(DG, root=None, max_branch=2, seed=42):
    DG_pruned = DG.copy()#DGni
    if root is None:
        # 入次数0のノードをroot扱い
        roots = [n for n in DG_pruned.nodes if DG_pruned.in_degree(n)==0]
        if not roots:
            raise ValueError("No root node found.")
        root = roots[0]
    dfs_prune(DG_pruned, root, max_branch, seed)
    return DG_pruned


def make_sketch_v2(DG,output_dir,index):
    #枝狩り処理
    # G_pruned=delete_branches(DG)
    G_pruned=prune_branches_dfs(DG,max_branch=2)
    G_pruned=prune_branches_dfs(G_pruned,max_branch=2)
    voxel_data=create_voxel_data(G_pruned,256,256,256)
    # visualize_with_timeout4voxel(voxel_data)
    make_sketch_sobel(voxel_data,output_dir,index)




def make_sketch_sobel(svs_data,output_dir,index):
    import numpy as np
    import cv2
    import os
    from PIL import Image

    voxel_data = np.copy(svs_data)
    # 各方向の投影画像作成（元のコードと同様）
    vox_cam = voxel_data
    proj_front = np.max(vox_cam, axis=2)
    proj_front = np.flip(proj_front, axis=0)

    vox_cam = np.rot90(voxel_data, k=1, axes=(1,2))
    proj_back = np.max(vox_cam, axis=2)
    proj_back = np.flip(proj_back, axis=0)

    vox_cam = np.rot90(voxel_data, k=2, axes=(1,2))
    proj_left = np.max(vox_cam, axis=2)
    proj_left = np.flip(proj_left, axis=0)

    vox_cam = np.rot90(voxel_data, k=3, axes=(1,2))
    proj_right = np.max(vox_cam, axis=2)
    proj_right = np.flip(proj_right, axis=0)

    def process_sobel(img):
        # 前処理関数（2値化や正規化などを入れる場合はここで）
        img = np.clip(img*255,0,255).astype(np.uint8)
        # Sobelフィルター適用（x, y方向両方）
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        # ノイズ抑制のためのしきい値（二値化）
        _, binary = cv2.threshold(sobel, 40, 255, cv2.THRESH_BINARY)
        return binary

    # 各方向
    front_sobel = process_sobel(proj_front)
    back_sobel = process_sobel(proj_back)
    left_sobel = process_sobel(proj_left)
    right_sobel = process_sobel(proj_right)

    # output_dirをbase_dirとして、方向ごとにサブディレクトリを作成して保存
    directions = {
        "front": front_sobel,
        "back": back_sobel,
        "left": left_sobel,
        "right":right_sobel
    }

    for direction, sobel_img in directions.items():
        dir_path = os.path.join(output_dir, direction)
        os.makedirs(dir_path, exist_ok=True)
        Image.fromarray(sobel_img).save(os.path.join(dir_path, f"sketch_{direction}_sobel_{index}.png"))

    return front_sobel, back_sobel, left_sobel, right_sobel

def make_sketch(svs_data):
    import cv2
    voxel_data=np.copy(svs_data)
    # dense_voxel: (H, W, D)
    # 前（Z-方向）: そのまま
    vox_cam = voxel_data
    proj_front = np.max(vox_cam, axis=2)
    proj_front = np.flip(proj_front, axis=0)
    # 後（Z+方向）: z軸180度回転（z方向反転）
    vox_cam = np.rot90(voxel_data, k=1,axes=(1,2))
    proj_back = np.max(vox_cam, axis=2)
    proj_back = np.flip(proj_back, axis=0)
    # 左（X-方向）: y軸+90度回転（x→z、z→-x）
    vox_cam = np.rot90(voxel_data, k=2,axes=(1,2))
    proj_left = np.max(vox_cam, axis=2)
    proj_left = np.flip(proj_left, axis=0)
    # 右（X+方向）: y軸-90度回転（x→-z、z→x）
    vox_cam = np.rot90(voxel_data, k=3,axes=(1,2))
    proj_right = np.max(vox_cam, axis=2)
    proj_right = np.flip(proj_right, axis=0)
    
    
    # plt.imsave("proj_front.png", proj_front, cmap="gray")
    # plt.imsave("proj_back.png", proj_back, cmap="gray")
    # plt.imsave("proj_left.png", proj_left, cmap="gray")
    # plt.imsave("proj_right.png", proj_right, cmap="gray")

   
    front_img = pre_process_make_sketch(proj_front)
    canny_front = cv2.Canny(front_img, 100, 200)
    binary_front = (canny_front > 0).astype(np.uint8) * 255

    # 背面画像
    back_img = pre_process_make_sketch(proj_back)
    canny_back = cv2.Canny(back_img, 100, 200)
    binary_back = (canny_back > 0).astype(np.uint8) * 255

    # 左側画像
    left_img = pre_process_make_sketch(proj_left)
    canny_left = cv2.Canny(left_img, 100, 200)
    binary_left = (canny_left > 0).astype(np.uint8) * 255

    # 右側画像
    right_img = pre_process_make_sketch(proj_right)
    canny_right = cv2.Canny(right_img, 100, 200)
    binary_right = (canny_right > 0).astype(np.uint8) * 255

    # 保存処理
    import os
    from PIL import Image

    output_dir = "sketch_outputs"
    os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(binary_front).save(os.path.join(output_dir, "sketch_front.png"))
    Image.fromarray(binary_back).save(os.path.join(output_dir, "sketch_back.png"))
    Image.fromarray(binary_left).save(os.path.join(output_dir, "sketch_left.png"))
    Image.fromarray(binary_right).save(os.path.join(output_dir, "sketch_right.png"))

    

    


import os
from typing import List, Tuple
import numpy as np
import networkx as nx
import pyvista as pv



def initialize_tree() -> Tuple[nx.DiGraph, np.ndarray]:
    """
    ルートノードと変換行列の初期化を行う。
    
    Returns:
        Tuple[nx.DiGraph, np.ndarray]: 初期化済みのグラフと4x4の単位行列
    """
    pos = Pos(0, 0, 0)
    root = Node(pos, 1)
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
    uwagaki_flag: bool = False
) -> None:
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
    """
    # 上書きフラグにより、既存ファイルから開始番号を取得
    # 出力ディレクトリのパスを正しく連結
    front = "front"  # 必要に応じて適切なサフィックスやサブディレクトリ名に修正
    start_num: int = last_file_num(os.path.join(output_dir, front))
    print(f"start_num={start_num}")
    print(f"[{tree_folder}] start_num={start_num}")

    if start_num >= total_count and not uwagaki_flag:
        print(f"[{tree_folder}] すべてのファイルは処理済みです。")
        return



    for i in range(start_num, total_count + 1):
        # 入力ファイルパスの組み立て
        file_path = os.path.join(input_base_dir, tree_folder, tree_folder, f"{file_prefix}_{i%30000}.lstring")
        print(f"Processing file: {file_path}")
        lstring_lines = read_lstring_file(file_path)

        # グラフと変換行列の初期化
        graph, transform_matrix = initialize_tree()

        # Acaciaの場合は開始メッセージを表示
        if file_prefix == "Acacia":
            print("開始")

        # SVS生成とツリー構築
        make_svs(lstring_lines, 0, 1, 0, graph, transform_matrix)
        make_davinch_tree(graph, (1, 2))

        # 必要に応じてスケッチデータセットの生成
        # if make_sketch_dataset:
        #     clustered_leaf_points, pca_point, labels3d = leaf_cluster(graph)
        #     num_clusters = len(clustered_leaf_points)
        #     plotter = pv.Plotter()
        #     mix_branches_and_leaves(plotter, graph, clustered_leaf_points, num_clusters)
        #     plotter.show(screenshot="tree.png")

        # ボクセルデータの生成
        voxel_data = create_voxel_data(graph, *voxel_dims)

        if make_sketch_dataset:
        
            make_sketch_v2(graph,output_sketch_dir,i)
            # make_sketch_canny_edge(voxel_data,save_path=os.path.join(output_sketch_dir, f"sketch_canny_{i}.png"))
        import cupy as cp
        from scipy.spatial import ConvexHull
        # import cuspatial # type: ignore

        # # (1) 幹のボクセル（recon_xが1のもの）を取得
        # trunk_indices = cp.argwhere(cp.array(voxel_data) == 1)

        # # (2) Convex Hull の計算
        # if len(trunk_indices) >= 4 and convex_hull_flag:  # Convex Hull は4点以上必要
        #     print("Convex Hull の計算を実行")
        #     trunk_hull = ConvexHull(cp.asnumpy(trunk_indices))  # CPUで処理

        #     # (3) Delaunay 分割を実行（凸包の頂点ではなく、全トランクボクセルを使う）
        #     print("Delaunay 分割を実行")
        #     delaunay_trunk = cuspatial.spatial_delaunay(cp.array(trunk_indices))  # GPUでDelaunay分割

        #     # (4) Bounding Box で探索範囲を削減
        #     print("Bounding Box で探索範囲を削減")
        #     min_bounds = trunk_indices.min(axis=0)
        #     max_bounds = trunk_indices.max(axis=0)
        #     print(f"min_bounds={min_bounds}, max_bounds={max_bounds}")

        #     # (5) Bounding Box 内のボクセル座標を取得
        #     print("Bounding Box 内のボクセル座標を取得")
        #     x_range = cp.arange(min_bounds[0], max_bounds[0] + 1)
        #     y_range = cp.arange(min_bounds[1], max_bounds[1] + 1)
        #     z_range = cp.arange(min_bounds[2], max_bounds[2] + 1)
        #     xx, yy, zz = cp.meshgrid(x_range, y_range, z_range, indexing='ij')
        #     voxel_data_indices = cp.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T  # (N, 3)
        #     print(f"voxel_data_indices.shape={voxel_data_indices.shape}")

        #     # (6) Convex Hull 内にあるか判定 (Bounding Box 内のみ)
        #     print("Convex Hull 内にあるか判定")
        #     trunk_insidemask = cuspatial.spatial_delaunay_find_simplex(delaunay_trunk, voxel_data_indices) >= 0

        #     # (7) Convex Hull 内のボクセルを 1 に設定
        #     print("Convex Hull 内のボクセルを 1 に設定")
        #     voxel_data_gpu = cp.array(voxel_data)
        #     voxel_data_gpu[voxel_data_indices[trunk_insidemask][:, 0], 
        #                 voxel_data_indices[trunk_insidemask][:, 1], 
        #                 voxel_data_indices[trunk_insidemask][:, 2]] = 1

        #     # 結果をCPUに戻す（必要なら）
        #     voxel_data = cp.asnumpy(voxel_data_gpu)
                
                
        
        
        print(f"make_voxel_data: {i}")

        # 必要に応じてデータセットの保存（npz形式）
        if make_svs_dataset:
            # 4方向分保存するために、output_dirをBase_dirにして、4方向分のディレクトリを作成して保存
            directions = ["front", "back", "left", "right"]
            for direction in directions:
                dir_path = os.path.join(output_dir, direction)
                os.makedirs(dir_path, exist_ok=True)
                save_npzForm(voxel_data, dir_path, i)

        # 可視化処理（Acaciaはvoxelデータ、その他はグラフを描画）
        if visualize_flag:
            print(f"Visualizing file index: {i}")
            visualize_with_timeout4voxel(voxel_data)
            # plot_trunk_and_mainskelton_graph(graph)
            # plot_graph(graph)

def main() -> None:
    # 全体で利用するパラメータ設定
    total_files: int = 150000

    # 入力・出力のベースディレクトリ
    input_base_dir: str = "/mnt/nas/rmjapan2000/tree/l-strings"
    output_base_dir_svd: str = "/home/ryuichi/tree/TREE_PROJ/data_dir/train/svd_0.2"
    output_base_dir_svs: str = "/home/ryuichi/tree/TREE_PROJ/data_dir/train/svs_0.2"
    # output_base_dir_svs_cgvi: str = "/home/ryuichi/tree/TREE_PROJ/data_dir/train/svs_cgvi"
    # output_base_dir_sketch: str = "/home/ryuichi/tree/TREE_PROJ/data_dir/train/sketch_cgvi"
    output_base_dir_sketch: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/sketch_cgvi"
    output_base_dir_svs_cgvi: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/svs_cgvi"


    output_base_dir_svs_cgvi_256: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/svs_cgvi_ver2"
    output_base_dir_svs_cgvi_256_sketch: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/sketch_cgvi_ver2"

    output_base_dir_svs_cgvi=output_base_dir_svs_cgvi_256
    output_base_dir_sketch=output_base_dir_svs_cgvi_256_sketch

    # os.makedirs(output_base_dir_sketch)
    # os.makedirs(output_base_dir_svs_cgvi)
    voxel_dim=256

    # Acaciaの処理（可視化あり）
    process_tree_category(
        tree_folder="AcaciaClustered",
        file_prefix="Acacia",
        input_base_dir=input_base_dir,
        output_sketch_dir=output_base_dir_sketch,
        output_dir=output_base_dir_svs_cgvi,
        total_count=total_files,
        make_svs_dataset=True,
        make_sketch_dataset=True,
        visualize_flag=False,
        voxel_dims=(voxel_dim, voxel_dim, voxel_dim),
        convex_hull_flag=False,
        uwagaki_flag=False
    )

    # Birchの処理（可視化なし）
    process_tree_category(
        tree_folder="BirchClustered",
        file_prefix="Birch",
        input_base_dir=input_base_dir,
        output_sketch_dir=output_base_dir_sketch,
        output_dir=output_base_dir_svs_cgvi,
        total_count=total_files,
        make_svs_dataset=True,
        make_sketch_dataset=True,
        visualize_flag=False,
        voxel_dims=(voxel_dim, voxel_dim, voxel_dim),
        convex_hull_flag=False,
        uwagaki_flag=False
    )

    # Mapleの処理（可視化なし）
    process_tree_category(
        tree_folder="MapleClustered",
        file_prefix="Maple",
        input_base_dir=input_base_dir,
        output_sketch_dir=output_base_dir_sketch,
        output_dir=output_base_dir_svs_cgvi,
        total_count=total_files,
        make_svs_dataset=True,
        make_sketch_dataset=True,
        voxel_dims=(voxel_dim, voxel_dim, voxel_dim),
        visualize_flag=False,
        uwagaki_flag=False
    )

    # # Oakの処理（可視化なし）
    process_tree_category(
        tree_folder="OakClustered",
        file_prefix="Oak",
        input_base_dir=input_base_dir,
        output_sketch_dir=output_base_dir_sketch,
        output_dir=output_base_dir_svs_cgvi,
        total_count=total_files,
        voxel_dims=(voxel_dim, voxel_dim, voxel_dim),
        make_svs_dataset=True,
        make_sketch_dataset=True,
        visualize_flag=False,
        uwagaki_flag=False
    )

    # Pineの処理（出力ディレクトリがsvs_0.2、可視化なし）
    process_tree_category(
        tree_folder="PineClustered",
        file_prefix="Pine",
        input_base_dir=input_base_dir,
        output_sketch_dir=output_base_dir_sketch,
        output_dir=output_base_dir_svs_cgvi,
        total_count=total_files,
        make_svs_dataset=True,
        voxel_dims=(voxel_dim, voxel_dim, voxel_dim),
        make_sketch_dataset=True,
        visualize_flag=False,
        uwagaki_flag=False
    )

if __name__ == "__main__":
    # svs_path="/mnt/nas/rmjapan2000/tree/data_dir/train/svs_cgvi/svs_1499.npz"
    # svs_path="/mnt/nas/rmjapan2000/tree/data_dir/train/svd_0.2/svs_1.npz"
    # svs_data=np.load(svs_path)
    # from utils import npz2dense
    # svs_data=npz2dense(svs_data,256,256,256)
    # print(svs_data.shape)
    # # make_sketch_v2(svs_data)
    main()
