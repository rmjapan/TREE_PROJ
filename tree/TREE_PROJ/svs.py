from tkinter import S
from tracemalloc import start
from turtle import st
from xml.dom.minidom import Element
import numpy as np  
import networkx as nx
import math
import os
import re
from regex import F
from torch import fill

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
from visualize_func import visualize_voxel_data


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
    depth=depth*20
    if depth>30:
        
        depth=30
    
    
    
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
            # print("葉に到達")
            DG.nodes[current_index]["node"].attr=0
            return index
        else:
            # print(f"コマンドをスタックに追加:{l_list[index]}")
            stack.append(l_list[index])
        index+=1
    return index



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

def process_tree_category(
    tree_folder: str,
    file_prefix: str,
    input_base_dir: str,
    output_dir: str,
    total_count: int,
    make_svs_dataset: bool,
    make_sketch_dataset: bool,
    visualize_flag: bool,
    voxel_dims: Tuple[int, int, int] = (256, 256, 256)
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
    start_num: int = last_file_num(output_dir)
    print(f"[{tree_folder}] start_num={start_num}")

    if start_num >= total_count:
        print(f"[{tree_folder}] すべてのファイルは処理済みです。")
        return

    for i in range(start_num, total_count + 1):
        # 入力ファイルパスの組み立て
        file_path = os.path.join(input_base_dir, tree_folder, tree_folder, f"{file_prefix}_{i}.lstring")
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
        if make_sketch_dataset:
            clustered_leaf_points, pca_point, labels3d = leaf_cluster(graph)
            num_clusters = len(clustered_leaf_points)
            plotter = pv.Plotter()
            mix_branches_and_leaves(plotter, graph, clustered_leaf_points, num_clusters)
            plotter.show(screenshot="tree.png")

        # ボクセルデータの生成
        voxel_data = create_voxel_data(graph, *voxel_dims)
        import cupy as cp
        from scipy.spatial import ConvexHull
        import cuspatial

        # (1) 幹のボクセル（recon_xが1のもの）を取得
        trunk_indices = cp.argwhere(cp.array(voxel_data) == 1)

        # (2) Convex Hull の計算
        if len(trunk_indices) >= 4:  # Convex Hull は4点以上必要
            print("Convex Hull の計算を実行")
            trunk_hull = ConvexHull(cp.asnumpy(trunk_indices))  # CPUで処理

            # (3) Delaunay 分割を実行（凸包の頂点ではなく、全トランクボクセルを使う）
            print("Delaunay 分割を実行")
            delaunay_trunk = cuspatial.spatial_delaunay(cp.array(trunk_indices))  # GPUでDelaunay分割

            # (4) Bounding Box で探索範囲を削減
            print("Bounding Box で探索範囲を削減")
            min_bounds = trunk_indices.min(axis=0)
            max_bounds = trunk_indices.max(axis=0)
            print(f"min_bounds={min_bounds}, max_bounds={max_bounds}")

            # (5) Bounding Box 内のボクセル座標を取得
            print("Bounding Box 内のボクセル座標を取得")
            x_range = cp.arange(min_bounds[0], max_bounds[0] + 1)
            y_range = cp.arange(min_bounds[1], max_bounds[1] + 1)
            z_range = cp.arange(min_bounds[2], max_bounds[2] + 1)
            xx, yy, zz = cp.meshgrid(x_range, y_range, z_range, indexing='ij')
            voxel_data_indices = cp.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T  # (N, 3)
            print(f"voxel_data_indices.shape={voxel_data_indices.shape}")

            # (6) Convex Hull 内にあるか判定 (Bounding Box 内のみ)
            print("Convex Hull 内にあるか判定")
            trunk_insidemask = cuspatial.spatial_delaunay_find_simplex(delaunay_trunk, voxel_data_indices) >= 0

            # (7) Convex Hull 内のボクセルを 1 に設定
            print("Convex Hull 内のボクセルを 1 に設定")
            voxel_data_gpu = cp.array(voxel_data)
            voxel_data_gpu[voxel_data_indices[trunk_insidemask][:, 0], 
                        voxel_data_indices[trunk_insidemask][:, 1], 
                        voxel_data_indices[trunk_insidemask][:, 2]] = 1

            # 結果をCPUに戻す（必要なら）
            voxel_data = cp.asnumpy(voxel_data_gpu)
                
                
        
        
        print(f"make_voxel_data: {i}")

        # 必要に応じてデータセットの保存（npz形式）
        if make_svs_dataset:
            save_npzForm(voxel_data, output_dir, i)

        # 可視化処理（Acaciaはvoxelデータ、その他はグラフを描画）
        if visualize_flag:
            if file_prefix == "Acacia":
                print(f"Visualizing file index: {i}")
                visualize_voxel_data(voxel_data)
            else:
                plot_trunk_and_mainskelton_graph(graph)
                plot_graph(graph)

def main() -> None:
    # 全体で利用するパラメータ設定
    total_files: int = 30000

    # 入力・出力のベースディレクトリ
    input_base_dir: str = "/mnt/nas/rmjapan2000/tree/l-strings"
    output_base_dir_svd: str = "/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2"
    output_base_dir_svs: str = "/home/ryuichi/tree/TREE_PROJ/data_dir/svs_0.2"

    # Acaciaの処理（可視化あり）
    process_tree_category(
        tree_folder="AcaciaClustered",
        file_prefix="Acacia",
        input_base_dir=input_base_dir,
        output_dir=os.path.join(output_base_dir_svd, "AcaciaClustered"),
        total_count=total_files,
        make_svs_dataset=False,
        make_sketch_dataset=False,
        visualize_flag=True,
    )

    # Birchの処理（可視化なし）
    process_tree_category(
        tree_folder="BirchClustered",
        file_prefix="Birch",
        input_base_dir=input_base_dir,
        output_dir=os.path.join(output_base_dir_svd, "BirchClustered"),
        total_count=total_files,
        make_svs_dataset=False,
        make_sketch_dataset=False,
        visualize_flag=False,
    )

    # Mapleの処理（可視化なし）
    process_tree_category(
        tree_folder="MapleClustered",
        file_prefix="Maple",
        input_base_dir=input_base_dir,
        output_dir=os.path.join(output_base_dir_svd, "MapleClustered"),
        total_count=total_files,
        make_svs_dataset=False,
        make_sketch_dataset=False,
        visualize_flag=False,
    )

    # Oakの処理（可視化なし）
    process_tree_category(
        tree_folder="OakClustered",
        file_prefix="Oak",
        input_base_dir=input_base_dir,
        output_dir=os.path.join(output_base_dir_svd, "OakClustered"),
        total_count=total_files,
        make_svs_dataset=False,
        make_sketch_dataset=False,
        visualize_flag=False,
    )

    # Pineの処理（出力ディレクトリがsvs_0.2、可視化なし）
    process_tree_category(
        tree_folder="PineClustered",
        file_prefix="Pine",
        input_base_dir=input_base_dir,
        output_dir=os.path.join(output_base_dir_svs, "PineClustered"),
        total_count=total_files,
        make_svs_dataset=False,
        make_sketch_dataset=False,
        visualize_flag=False,
    )

if __name__ == "__main__":
    main()

# def main():
#     # 全体で利用するパラメータ設定
#     N = 30000  # 全データセット数
#     make_svs_dataset_flag = False  # svsデータセットを作成するかどうか
#     make_sketch_dataset_flag = False  # sketchデータセットを作成するかどうか
#     flag_npz = True  # npz形式で保存するかどうか
#     visualize_flag = True  # 可視化するかどうか
#     uwagaki_flag = True  # 上書きフラグ
#     start_num = 1
    
    
#     # 上書きフラグが有効なら、処理を行ってstart_numを更新
#     if uwagaki_flag:
#         dir_path = fR"/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2/AcaciaClustered"
#         num = last_file_num(dir_path)
#         print(f"num={num}")
#         uwagaki_flag = False
#         print(f"uwagaki_flag={uwagaki_flag}")
#         start_num = num
#     print(f"start_num={start_num}") 
#     if start_num < N:
#     # ループ: AcaciaのデータをN回処理
#         for i in range(start_num, N + 1):
        
#             dir_path = fR"/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2/AcaciaClustered"
#             make_svs_dataset_flag = False
#             make_sketch_dataset_flag = False
#             flag_npz = True
#             visualize_flag = True
            
#             # lstringファイル読み込み
#             path = fR"/home/ryuichi/tree/l-strings/AcaciaClustered/AcaciaClustered/Acacia_{i}.lstring"
#             file = open(path, "r")
#             l_list = file.readlines()
            
#             # 根ノードとグラフの初期化
#             pos = Pos(0, 0, 0)
#             root = Node(pos, 1)
#             DG = nx.DiGraph()
#             DG.add_node(1)
#             DG.nodes[1]["node"] = root
#             strmatrix = np.identity(4)
            
#             # svs生成処理
#             print("開始")
#             make_svs(l_list, 0, 1, 0, DG, strmatrix)
#             make_davinch_tree(DG, (1, 2))
            
#             # 必要に応じてスケッチデータセット用処理
#             if make_sketch_dataset_flag:
#                 clusterd_leaf_points, pca_point, labels3d = leaf_cluster(DG)
#                 num_cluster = len(clusterd_leaf_points)
#                 plotter = pv.Plotter()
#                 mix_branches_and_leaves(plotter, DG, clusterd_leaf_points, num_cluster)
#                 plotter.show(screenshot="tree.png")
            
#             # ボクセル生成
#             voxel_data = create_voxel_data(DG, 256, 256, 256)
#             print(f"make_voxel_data:{i}")
            
#             # データセット保存
#             if make_svs_dataset_flag:
#                 save_npzForm(voxel_data, dir_path, i)
            
#             # 可視化
#             if visualize_flag and i % 1 == 0:
#                 print(f"i={i}")
#                 visualize_voxel_data(voxel_data)
    
#     # 上書きフラグが有効ならパスやstart_numを更新
#     uwagaki_flag = True
#     if uwagaki_flag:
#         dir_path = fR"/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2/BirchClustered"
#         num = last_file_num(dir_path)
#         print(f"num={num}")
#         uwagaki_flag = False
#         start_num = num
#     print("BirchClustered")
#     print(f"start_num={start_num}")

#     if start_num < N:   
#         # ループ: BirchのデータをN回処理
#         for i in range(start_num, N + 1):
            
#             make_svs_dataset_flag = False
#             make_sketch_dataset_flag = False
#             flag_npz = True
#             visualize_flag = False
            
#             path = fR"/home/ryuichi/tree/l-strings/BirchClustered/BirchClustered/Birch_{i}.lstring"
#             print(path)
#             file = open(path, "r")
#             l_list = file.readlines()
            
#             pos = Pos(0, 0, 0)
#             root = Node(pos, 1)
#             DG = nx.DiGraph()
#             DG.add_node(1)
#             DG.nodes[1]["node"] = root
#             strmatrix = np.identity(4)
            
#             # svs生成
#             make_svs(l_list, 0, 1, 0, DG, strmatrix)
#             make_davinch_tree(DG, (1, 2))
            
#             # スケッチデータセット用処理（必要なら）
#             if make_sketch_dataset_flag:
#                 clusterd_leaf_points, pca_point, labels3d = leaf_cluster(DG)
#                 num_cluster = len(clusterd_leaf_points)
#                 plotter = pv.Plotter()
#                 mix_branches_and_leaves(plotter, DG, clusterd_leaf_points, num_cluster)
#                 plotter.show(screenshot="tree.png")
            
#             voxel_data = create_voxel_data(DG,256,256,256)
            
#             if make_svs_dataset_flag:
#                 save_npzForm(voxel_data, dir_path, i)
            
#             if visualize_flag:
#                 plot_trunk_and_mainskelton_graph(DG)
#                 plot_graph(DG)
        
#     uwagaki_flag = True
#     if uwagaki_flag:
#             dir_path = fR"/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2/MapleClustered"
#             num = last_file_num(dir_path)
#             print(f"num={num}")
#             uwagaki_flag = False
#             start_num = num
#     print("MapleClustered")
#     print(f"start_num={start_num}")
        
#     if start_num < N:

#             # ループ: MapleのデータをN回処理
#             for i in range(start_num, N + 1):
#                 make_svs_dataset_flag = False
#                 make_sketch_dataset_flag = False
#                 flag_npz = True
#                 visualize_flag = False
                
#                 path = fR"/home/ryuichi/tree/l-strings/MapleClustered/MapleClustered/Maple_{i}.lstring"
#                 file = open(path, "r")
#                 l_list = file.readlines()
                
#                 pos = Pos(0, 0, 0)
#                 root = Node(pos, 1)
#                 DG = nx.DiGraph()
#                 DG.add_node(1)
#                 DG.nodes[1]["node"] = root
                
#                 strmatrix = np.identity(4)
#                 make_svs(l_list, 0, 1, 0, DG, strmatrix)
#                 make_davinch_tree(DG, (1, 2))
                
#                 if make_sketch_dataset_flag:
#                     clusterd_leaf_points, pca_point, labels3d = leaf_cluster(DG)
#                     num_cluster = len(clusterd_leaf_points)
#                     plotter = pv.Plotter()
#                     mix_branches_and_leaves(plotter, DG, clusterd_leaf_points, num_cluster)
#                     plotter.show(screenshot="tree.png")

#                 voxel_data = create_voxel_data(DG, 256, 256, 256)
                
#                 if make_svs_dataset_flag:
#                     save_npzForm(voxel_data, dir_path, i)

#                 if visualize_flag and i % 1 == 0:
#                     plot_trunk_and_mainskelton_graph(DG)
#                     plot_graph(DG)
#     uwagaki_flag = True
#     if uwagaki_flag:
#             dir_path = fR"/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2/OakClustered"
#             num = last_file_num(dir_path)
#             print(f"num={num}")
#             uwagaki_flag = False
#             start_num = num
#     print("OakClustered")
#     print(f"start_num={start_num}")
#     if start_num < N:
#         # ループ: OakのデータをN回処理
#             for i in range(start_num, N + 1):
#                 make_svs_dataset_flag = False
#                 make_sketch_dataset_flag = False
#                 flag_npz = True
#                 visualize_flag = False
                
#                 path = fR"/home/ryuichi/tree/l-strings/OakClustered/OakClustered/Oak_{i}.lstring"
#                 file = open(path, "r")
#                 l_list = file.readlines()
                
#                 pos = Pos(0, 0, 0)
#                 root = Node(pos, 1)
#                 DG = nx.DiGraph()
#                 DG.add_node(1)
#                 DG.nodes[1]["node"] = root
                
#                 strmatrix = np.identity(4)
#                 make_svs(l_list, 0, 1, 0, DG, strmatrix)
#                 make_davinch_tree(DG, (1, 2))
                
#                 if make_sketch_dataset_flag:
#                     clusterd_leaf_points, pca_point, labels3d = leaf_cluster(DG)
#                     num_cluster = len(clusterd_leaf_points)
#                     plotter = pv.Plotter()
#                     mix_branches_and_leaves(plotter, DG, clusterd_leaf_points, num_cluster)
#                     plotter.show(screenshot="tree.png")

#                 voxel_data = create_voxel_data(DG, 256, 256, 256)
                
#                 if make_svs_dataset_flag:
#                     save_npzForm(voxel_data, dir_path, i)

#                 if visualize_flag and i % 1 == 0:
#                     plot_trunk_and_mainskelton_graph(DG)
#                     plot_graph(DG)
#     uwagaki_flag = True
#     if uwagaki_flag:
#             dir_path = fR"/home/ryuichi/tree/TREE_PROJ/data_dir/svs_0.2/PineClustered"
#             num = last_file_num(dir_path)
#             print(f"num={num}")
#             uwagaki_flag = False
#             start_num = num
#     print("PineClustered")
#     print(f"start_num={start_num}")
#     if start_num <N:
#             # ループ: PineのデータをN回処理
#             for i in range(start_num, N + 1):
#                 make_svs_dataset_flag = False
#                 make_sketch_dataset_flag = False
#                 flag_npz = True
#                 visualize_flag = False
                
#                 path = fR"/home/ryuichi/tree/l-strings/PineClustered/PineClustered/Pine_{i}.lstring"
#                 file = open(path, "r")
#                 l_list = file.readlines()
                
#                 pos = Pos(0, 0, 0)
#                 root = Node(pos, 1)
#                 DG = nx.DiGraph()
#                 DG.add_node(1)
#                 DG.nodes[1]["node"] = root
                
#                 strmatrix = np.identity(4)
#                 make_svs(l_list, 0, 1, 0, DG, strmatrix)
#                 make_davinch_tree(DG, (1, 2))
                
#                 if make_sketch_dataset_flag:
#                     clusterd_leaf_points, pca_point, labels3d = leaf_cluster(DG)
#                     num_cluster = len(clusterd_leaf_points)
#                     plotter = pv.Plotter()
#                     mix_branches_and_leaves(plotter, DG, clusterd_leaf_points, num_cluster)
#                     plotter.show(screenshot="tree.png")

#                 voxel_data = create_voxel_data(DG, 256, 256, 256)
                
#                 if make_svs_dataset_flag:
#                     save_npzForm(voxel_data, dir_path, i)

#                 if visualize_flag and i % 1 == 0:
#                     plot_trunk_and_mainskelton_graph(DG)
#                     plot_graph(DG)


if __name__ == "__main__":
    main()
