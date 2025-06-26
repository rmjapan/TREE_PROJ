import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch
import time
import json
import os

time_list=[]
time_dict={
    "頂点の追加":0,
    "辺の追加１":0,
    "辺の追加２":0,
    "辺の追加３":0,
}






OCTREE_CHILD_NEIGHBOR_PAIRS = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

LOOKUP_TABLE_DIRECTION = np.array([
    [-1,5,0,-1,2,-1,-1,-1],#0:1,2,4
    [4,-1,-1,0,-1,2,-1,-1],#1:0,3,5
    [1,-1,-1,5,-1,-1,2,-1],#2:0,3,6
    [-1,1,4,-1,-1,-1,-1,2],#3:1,2,7
    [3,-1,-1,-1,-1,5,0,-1],#4:0,5,6
    [-1,3,-1,-1,4,-1,-1,0],#5:1,4,7
    [-1,-1,3,-1,1,-1,-1,5],#6:2,4,7
    [-1,-1,-1,3,-1,1,4,-1]#7:3,5,6
])

DIRECTION_TO_CHILD_INDICES = {
    0: [0,1,4,5],  # 前
    1: [2,3,6,7],  # 後
    2: [0,1,2,3],  # 左
    3: [4,5,6,7],  # 右
    4: [1,3,5,7],  # 上
    5: [0,2,4,6],  # 下
}
CONNECT_PATTERNS = {
        0: [(2, 0), (3, 1), (6, 4), (7, 5)],  # 後ろ→前
        1: [(0, 2), (1, 3), (4, 6), (5, 7)],  # 前→後
        2: [(4, 0), (5, 1), (6, 2), (7, 3)],  # 左→右
        3: [(0, 4), (1, 5), (2, 6), (3, 7)],  # 右→左
        4: [(0, 1), (2, 3), (4, 5), (6, 7)],  # 下→上
        5: [(1, 0), (3, 2), (5, 4), (7, 6)],  # 上→下
    }

AVAILLABLE_CONNECT_TABLE={
    0:[1,2,4],
    1:[0,3,5],
    2:[0,3,6],
    3:[1,2,7],
    4:[0,5,6],
    5:[1,4,7],
    6:[2,4,7],
    7:[3,5,6],
}
def get_direction_code(dx, dy, dz):
    if dx > 0 and dy == 0 and dz == 0:
        return 2  # right
    elif dx < 0 and dy == 0 and dz == 0:
        return 3  # left
    elif dy > 0 and dx == 0 and dz == 0:
        return 0  # back
    elif dy < 0 and dx == 0 and dz == 0:
        return 1  # front
    elif dz > 0 and dx == 0 and dy == 0:
        return 5  # up
    elif dz < 0 and dx == 0 and dy == 0:
        return 4  # down
    else:
        return -1  # unknown
def add_child_node(G,node):
    for child in node.children:
        child.parent=node
        G.add_node(child)
def connect_same_parent_child_node(G,same_parent_child_node_list):
    #OCTREE_CHILD_NEIGHBOR_PAIRsで定義されたペア同士をEdgeとして追加
    for edge_pair in OCTREE_CHILD_NEIGHBOR_PAIRS:
        G.add_edge(
            same_parent_child_node_list[edge_pair[0]],
            same_parent_child_node_list[edge_pair[1]]
            )
def add_same_parent_child_edge(G,parent_node_list):
    for parent_node in parent_node_list:
        #グラフGに含まれるノードの内、同じ親ノードを共有するノード集合を取得する.
        same_parent_child_node_list=parent_node.children
        #同じ親ノードを共有するノード同士の接続
        connect_same_parent_child_node(G,same_parent_child_node_list)
def search_connected_node_list(old_G,leaf_node):
    #leaf_nodeの元の接続辺の子ノードを探す.
    connected_node_list=list(old_G.neighbors(leaf_node))
    return connected_node_list
def connect_leaf_node2child_node_by_lookup_table(new_G,G, leaf_node, connect_node):
    #Leaf_nodeとconnect_nodeが異なる深さの場合
    if abs(connect_node.depth-leaf_node.depth)>0:
        brother_of_connect_node_list=connect_node.parent.children
        for brother_node in brother_of_connect_node_list:
            if G.has_edge(leaf_node,brother_node):
                new_G.add_edge(leaf_node,connect_node.children[brother_node.id])
    #leaf_nodeとconnect_nodeが同じ深さの場合
    else:
        connect_node_id=connect_node.id    
        leaf_node_id=leaf_node.id
        #兄弟ノードでない場合は、タプル代入による交換を行う.（方向が逆向きになるから)
        if leaf_node.parent != connect_node.parent:
            leaf_node_id,connect_node_id=connect_node_id,leaf_node_id
        #葉ノードとconnect_nodeの接続方向を求める.
        connect_direction=LOOKUP_TABLE_DIRECTION[leaf_node_id][connect_node_id]
        if connect_direction!=-1:
                connect_child_id=DIRECTION_TO_CHILD_INDICES[connect_direction]
                for idx in connect_child_id:
                    new_G.add_edge(leaf_node,connect_node.children[idx])
def is_node_in_table(node_id,table):
    if node_id in table:
        return True
    else:
        return False
def search_availlable_parent_node(parent_node,parent_node_list):
    # まず重要なのは親の親までが一致しているかである。
    # parent_nodeのid_listのうち、親の親まで（深さ-2まで）を比較する
    # parent_node_listの中から、親の親までのid_listが一致するノードを抽出する
    #繋がり方が重要.親の位置
    parent_id_list = parent_node.id_list[:-1]  # 親の親まで共有面をちゃんと見てあげる必要がある.
    matched_parent_nodes = []
    for node in parent_node_list:
        if node is parent_node:
            continue
        flag=True
        for i in range(len(parent_id_list)):
            parent_node_id=parent_id_list[i]
            node_id=node.id_list[i]
            #一致していない場合は、lookup_tableを使って接続可能か確認する.
            if parent_node_id!=node_id:
                #table=lookup_table(parent_node_id)
                table=AVAILLABLE_CONNECT_TABLE[parent_node_id]
                if not is_node_in_table(node_id,table):
                    flag=False
                    break
        if flag:
            matched_parent_nodes.append(node)

                    

    availlable_table_list=AVAILLABLE_CONNECT_TABLE[parent_node.id]
    other_parent_node_list=matched_parent_nodes
    # other_parent_node_list.remove(parent_node)
    availlable_parent_node_list=[]
    for other_parent_node in other_parent_node_list:
        if is_node_in_table(other_parent_node.id,availlable_table_list):
            availlable_parent_node_list.append(other_parent_node)
    return availlable_parent_node_list




    

def build_dual_octree_graph(old_G, depth,full_depth=8):
    """
    双方向オクツリーグラフを作成する.
    """
    global time_dict, time_list
    if depth==full_depth:
        return old_G
    print(f"extract_octree_graph:depth: {depth}")

    # 頂点集合V_dを作成する.
    parent_node_list = []
    leaf_node_list = []
    new_G = old_G.copy()

    # 頂点の追加
    t0 = time.time()
    for node in old_G.nodes():
        #葉ノードでない⇒ノードを分割して、子ノードを要素として追加する.
        if not node.is_leaf:
            parent_node_list.append(node)
            add_child_node(new_G,node)
        #葉ノードならばそのままにする.
        else:
            leaf_node_list.append(node)
    #分割された親ノードを除去する.
    new_G.remove_nodes_from(parent_node_list)
    t1 = time.time()
    time_dict["頂点の追加"] = t1 - t0

    # 辺の追加１
    t2 = time.time()
    # 同一親子関係同士の接続辺を追加する.
    add_same_parent_child_edge(new_G, parent_node_list)
    t3 = time.time()
    time_dict["辺の追加１"] = t3 - t2

    # 辺の追加２
    t4 = time.time()
    # 葉ノードがGの時点で接続していた頂点Listを列挙
    connected_nodes_list = {
        leaf_node: search_connected_node_list(old_G, leaf_node)
        for leaf_node in leaf_node_list
    }
    # 葉ノードに元の接続先の子ノードを適切に接続する.
    for leaf_node in leaf_node_list:
        for connected_node in connected_nodes_list[leaf_node]:
            if connected_node.is_leaf:
                continue
            connect_leaf_node2child_node_by_lookup_table(new_G, old_G, leaf_node, connected_node)
    t5 = time.time()
    time_dict["辺の追加２"] = t5 - t4

    # 辺の追加３
    t6 = time.time()
    # 異親子関係同士 and 同じ深さ同士のChild_node間に接続辺を追加する.
    if depth < 0:  # ここの処理のみにしか使えない（使わなくてもいい）
        for parent_node in parent_node_list:
            # 親同士が接続しているノードがavailable_parent_node
            if parent_node.parent is None:
                continue
            brother_node_list = parent_node.parent.children
            availlable_parent_node_list = [
                brother_node
                for brother_node in brother_node_list
                if old_G.has_edge(parent_node, brother_node)
            ]
            for availlable_parent_node in availlable_parent_node_list:
                connect_direct = LOOKUP_TABLE_DIRECTION[parent_node.id][availlable_parent_node.id]
                for idx1, idx2 in CONNECT_PATTERNS[connect_direct]:
                    new_G.add_edge(parent_node.children[idx1], availlable_parent_node.children[idx2])

    else:
        parent_node_set=set(parent_node_list)
        # 親ノードの隣接ノードを取得する.
        neighbor_map={
            parent_node:[
                neighbor_node
                for neighbor_node in old_G.neighbors(parent_node)
                if neighbor_node in parent_node_set and neighbor_node != parent_node
            ]
            for parent_node in parent_node_list
        }
        for source_parent_node in parent_node_list:
            candidate_parent_nodes=neighbor_map[source_parent_node]
            # 位置関係で繋がり方向を求める.
            for target_parent_node in candidate_parent_nodes:
                if target_parent_node.is_leaf:
                    continue
                # source_parent_nodeとtarget_parent_nodeの位置関係を判定
                dx = target_parent_node.x - source_parent_node.x
                dy = target_parent_node.y - source_parent_node.y
                dz = target_parent_node.z - source_parent_node.z

                direction_code = get_direction_code(dx, dy, dz)
                if direction_code == -1:
                    continue
                for idx1, idx2 in CONNECT_PATTERNS[direction_code]:
                    new_G.add_edge(source_parent_node.children[idx1], target_parent_node.children[idx2])
    t7 = time.time()
    time_dict["辺の追加３"] = t7 - t6

    # 各深さごとの時間を記録
    time_list.append({
        "depth": depth,
        "頂点の追加": time_dict["頂点の追加"],
        "辺の追加１": time_dict["辺の追加１"],
        "辺の追加２": time_dict["辺の追加２"],
        "辺の追加３": time_dict["辺の追加３"],
    })

    # 可視化関数を呼び出す.
    visualize_dual_octree_graph_plotly3d(new_G, depth)
    print(f"time_dict:\n {time_dict}")
    return build_dual_octree_graph(new_G, depth + 1)
#def visualize_dual_octree_graph_3d(graph, depth, filename_prefix="octree_graph"):
    """
    グラフの3D可視化を行い、ファイルに保存する関数

    Parameters:
        graph: networkx.Graph
            可視化対象のグラフ
        depth: int
            現在の深さ（ファイル名に利用）
        filename_prefix: str
            保存ファイル名のプレフィックス
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    node_xyz = {}
    node_depths = []
    node_sizes = []
    for node in graph.nodes():
        # 中心座標
        cx = node.x + node.size / 2
        cy = node.y + node.size / 2
        cz = node.z + node.size / 2
        if node.depth==1:
            print("--------------------------------")
            print(node.id)
            print(node.size)
            print(node.x)
            print(node.y)
            print(node.z)
            print("--------------------------------")
            
        node_xyz[node] = (cx, cy, cz)
        d = getattr(node, 'depth', 0)
        node_depths.append(d)
        # サイズで点の大きさを変える（最小10, 最大100）
        node_sizes.append(np.clip(node.size * 5, 10, 100))

    xs = [pos[0] for pos in node_xyz.values()]
    ys = [pos[1] for pos in node_xyz.values()]
    zs = [pos[2] for pos in node_xyz.values()]

    # depthごとに色を割り当てる
    unique_depths = sorted(set(node_depths))
    norm = mcolors.Normalize(vmin=min(unique_depths), vmax=max(unique_depths))
    cmap = cm.get_cmap('jet', len(unique_depths))
    depth_to_color = {d: cmap(norm(d)) for d in unique_depths}
    colors = [depth_to_color[d] for d in node_depths]

    # ノードを描画（depthごとに色分け、サイズも反映）
    ax.scatter(xs, ys, zs, s=node_sizes, c=colors, alpha=0.85, edgecolors='k', linewidths=0.2)

    # エッジを描画（さらに薄く）
    for edge in graph.edges():
        n1, n2 = edge
        x = [node_xyz[n1][0], node_xyz[n2][0]]
        y = [node_xyz[n1][1], node_xyz[n2][1]]
        z = [node_xyz[n1][2], node_xyz[n2][2]]
        ax.plot(x, y, z, color='gray', alpha=0.15, linewidth=0.7)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{depth}_3d.png", dpi=300)
    plt.close(fig)

def visualize_dual_octree_graph_plotly3d(graph, depth, filename_prefix="octree_graph"):
    """
    クリックで近傍ハイライト (Scatter3d 対応版)
    """
    import plotly.graph_objects as go
    import numpy as np, json, html, time, os

    # ---------- 1. ノード/エッジ numpy 化 ----------
    nodes = list(graph.nodes());  N=len(nodes)
    xs,ys,zs = np.zeros(N),np.zeros(N),np.zeros(N)
    sizes,colors,hovers = np.zeros(N),[],[]
    cmap={0:'#e41a1c',1:'#377eb8',2:'#4daf4a',3:'#ff7f00',
          4:'#984ea3',5:'#00bfc4',6:'#ffdf00',7:'#ca0020'}

    n2i={n:i for i,n in enumerate(nodes)}
    for i,n in enumerate(nodes):
        xs[i],ys[i],zs[i]=n.x,n.y,n.z
        sizes[i]=max(n.size*5,8)
        d=getattr(n,'depth',0); colors.append(cmap[d%8])
        hovers.append(f"ID:{n.id}<br>depth:{d}<br>size:{n.size:.3f}")

    edges=[[n2i[u],n2i[v]] for u,v in graph.edges()]
    E=len(edges); ex,ey,ez=np.empty(E*3),np.empty(E*3),np.empty(E*3)
    for j,(u,v) in enumerate(edges):
        k=j*3
        ex[k:k+3]=[xs[u],xs[v],None]
        ey[k:k+3]=[ys[u],ys[v],None]
        ez[k:k+3]=[zs[u],zs[v],None]

    # ---------- 2. ベーストレース ----------
    edge_trace=go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',
          line=dict(color='rgba(70,70,70,0.6)',width=2),hoverinfo='none')
    node_trace=go.Scatter3d(x=xs,y=ys,z=zs,mode='markers+text',
          marker=dict(size=sizes,color=colors,line=dict(width=0.5,color='rgb(50,50,50)')),
          text=[str(n.id) for n in nodes],textposition="top center",
          hovertext=hovers,hoverinfo='text')

    fig=go.Figure([edge_trace,node_trace])
    fig.update_layout(title=f'Octree Graph (depth={depth})',
        scene=dict(aspectmode='data'),template='plotly_white',
        margin=dict(l=0,r=0,b=0,t=40),hovermode='closest',showlegend=False)

    # ---------- 3. HTML 出力 ----------
    plot_id=f"plot_{depth}_{int(time.time())}"
    out=f"{filename_prefix}_{depth}.html"
    edges_js=html.escape(json.dumps(edges))
    xs_js,ys_js,zs_js=[json.dumps(arr.tolist()) for arr in (xs,ys,zs)]

    with open(out,'w',encoding='utf-8') as fh:
        fh.write(f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script></head><body>
<div id="{plot_id}" style="width:100%;height:95vh;"></div>
<script>
document.addEventListener('DOMContentLoaded',()=>{{
 const fig={fig.to_json()};
 const div=document.getElementById('{plot_id}');
 const xs={xs_js},ys={ys_js},zs={zs_js},edges={edges_js},N={N};
 const nbr=Array.from({{length:N}},()=>[]);
 edges.forEach(([u,v])=>{{nbr[u].push(v);nbr[v].push(u);}});

 Plotly.newPlot(div,fig.data,fig.layout).then(()=>{{
   // ハイライト用トレース (ノード+エッジ) を追加
   Plotly.addTraces(div,[{{x:[],y:[],z:[],mode:'markers',
     marker:{{color:'rgba(255,0,0,0.9)',size:12}},hoverinfo:'none',visible:false}},
     {{x:[],y:[],z:[],mode:'lines',line:{{color:'rgba(255,0,0,0.8)',width:4}},
       hoverinfo:'none',visible:false}}]);
   const redNodeIdx=div.data.length-2, redEdgeIdx=div.data.length-1;
   let highlighted=false;

   function reset(){{
     if(!highlighted) return;
     Plotly.update(div,{{visible:false}},{{}},[redNodeIdx,redEdgeIdx]);
     highlighted=false;
   }}
   function highlight(i){{
     const nx=[xs[i]],ny=[ys[i]],nz=[zs[i]];
     const ex=[],ey=[],ez=[];
     nbr[i].forEach(j=>{{
        nx.push(xs[j]);ny.push(ys[j]);nz.push(zs[j]);
        ex.push(xs[i],xs[j],null);
        ey.push(ys[i],ys[j],null);
        ez.push(zs[i],zs[j],null);
     }});
     Plotly.update(div,{{x:[nx],y:[ny],z:[nz],visible:true}},{{}},[redNodeIdx]);
     Plotly.update(div,{{x:[ex],y:[ey],z:[ez],visible:true}},{{}},[redEdgeIdx]);
     highlighted=true;
   }}

   div.on('plotly_click',ev=>{{
     if(!ev.points.length) return reset();
     const p=ev.points[0];
     if(p.curveNumber!==1) return; // ノードのみ
     highlighted?reset():highlight(p.pointNumber);
   }});
 }});
}});
</script></body></html>""")

    print("HTML saved:", os.path.abspath(out))
    return out
