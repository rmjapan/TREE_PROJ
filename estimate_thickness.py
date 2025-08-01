import networkx as nx

def make_davinch_tree(DG, Edge):
    """
    『枝の直径は分岐する全ての枝の直径の合計である』というダビンチルールの指針に基づいて
    辺の太さを計算する関数
    """
    start_node = Edge[0]
    end_node = Edge[1]
    thickness = 0

    # 同じ太さのエッジリスト
    same_thickness_edge_list = [Edge]

    while True:
        # 次のノードの出るエッジを取得
        adjacent_edge_list = list(DG.out_edges(end_node))

        if len(adjacent_edge_list) == 0:  # 葉に到達
            thickness += 0.35
            for same_edge in same_thickness_edge_list:
                DG.edges[same_edge]["edge"].thickness = thickness
                #辺の属性もここで一緒に
                start_node = same_edge[0]
                
                if DG.nodes[start_node]["node"].attr==1 and DG.nodes[end_node]["node"].attr==1:
                    edge=DG.edges[same_edge]["edge"].attr=1
                else:
                    edge=DG.edges[same_edge]["edge"].attr=0.5
            thickness=thickness*0.4
            return thickness

        if len(adjacent_edge_list) > 1:  # 分岐がある場合
            for edge in adjacent_edge_list:
                thickness += make_davinch_tree(DG, edge)
            for same_edge in same_thickness_edge_list:
                DG.edges[same_edge]["edge"].thickness = thickness
                #辺の属性もここで一緒に
            thickness=thickness*0.63
            return thickness

        elif len(adjacent_edge_list) == 1:  # 次のエッジが1つの場合
            same_thickness_edge_list.append(adjacent_edge_list[0])
            end_node = adjacent_edge_list[0][1]
def  scaleThicknessBylength(DG,Edge):
    """
    Siggraph2011の「Texture-Lobes for Tree Modelling」の方法に基づいて、
    Edgeの太さを計算する関数（一応ダビンチルールを参考にはしているらしい）
    d(u) = droot * (l(u)/l(root))^γ
    d'(u) = d(u) * f(s)*d(u)
        d(u):ノードuの太さ
        droot:根ノードの太さ
        l(u):ノードuの全ての部分木の辺の長さの合計
        l(root):全ての辺の長さの合計
        γ:係数（デフォルトでは1.5)
        f(s):樹種ごとの調整係数らしい（この辺は自分ようにアレンジしてもいいかも）
    この方法は初期値として根ノードの太さを与える必要がある.
    """




