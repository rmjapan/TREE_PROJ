"""
Octree データ構造とその構築関連の機能を提供するモジュール

このモジュールでは、3次元ボクセルデータを効率的に表現するためのOctree構造を実装します。
Octreeは空間を階層的に8分割することで、均一でないデータの効率的な表現を可能にします。
"""

from curses import KEY_UNDO
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from ocnn.octree import octree,points
from ocnn.utils import cumsum
from ocnn.octree.shuffled_key import xyz2key,key2xyz
import sys
import time
import networkx as nx

sys.path.append('/home/ryuichi/tree/TREE_PROJ/')
from my_dataset.svsdataset import SvsDataset
from Octree.octfusion.models.networks.dualoctree_networks.dual_octree import DualOctree
from Octree.graph_utils import edge_index_to_networkx
from Octree.dual_octree import build_dual_octree_graph
class OctreeNode:
    """
    Octreeのノードを表すクラス
    
    Attributes:
        is_leaf (bool): 葉ノードかどうか
        value (float, optional): ノードの値（-1: 空, 1: 占有, その他: 中間状態）
        children (List[OctreeNode], optional): 子ノードのリスト（内部ノードの場合）
    """
    def __init__(self, is_leaf: bool, value: Optional[float] = None, 
                 children: Optional[List['OctreeNode']] = None,depth: int = 0,x:int=0,y:int=0,z:int=0,size:int=0,id:int=0,id_list=None,parent=None):
        """
        OctreeNodeを初期化します
        
        Args:
            is_leaf: 葉ノードかどうか
            value: ノードの値
            children: 子ノードのリスト（内部ノードの場合）
        """
        self.is_leaf = is_leaf    # 葉ノードかどうか
        self.value = value        #空白:-1,葉:0,枝:0.5,幹:1
        self.children = children  # 8個の子ノード（内部ノードの場合）
        self.parent=parent
        self.depth = depth
        self.x=x
        self.y=y
        self.z=z    
        self.size=size
        self.id=id
        self.id_list=id_list
        

    def __repr__(self) -> str:
        """ノードの文字列表現を返します"""
        if self.is_leaf:
            return f"LeafNode(value={self.value})"
        else:
            return f"InternalNode(children_count={len(self.children) if self.children else 0})"
class Octree4voxel(octree.Octree):
    def __init__(self,depth:int=8,full_depth:int=3,batch_size:int=1,device:torch.device=torch.device('cpu')):
        super().__init__(depth=depth,full_depth=full_depth,batch_size=batch_size,device=device)
        self.depth=depth
        self.full_depth=full_depth
        self.batch_size=batch_size
        self.device=device
    
        #xyz->モートンキーに変換
    def build_octree_morton(self,svs_voxel:torch.Tensor,blank_value_flag=False):
        """
        モートン法によるボクセルデータからOctreeを構築する
        """
        if svs_voxel.ndim==3:
            scale=2**(self.depth-1)
            
            x,y,z=self.get_valid_voxel_indices(svs_voxel)
            x=x*scale
            y=y*scale
            z=z*scale
            
            
            # x,y,z=torch.where(svs_voxel!=-2)
            b=None if self.batch_size==1 else torch.arange(self.batch_size,device=self.device)
            key=xyz2key(x,y,z,b,self.depth)
            node_key, idx, counts = torch.unique(key, sorted=True, 
            return_inverse=True, return_counts=True)


            for d in range(self.full_depth+1):
                self.octree_grow_full(d,update_neigh=False)
                
            for d in range(self.depth, self.full_depth, -1):
                pkey = node_key>>3
                
                pkey,pidx,_=torch.unique_consecutive(
                    pkey,
                    return_inverse=True,
                    return_counts=True
                    )
                
                key=(pkey.unsqueeze(-1)<<3)+torch.arange(8,device=self.device)
                self.keys[d]=key.view(-1)
                self.nnum[d]=key.numel()
                self.nnum_nempty[d]=node_key.numel()
       
                addr=(pidx<<3)|(node_key%8)
                children=-torch.ones(
                    self.nnum[d].item(),dtype=torch.int32,device=self.device)
                children[addr]=torch.arange(
                    self.nnum_nempty[d],dtype=torch.int32,device=self.device)
                self.children[d]=children
                
                node_key=pkey if self.batch_size==1 else \
                    ((pkey>>45)<<48)|(pkey&((1<<45)-1))
            
            d=self.full_depth
            children=-torch.ones_like(self.children[d])
            nempty_idx=node_key if self.batch_size==1 else \
                ((node_key>>48)<<(3*d))|(node_key&((1<<48)-1))
            children[nempty_idx]=torch.arange(
                node_key.numel(),dtype=torch.int32,device=self.device)
            self.children[d]=children
            self.nnum_nempty[d]=node_key.numel()
            
    def get_valid_voxel_indices(self, svs_voxel: torch.Tensor):
        """
        ボクセルデータの内、空白のボクセルを除いたインデックスを取得する.
        """
        #ボクセルデータの内、空白のボクセルを除いたインデックスを取得する
        x,y,z=torch.where(svs_voxel!=-1)
        x=x/2**(self.depth-1)
        y=y/2**(self.depth-1)
        z=z/2**(self.depth-1)
        return x,y,z
    
    def build_octree_recursive(self,data: torch.Tensor, depth: int = 0,full_depth:int=3) -> OctreeNode:
        """
        3次元ボクセルデータからOctreeを構築します
        
        Args:
            data: 3次元のNumPy配列（立方体形状のボクセルデータ）
            depth: 開始深さレベル
            
        Returns:
            構築されたOctreeのルートノード
            
        Raises:
            AssertionError: 入力データが立方体でない場合
        """
        
        def recurse(x0: int, y0: int, z0: int, size: int, depth: int, id: int, 
                    current_id_list: List[int]) -> OctreeNode:
            """
            再帰的にOctreeを構築する内部関数
            
            Args:
                x0, y0, z0: 現在の領域の開始座標
                size: 現在の領域のサイズ
                depth: 現在の深さ
                id: 現在のノードのID
                current_id_list: 現在のノードまでのIDパス
                parent: 親ノード
                
            Returns:
                生成されたOctreeノード
            """
            # 対象領域のデータを取得
            cube = data[x0:x0+size, y0:y0+size, z0:z0+size]
            first_value=cube[0,0,0]
            # 同じ値のみで構成される場合は葉ノードとして返す
            if cube.eq(first_value).all() and depth > 2:
                return OctreeNode(
                    is_leaf=True, 
                    value=first_value, 
                    depth=depth,
                    x=x0,
                    y=y0,
                    z=z0,
                    size=size,
                    id=id,
                    id_list=current_id_list.copy(),
                )
            
            # これ以上分割できない場合も葉ノードとして返す
            elif size == 1 and depth > self.full_depth:
                return OctreeNode(
                    is_leaf=True, 
                    value=first_value, 
                    depth=depth,
                    x=x0,
                    y=y0,
                    z=z0,
                    size=size,
                    id=id,
                    id_list=current_id_list.copy(),
                )
            
            # それ以外の場合は分割して再帰的に処理
            else:
                half = size // 2
                children = []
                
                # 8分割して子ノードを生成
                child_id = 0
                for dx in [0, half]:
                    for dy in [0, half]:
                        for dz in [0, half]:
                            # 子ノードのIDパスを作成
                            child_id_list = current_id_list.copy()
                            child_id_list.append(child_id)
                            
                            # 子ノードを再帰的に生成
                            child = recurse(
                                x0 + dx, y0 + dy, z0 + dz, 
                                half, depth + 1, 
                                child_id, child_id_list
                            )
                            
                            children.append(child)
                            child_id += 1
                
                # 内部ノードを生成
                node = OctreeNode(
                    is_leaf=False, 
                    children=children, 
                    depth=depth,
                    x=x0,
                    y=y0,
                    z=z0,
                    size=size,
                    id=id,
                    id_list=current_id_list.copy(),
                )
                
                return node



        assert data.shape[0] == data.shape[1] == data.shape[2], "入力データは立方体である必要があります"
        
        # Octreeの構築を開始
        size = data.shape[0]
        return recurse(0, 0, 0, size, depth, id=0, current_id_list=[0])
                
    def get_input_feature(self,all_leaf_nodes=True):
        features=[]
        depth=self.depth
        if 'P' in self.feature:
            #グローバル座標を-1~1に正規化
            scale=2**(1-self.depth)
            global_points=self.data*scale-1.0
            features.append(global_points)
        if 'V' in self.feature:
            features.append(self.value)
class DualOctree4voxel(DualOctree):
    def __init__(self,octree):
        self.octree=octree
        self.device=octree.device
        self.depth=octree.depth
        self.full_depth=octree.full_depth
        self.batch_size=octree.batch_size
        self.nnum=octree.nnum
        self.nenum=octree.nnum_nempty
        self.ncum=cumsum(self.nnum,dim=0,exclusive=True)
        #葉ノード:（全体ノードの数）-（非空ノードの数）=空ノードの数
        self.lnum=self.nnum-self.nenum
        self.node_depth=self._node_depth()
        self.child=torch.cat(
            [child for child in octree.children if child != None]
        )
        self.key=torch.cat(
            [key for key in octree.keys if key != None]
        )
        self.keyd=self.key | (self.node_depth<<58)
        
        xyzi=key2xyz(self.key)
        x,y,z,i=xyzi
        self.xyzi=torch.stack((x,y,z,i),dim=1)
        self.xyz=self.xyzi[:, :3]
        self.batch=self.xyzi[:, 3]
        
        super().__init__(octree)
        
        
import trimesh
import numpy as np

def sample_points_from_off(path, n_points=1024):
    mesh = trimesh.load(path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()  # 複数メッシュを統合
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)

def build_octrees(device, batch_size, svs_dataset, modelnet10_path):
    """Octree構築のためのヘルパー関数"""
    # Octree4voxelの構築
    octree4voxel = Octree4voxel(depth=8, full_depth=3, batch_size=batch_size, device=device)
    # Octree4pointの構築
    octree4point = octree.Octree(depth=8, full_depth=3, batch_size=batch_size, device=device)
    # メッシュデータのサンプル点を取得
    sampled_points = sample_points_from_off(modelnet10_path)
    points_tensor = torch.from_numpy(sampled_points).to(device)
    point_cloud = points.Points(points_tensor)
    # ボクセルデータの取得
    voxel_data = svs_dataset[0].squeeze(0).to(device)
    return octree4voxel, octree4point, point_cloud, voxel_data

def process_dual_octree_graph(octree4voxel, octree4point, point_cloud, voxel_data):
    """DualOctreeGraphの構築と後処理"""
    # Octree構築
    octree4voxel.build_octree_morton(voxel_data, blank_value_flag=False)
    octree4point.build_octree(point_cloud)
    # DualOctreeGraphの構築
    dual_octree_graph_voxel = DualOctree4voxel(octree4voxel)
    dual_octree_graph_voxel.post_processing_for_docnn()
    dual_octree_graph_point = DualOctree(octree4point)
    dual_octree_graph_point.post_processing_for_docnn()
    return dual_octree_graph_voxel, dual_octree_graph_point

def build_networkx_graphs(dual_octree_graph_voxel, dual_octree_graph_point, graph_depth):
    """networkxグラフの構築"""
    G_voxel = edge_index_to_networkx(
        dual_octree_graph_voxel._graph[graph_depth]['edge_idx'],
        dual_octree_graph=dual_octree_graph_voxel,
        depth=graph_depth
    )
    G_point = edge_index_to_networkx(
        dual_octree_graph_point._graph[graph_depth]['edge_idx'],
        dual_octree_graph=dual_octree_graph_point,
        depth=graph_depth
    )
    return G_voxel, G_point

def visualize_graphs(G_voxel, G_point, graph_depth):
    """グラフの可視化"""
    from Octree.dual_octree import visualize_dual_octree_graph_plotly3d
    visualize_dual_octree_graph_plotly3d(G_voxel, depth=graph_depth, filename_prefix="octree_dual_graph")
    visualize_dual_octree_graph_plotly3d(G_point, depth=graph_depth, filename_prefix="octree_dual_graph4point")

def main():
    import time

    # デバイスとパス等の設定
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    batch_size = 1
    svs_dataset = SvsDataset()
    modelnet10_path = "/home/ryuichi/tree/TREE_PROJ/Octree/dataset/monitor_0368.off"
    graph_depth = 7
    # Octree構築
    octree4voxel=Octree4voxel(depth=8,full_depth=3,batch_size=batch_size,device=device)
    t0 = time.time()
   
    octree4voxel, octree4point, point_cloud, voxel_data = build_octrees(device, batch_size, svs_dataset, modelnet10_path)
    t1 = time.time()

    # DualOctreeGraph構築と後処理
    dual_octree_graph_voxel, dual_octree_graph_point = process_dual_octree_graph(
        octree4voxel, octree4point, point_cloud, voxel_data
    )
    t2 = time.time()

    # networkxグラフ構築
    G_voxel, G_point = build_networkx_graphs(dual_octree_graph_voxel, dual_octree_graph_point, graph_depth)
    t3 = time.time()

    # 可視化
    visualize_graphs(G_voxel, G_point, graph_depth)
    t4 = time.time()

    print(f"build octree time: {t1-t0:.3f}秒")
    print(f"build dual octree graph time: {t2-t1:.3f}秒")
    print(f"build networkx graph time: {t3-t2:.3f}秒")
    print(f"visualize time: {t4-t3:.3f}秒")

if __name__ == "__main__":
    # main()
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device=torch.device('cpu')
    batch_size=1
    svs_dataset=SvsDataset()
    point_cloud=sample_points_from_off("/home/ryuichi/tree/TREE_PROJ/Octree/dataset/monitor_0368.off")
    point_cloud=torch.from_numpy(point_cloud).to(device)
    point_cloud=points.Points(point_cloud)
    
  
    voxel_data=svs_dataset[0].squeeze(0).to(device)
    
    octree4voxel=Octree4voxel(depth=8,full_depth=3,batch_size=1,device=device)
    octree4point=octree.Octree(depth=8,full_depth=3,batch_size=1,device=device)
    octree4point.build_octree(point_cloud)
    
    octree4voxel.build_octree_morton(voxel_data, blank_value_flag=False)
    dual_octree_graph_voxel = DualOctree4voxel(octree4voxel)
    print(f"key.shape:{dual_octree_graph_voxel.key.shape}")
    print(f"node_depth.shape:{dual_octree_graph_voxel.node_depth.shape}")
    
    print(f"深さ0から見る")
    print(f"index[深さ0]:{dual_octree_graph_voxel.ncum[0]}~{dual_octree_graph_voxel.ncum[1]}")
    print(f"key[深さ0]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[0]:dual_octree_graph_voxel.ncum[1]]}")

    # print(f"keyd[深さ0]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[0]:dual_octree_graph_voxel.ncum[1]]}")
    # print(f"node_depth[深さ0]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[0]:dual_octree_graph_voxel.ncum[1]]}")
    
  
    
    print(f"深さ１から見る")
    print(f"index[深さ1]:{dual_octree_graph_voxel.ncum[1]}~{dual_octree_graph_voxel.ncum[2]}")
    print(f"key[深さ1]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[1]:dual_octree_graph_voxel.ncum[2]]}")

    # print(f"keyd[深さ1]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[1]:dual_octree_graph_voxel.ncum[2]]}")

    # print(f"node_depth[深さ1]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[1]:dual_octree_graph_voxel.ncum[2]]}")
    
 
    print(f"深さ２から見る")
    print(f"index[深さ2]:{dual_octree_graph_voxel.ncum[2]}~{dual_octree_graph_voxel.ncum[3]}")
    # print(f"key[深さ2]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[2]:dual_octree_graph_voxel.ncum[3]]}")
    # print(f"keyd[深さ2]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[2]:dual_octree_graph_voxel.ncum[3]]}")
    # print(f"node_depth[深さ2]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[2]:dual_octree_graph_voxel.ncum[3]]}")
 
    
    
    print(f"深さ3から見る")
    print(f"index[深さ3]:{dual_octree_graph_voxel.ncum[3]}~{dual_octree_graph_voxel.ncum[4]}")
    print(f"key[深さ3]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[3]:dual_octree_graph_voxel.ncum[4]]}")
    # print(f"keyd[深さ3]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[3]:dual_octree_graph_voxel.ncum[4]]}")
    # print(f"node_depth[深さ3]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[3]:dual_octree_graph_voxel.ncum[4]]}")
    

    
    print(f"深さ4から見る")
    print(f"index[深さ4]:{dual_octree_graph_voxel.ncum[4]}~{dual_octree_graph_voxel.ncum[5]}")
    print(f"key[深さ4]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[4]:dual_octree_graph_voxel.ncum[5]]}")
    # print(f"keyd[深さ4]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[4]:dual_octree_graph_voxel.ncum[5]]}")
    # print(f"node_depth[深さ4]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[4]:dual_octree_graph_voxel.ncum[5]]}")
    
    print(f"深さ5から見る")
    print(f"index[深さ5]:{dual_octree_graph_voxel.ncum[5]}~{dual_octree_graph_voxel.ncum[6]}")
    print(f"key[深さ5]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[5]:dual_octree_graph_voxel.ncum[6]]}")
    # print(f"keyd[深さ5]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[5]:dual_octree_graph_voxel.ncum[6]]}")
    # print(f"node_depth[深さ5]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[5]:dual_octree_graph_voxel.ncum[6]]}")
    print(f"深さ6から見る")
    print(f"index[深さ6]:{dual_octree_graph_voxel.ncum[6]}~{dual_octree_graph_voxel.ncum[7]}")
    print(f"key[深さ6]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[6]:dual_octree_graph_voxel.ncum[7]]}")
    # print(f"keyd[深さ6]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[6]:dual_octree_graph_voxel.ncum[7]]}")
    # print(f"node_depth[深さ6]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[6]:dual_octree_graph_voxel.ncum[7]]}")

    print(f"深さ7から見る")
    print(f"index[深さ7]:{dual_octree_graph_voxel.ncum[7]}~{dual_octree_graph_voxel.ncum[8]}")
    print(f"key[深さ7]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[7]:dual_octree_graph_voxel.ncum[8]]}")
    # print(f"keyd[深さ7]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[7]:dual_octree_graph_voxel.ncum[8]]}")
    # print(f"node_depth[深さ7]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[7]:dual_octree_graph_voxel.ncum[8]]}")
    
    print(f"深さ8から見る")
    print(f"index[深さ8]:{dual_octree_graph_voxel.ncum[8]}~{dual_octree_graph_voxel.ncum[9]}")
    print(f"key[深さ8]:{dual_octree_graph_voxel.key[dual_octree_graph_voxel.ncum[8]:dual_octree_graph_voxel.ncum[9]]}")
    # print(f"keyd[深さ8]:{dual_octree_graph_voxel.keyd[dual_octree_graph_voxel.ncum[8]:dual_octree_graph_voxel.ncum[9]]}")
    # print(f"node_depth[深さ8]:{dual_octree_graph_voxel.node_depth[dual_octree_graph_voxel.ncum[8]:dual_octree_graph_voxel.ncum[9]]}")
    
    
    
    