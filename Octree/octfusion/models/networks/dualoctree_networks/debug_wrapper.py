import torch
import os
import sys
from visualize_debug import visualize_dual_octree_debug

class DualOctreeDebugWrapper:
    """
    デュアルオクトリーのデバッグ用ラッパークラス
    dual_octree.pyのコードの動作を可視化するためのラッパー
    """
    
    def __init__(self, model):
        """
        元のデュアルオクトリーモデルをラップ
        
        Parameters:
            model: デュアルオクトリーのモデルインスタンス
        """
        self.model = model
        self.debug_dir = 'dual_octree_debug'
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_count = 0
        
    def debug_point(self, vi, vj, depth, message=None):
        """
        デバッグポイントで可視化を実行
        
        Parameters:
            vi, vj: 対象ノードインデックス
            depth: 深さ
            message: デバッグメッセージ（オプション）
        """
        if message:
            print(f"DEBUG[{self.debug_count}]: {message}")
        
        # 可視化を実行
        save_dir = os.path.join(self.debug_dir, f'debug_{self.debug_count}')
        visualize_dual_octree_debug(self.model, depth, vi, vj, save_dir=save_dir)
        
        # デバッグ情報をログに記録
        with open(os.path.join(save_dir, 'debug_info.txt'), 'w') as f:
            f.write(f"デバッグポイント #{self.debug_count}\n")
            f.write(f"深さ: {depth}\n")
            f.write(f"ノード vi: {vi.item()}\n")
            f.write(f"ノード vj: {vj.item()}\n")
            if message:
                f.write(f"メッセージ: {message}\n")
            
            # モデルの情報を記録
            f.write("\n--- モデル情報 ---\n")
            if hasattr(self.model, 'remap'):
                f.write(f"方向リマップ: {self.model.remap}\n")
            if hasattr(self.model, 'dir_table'):
                f.write(f"方向テーブル:\n{self.model.dir_table}\n")
            
            # 計算過程の値を記録
            rel_dir = self.model.relative_dir(vi, vj, depth - 1, rescale=False)
            f.write(f"\n--- 計算過程 ---\n")
            f.write(f"相対方向: {rel_dir.item()}\n")
            
            child_vi = self.model.child[vi]
            child_vj = self.model.child[vj]
            
            f.write(f"vi の子ノード: {child_vi.tolist()}\n")
            f.write(f"vj の子ノード: {child_vj.tolist()}\n")
            
            # 変換後の接続関係を計算して記録
            row_o2 = child_vi.unsqueeze(1) * 8 + self.model.dir_table[rel_dir, :]
            row_o2 = row_o2.view(-1)
            if hasattr(self.model, 'ncum'):
                row_o2 = row_o2 + self.model.ncum[depth]
            
            rel_dir_col = self.model.remap[rel_dir]
            col_o2 = child_vj.unsqueeze(1) * 8 + self.model.dir_table[rel_dir_col, :]
            col_o2 = col_o2.view(-1)
            if hasattr(self.model, 'ncum'):
                col_o2 = col_o2 + self.model.ncum[depth]
            
            f.write(f"\n接続されるノードペア:\n")
            for r, c in zip(row_o2, col_o2):
                f.write(f"{r.item()} -> {c.item()}\n")
        
        self.debug_count += 1
        
    def instrument_code(self):
        """
        デュアルオクトリーコードの特定の箇所にデバッグポイントを挿入する
        
        これは実際のコードを書き換えるものではなく、どこにデバッグコードを
        挿入すべきかの例を示すものです。
        """
        print("デュアルオクトリーコードに以下のデバッグポイントを挿入してください:")
        print("\n例：接続関係の計算箇所にデバッグコードを追加\n")
        
        print("""
        # 元のコード
        vi, vj = row[invalid_both_vtx], col[invalid_both_vtx]
        rel_dir = self.relative_dir(vi, vj, depth - 1, rescale=False)
        row_o2 = self.child[vi].unsqueeze(1) * 8 + self.dir_table[rel_dir, :]
        row_o2 = row_o2.view(-1) + ncum_d
        dir_o2 = rel_dir.unsqueeze(1).repeat(1, 4).view(-1)
        rel_dir_col = self.remap[rel_dir]
        col_o2 = self.child[vj].unsqueeze(1) * 8 + self.dir_table[rel_dir_col, :]
        col_o2 = col_o2.view(-1) + ncum_d
        
        # デバッグコードを追加
        vi, vj = row[invalid_both_vtx], col[invalid_both_vtx]
        
        # デバッグポイント - コードの重要な箇所で呼び出し
        if hasattr(self, 'debugger'):
            for i in range(len(vi)):
                self.debugger.debug_point(vi[i], vj[i], depth, f"接続計算 #{i}")
        
        rel_dir = self.relative_dir(vi, vj, depth - 1, rescale=False)
        # 以下は元のコード
        """)
        
        print("\n実際の使用例：\n")
        print("""
        # モデルのインスタンス化
        model = DualOctree()
        
        # デバッグラッパーを設定
        from debug_wrapper import DualOctreeDebugWrapper
        debugger = DualOctreeDebugWrapper(model)
        
        # デバッグラッパーをモデルに追加
        model.debugger = debugger
        
        # 通常通りモデルを実行
        # デバッグポイントで可視化が自動的に実行される
        """)

# 使用例
if __name__ == "__main__":
    # サンプルコード
    print("デバッグラッパーの使用例:")
    print("1. dual_octree.pyの適切な場所にデバッグコードを挿入")
    print("2. モデル実行時にラッパーでデバッグポイントを可視化") 