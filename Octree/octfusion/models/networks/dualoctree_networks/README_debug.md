# デュアルオクトリーデバッグ可視化ツール

このツールは、デュアルオクトリーのアルゴリズム、特に親ノード間の接続関係から子ノード間の接続関係を計算する部分を可視化してデバッグするためのものです。

## 概要

`dual_octree.py`のコードには、オクトリーノード間の接続関係を計算する複雑なロジックが含まれています。特に以下のコード部分は理解が難しいため、視覚的に確認できるようにしました：

```python
vi, vj = row[invalid_both_vtx], col[invalid_both_vtx]
rel_dir = self.relative_dir(vi, vj, depth - 1, rescale=False)
row_o2 = self.child[vi].unsqueeze(1) * 8 + self.dir_table[rel_dir, :]
row_o2 = row_o2.view(-1) + ncum_d
dir_o2 = rel_dir.unsqueeze(1).repeat(1, 4).view(-1)
rel_dir_col = self.remap[rel_dir]
col_o2 = self.child[vj].unsqueeze(1) * 8 + self.dir_table[rel_dir_col, :]
col_o2 = col_o2.view(-1) + ncum_d
```

このデバッグツールを使用すると、以下のことが可能になります：

1. 親ノード間の接続関係と方向の視覚化
2. 子ノード間の接続関係の計算過程の確認
3. 接続されるノードペアの可視化

## ファイル構成

- `visualize_debug.py` - 可視化機能のコア実装
- `debug_wrapper.py` - デバッグ用ラッパークラス
- `README_debug.md` - このドキュメント

## インストール

必要なパッケージをインストールします：

```bash
pip install matplotlib networkx plotly torch numpy
```

## 使用方法

### 1. デバッグコードの挿入

`dual_octree.py`の対象となるコード部分に以下のようなデバッグコードを挿入します：

```python
# デバッグモジュールをインポート
from debug_wrapper import DualOctreeDebugWrapper

# モデルの初期化時にデバッグラッパーを設定
def __init__(self, ...):
    # 通常の初期化処理
    ...
    
    # デバッグラッパーを追加
    self.debugger = DualOctreeDebugWrapper(self)

# 対象のメソッド内にデバッグポイントを挿入
def update_dual_node(self, ...):
    ...
    
    # デバッグしたい箇所
    vi, vj = row[invalid_both_vtx], col[invalid_both_vtx]
    
    # デバッグポイントの追加
    if hasattr(self, 'debugger'):
        for i in range(len(vi)):
            self.debugger.debug_point(vi[i], vj[i], depth, f"接続計算 #{i}")
    
    # 通常のコード続行
    rel_dir = self.relative_dir(vi, vj, depth - 1, rescale=False)
    ...
```

### 2. デバッグ実行

通常通りモデルを実行します。デバッグポイントに到達すると、自動的に可視化が生成され保存されます：

```python
# モデルをインスタンス化
model = DualOctree()

# デバッグラッパーを有効化する場合
from debug_wrapper import DualOctreeDebugWrapper
model.debugger = DualOctreeDebugWrapper(model)

# 通常通りモデルを実行
result = model.forward(...)
```

### 3. 可視化結果の確認

デバッグ実行後、`dual_octree_debug`ディレクトリに以下のファイルが生成されます：

- `debug_N/octree_debug_vi_vj_depthD.png` - 静的な可視化画像
- `debug_N/octree_debug_vi_vj_depthD_interactive.html` - インタラクティブな3D可視化（Plotly）
- `debug_N/debug_info.txt` - デバッグ情報のテキストログ

各ビジュアライゼーションでは：
- 赤色ノード: viとその子ノード
- 青色ノード: vjとその子ノード
- 黒色線: 親ノード間の接続
- 緑色線: 計算された子ノード間の接続

## インタラクティブ可視化の操作方法

HTMLファイルをブラウザで開くと、以下の操作が可能です：

- マウスドラッグ: 3Dビューの回転
- マウスホイール: ズームイン/アウト
- ダブルクリック: ビューのリセット
- ノードにホバー: ノード情報の表示

## テスト実行

テスト用のモックデータで可視化をテストするには：

```bash
python visualize_debug.py
```

これにより、`debug_test`ディレクトリにサンプルの可視化が生成されます。

## トラブルシューティング

### 可視化が生成されない

- デバッグラッパーが正しく設定されているか確認
- デバッグポイントが到達する箇所に配置されているか確認
- `visualize_debug.py`と`debug_wrapper.py`がPythonのパスに含まれているか確認

### 可視化が不正確

- モデルの`child`、`dir_table`、`remap`などの属性が期待通りに設定されているか確認
- 深さパラメータが正しく渡されているか確認

## カスタマイズ

より高度な可視化のカスタマイズが必要な場合は、`visualize_debug.py`の以下の関数を調整してください：

- `plot_octant_connections`: 静的な可視化の設定
- `visualize_dual_octree_plotly`: インタラクティブな可視化の設定

## ライセンス

このデバッグツールは研究・教育目的で自由に使用できます。 