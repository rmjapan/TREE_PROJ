# Octree Visualization with Interactive Child Key Display

このプロジェクトは、Octree構造を3D可視化し、親キーをクリックすると子キーを表示する対話的なビジュアライゼーション機能を提供します。

## 機能

- Octree構造の3D可視化
- 親キーをクリックすると子キーを表示
- カラーグラデーションによる空間位置の視覚化
- 赤色の頂点マーカー表示
- 直感的な座標軸と目盛り

## 使用方法

### 1. テスト可視化の実行

```bash
python visualize_octree.py
```

このスクリプトは、深さ1の親キー（0-7）を可視化します。生成されたHTMLファイルをブラウザで開いて確認してください。

### 2. 親キーをクリックして子キーを表示する

1. ブラウザでHTMLファイルを開く
2. キューブにマウスを合わせると「Click to view children」というメッセージが表示されます
3. キューブをクリックすると、JSONファイルのダウンロードが始まります
4. ダウンロードされたJSONファイルを`octree_data`フォルダに保存します
5. 表示される指示に従って、子キー表示用のPythonスクリプトを実行します:
   ```bash
   python octree_data/visualize_children.py
   ```
6. 新しいHTMLファイルが生成され、選択した親キーの子キーが表示されます

### 3. コードからの使用

```python
from Octree.octree import Octree4voxel

# Octreeの初期化
octree = Octree4voxel(depth=8, full_depth=3)

# 特定の親キーを可視化
parent_keys = [0, 1, 2, 3]  # 親キーのリスト
depth = 1                   # 深さ
html_file = octree.visualize_parent_key(parent_keys, depth)

# または、特定の親キーの子キーを直接可視化
parent_key = 3  # 親キー
depth = 1       # 親キーの深さ
html_file = octree.visualize_child_keys(parent_key, depth)
```

## 仕組み

1. `octree.py`がOctreeのデータ構造とキー計算を管理します
2. `visualizer.py`がPlotlyを使用して3D可視化を行います
3. クリックイベントはJavaScriptを通じて処理され、子キーの計算と可視化が行われます

## 注意

- 大量のキューブを可視化する場合は、パフォーマンスのために頂点のサンプリングが行われます
- JSONファイルの保存とPythonスクリプトの実行が必要なのは、ブラウザのセキュリティ制限によるものです 