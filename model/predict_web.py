from PIL import Image
from svdiffusion import DiffusionModel
from utils import *
from torchvision import transforms
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify, send_file
import base64
import io
import os

app = Flask(__name__)

# グローバル変数でモデルを保持
model = None
current_voxel = None

def load_model():
    """モデルを読み込む"""
    global model
    data_folder = "/mnt/nas/rmjapan2000/tree/data_dir/train/sketch_cgvi_ver2/right/sketch_right_sobel_18055.png"
    ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=08-losstrain_loss=1.1489.ckpt"
    model = DiffusionModel.load_from_checkpoint(
        ckpt,
        verbose=False,
        batch_size=1,
        with_attention=True,
        with_SE=True
    ).cuda()
    print("Model loaded successfully!")

def process_image(image_path):
    """画像を処理してボクセルを生成"""
    global current_voxel
    
    img = Image.open(image_path).convert('RGB')
    img_size = 256

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    print(f"Image tensor shape: {img_tensor.shape}")

    # ボクセル生成
    voxel = model.sample_with_img(img_tensor.cuda(), steps=21, verbose=True)
    size = 64
    voxel = voxel.reshape([size, size, size])
    current_voxel = voxel.cpu().numpy()
    
    # ファイルに保存
    np.save("web_output.npy", current_voxel)
    
    return current_voxel

def create_3d_plot(voxel, threshold=0.0):
    """ボクセルデータから3Dプロットを作成"""
    size = voxel.shape[0]
    
    # ボクセルデータから点群を抽出
    xyz = []
    colors = []
    sizes = []
    
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if voxel[x, y, z] > threshold:
                    xyz.append([x, y, z])
                    
                    # 値に応じて色とサイズを設定
                    if voxel[x, y, z] > 0.8:
                        colors.append('red')
                        sizes.append(8)
                    elif voxel[x, y, z] > 0.3:
                        colors.append('orange')
                        sizes.append(6)
                    else:
                        colors.append('green')
                        sizes.append(4)
    
    if not xyz:
        return None
    
    xyz = np.array(xyz)
    
    # 3Dスキャッタープロット作成
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8
        ),
        text=[f'Value: {voxel[int(xyz[i, 0]), int(xyz[i, 1]), int(xyz[i, 2])]:.3f}' 
              for i in range(len(xyz))],
        hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>'
    )])
    
    # レイアウト設定
    fig.update_layout(
        title='3D Voxel Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """画像アップロードと処理"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # 画像を保存
    filename = 'uploaded_image.png'
    file.save(filename)
    
    try:
        # 画像処理
        voxel = process_image(filename)
        
        # 3Dプロット作成
        fig = create_3d_plot(voxel)
        if fig is None:
            return jsonify({'error': 'No voxels generated'}), 400
        
        # PlotlyのHTMLを生成
        plot_html = fig.to_html(include_plotlyjs='cdn')
        
        return jsonify({
            'success': True,
            'plot_html': plot_html,
            'voxel_stats': {
                'total_voxels': np.sum(voxel > 0),
                'max_value': float(np.max(voxel)),
                'min_value': float(np.min(voxel))
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/adjust_threshold', methods=['POST'])
def adjust_threshold():
    """閾値調整"""
    global current_voxel
    
    if current_voxel is None:
        return jsonify({'error': 'No voxel data available'}), 400
    
    threshold = float(request.json.get('threshold', 0.0))
    
    try:
        fig = create_3d_plot(current_voxel, threshold)
        if fig is None:
            return jsonify({'error': 'No voxels above threshold'}), 400
        
        plot_html = fig.to_html(include_plotlyjs='cdn')
        
        return jsonify({
            'success': True,
            'plot_html': plot_html,
            'visible_voxels': np.sum(current_voxel > threshold)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_voxel')
def download_voxel():
    """ボクセルデータをダウンロード"""
    if os.path.exists('web_output.npy'):
        return send_file('web_output.npy', as_attachment=True)
    else:
        return jsonify({'error': 'No voxel data available'}), 404

if __name__ == '__main__':
    # モデル読み込み
    load_model()
    
    # Flaskアプリ起動
    app.run(host='0.0.0.0', port=5000, debug=True) 