import sys
import os
sys.path.append("/home/ryuichi/tree/TREE_PROJ")

import torch
import numpy as np
from model.svsae import VQGAN, AutoencoderConfig
from my_dataset.svsdataset import SvsDataLoader
from visualize_func import visualize_voxel_data, visualize_and_save_volume, visualize_with_timeout4voxel
from utils import npz2dense
import matplotlib.pyplot as plt
from pathlib import Path


class VQGANTester:
    def __init__(self, model_path=None, config=None):
        """
        VQGANテストクラス
        
        Args:
            model_path: 訓練済みモデルのパス
            config: モデル設定
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config is None:
            config = AutoencoderConfig(
                latent_channels=1,
                encoder_type="ver2",
                decoder_type="ver2",
                device=self.device
            )
        
        self.config = config
        self.model = VQGAN(
            config=config,
            beta=0.25
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("No model loaded - using random weights for testing")
    
    def load_model(self, model_path):
        """訓練済みモデルを読み込む"""
        if model_path.endswith('.ckpt'):
            # Lightning checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            # Lightning checkpointの場合、state_dictキーから取得
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Lightning形式のキー名を調整（model.で始まるキーを削除）
                adjusted_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # 'model.'を削除
                        adjusted_state_dict[new_key] = value
                    else:
                        adjusted_state_dict[key] = value
                self.model.load_state_dict(adjusted_state_dict, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            # 通常のPyTorchモデル
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
    
    def test_single_sample(self, data_path):
        """単一サンプルでのテスト"""
        print(f"Testing with sample: {data_path}")
        
        # データ読み込み
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            voxel = npz2dense(data)
        else:
            voxel = np.load(data_path)
        
        # 前処理
        voxel_tensor = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        print(f"Input shape: {voxel_tensor.shape}")
        
        # モデル推論
        self.model.eval()
        with torch.no_grad():
            try:
                result = self.model(voxel_tensor, return_dict=False)
                if isinstance(result, (tuple, list)) and len(result) == 4:
                    reconstructed, vq_loss, quantized, encodings = result
                elif isinstance(result, dict):
                    reconstructed = result["reconstructed"]
                    vq_loss = result["vq_loss"]
                    quantized = result["quantized"]
                    encodings = result["encodings"]
                else:
                    print(f"Unexpected output format: {result}")
                    return None
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"VQ Loss: {vq_loss.item():.4f}")
        
        # 結果を可視化
        self.visualize_results(voxel_tensor, reconstructed, quantized, data_path)
        
        return {
            'input': voxel_tensor,
            'reconstructed': reconstructed,
            'quantized': quantized,
            'vq_loss': vq_loss.item()
        }
    
    def test_batch(self, batch_size=1, num_samples=20):
        """バッチでのテスト（VQVAEスタイル）"""
        print(f"Testing with batch size: {batch_size}")
        
        # データローダー作成（VQVAEと同じ設定）
        dataloader = SvsDataLoader(batch_size=batch_size, sub_dataset=False)
        dataloader.setup()
        
        test_loader = dataloader.train_dataloader()  # VQVAEではtrain_dataloaderを使用
        
        self.model.eval()
        total_vq_loss = 0
        num_batches = 0
        count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if count > num_samples:
                    break
                count += 1
                    
                batch = batch.to(self.device)
                print(f"Sample {count} - Input shape: {batch.shape}")
                
                # モデル推論（デバッグ情報付き）
                try:
                    result = self.model(batch, return_dict=False)
                    
                    if isinstance(result, (tuple, list)) and len(result) == 4:
                        reconstructed, vq_loss, quantized, encodings = result
                    elif isinstance(result, dict):
                        reconstructed = result["reconstructed"]
                        vq_loss = result["vq_loss"]
                        quantized = result["quantized"]
                        encodings = result["encodings"]
                    else:
                        print(f"Unexpected output format: {result}")
                        continue
                        
                    # 閾値処理を適用
                    reconstructed_processed = self.apply_threshold_processing(reconstructed)
                    
                    total_vq_loss += vq_loss.item()
                    num_batches += 1
                    
                    print(f"Sample {count} - VQ Loss: {vq_loss.item():.4f}")
                    
                    # 各サンプルをインタラクティブ可視化
                    input_np = batch.squeeze(0).detach().cpu().numpy()
                    recon_np = reconstructed_processed.squeeze(0).detach().cpu().numpy()
                    
                    try:
                        print(f"Visualizing sample {count}...")
                        visualize_with_timeout4voxel(recon_np, timeout=15, title=f"recon_x_{count:02d}")
                        visualize_with_timeout4voxel(input_np[0], timeout=15, title=f"x_{count:02d}")
                    except Exception as e:
                        print(f"Visualization failed for sample {count}: {e}")
                    
                    # 静的画像保存（最初の数サンプルのみ）
                    if count <= 5:
                        self.visualize_batch_results(batch, reconstructed, quantized, count)
                        
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        avg_vq_loss = total_vq_loss / num_batches if num_batches > 0 else 0
        print(f"Average VQ Loss: {avg_vq_loss:.4f}")
        
        return avg_vq_loss
    
    def visualize_results(self, input_tensor, reconstructed, quantized, data_path):
        """結果の可視化"""
        save_dir = Path("vqgan_test_results")
        save_dir.mkdir(exist_ok=True)
        
        sample_name = Path(data_path).stem
        
        # CPU に移動して numpy に変換
        input_np = input_tensor.cpu().squeeze().numpy()
        reconstructed_np = reconstructed.cpu().squeeze().numpy()
        quantized_np = quantized.cpu().squeeze().numpy()
        
        print(f"Input range: [{input_np.min():.3f}, {input_np.max():.3f}]")
        print(f"Reconstructed range: [{reconstructed_np.min():.3f}, {reconstructed_np.max():.3f}]")
        
        # VQVAEスタイルの後処理を適用（閾値処理）
        reconstructed_processed = self.apply_threshold_processing(reconstructed_np)
        print(f"Processed reconstructed range: [{reconstructed_processed.min():.3f}, {reconstructed_processed.max():.3f}]")
        
        # 3D可視化（インタラクティブ）- VQVAEスタイルの可視化関数を使用
        try:
            print("Visualizing input...")
            visualize_with_timeout4voxel(input_np, timeout=15, title=f"Input - {sample_name}")
            
            print("Visualizing reconstructed (raw)...")
            visualize_with_timeout4voxel(reconstructed_np, timeout=15, title=f"Reconstructed Raw - {sample_name}")
            
            print("Visualizing reconstructed (processed)...")
            visualize_with_timeout4voxel(reconstructed_processed, timeout=15, title=f"Reconstructed Processed - {sample_name}")
        except Exception as e:
            print(f"Interactive visualization failed: {e}")
        
        # 静的画像として保存
        self.save_comparison_images(input_np, reconstructed_np, quantized_np, save_dir, sample_name, reconstructed_processed)
    
    def apply_threshold_processing(self, recon_x):
        """VQVAEスタイルの閾値処理を適用"""
        # 0.8以上は1.0, 0.4以上は0.5, 0.0以上は0.0, それ以外は-1.0
        if isinstance(recon_x, np.ndarray):
            recon_tensor = torch.tensor(recon_x)
        else:
            recon_tensor = recon_x
            
        processed = torch.where(
            recon_tensor > 0.8, 1.0,
            torch.where(
                recon_tensor > 0.4, 0.5,
                torch.where(
                    recon_tensor > 0.0, 0.0, -1.0
                )
            )
        )
        
        if isinstance(recon_x, np.ndarray):
            return processed.numpy()
        return processed
    
    def visualize_batch_results(self, input_batch, reconstructed_batch, quantized_batch, batch_idx):
        """バッチ結果の可視化"""
        save_dir = Path("vqgan_batch_results")
        save_dir.mkdir(exist_ok=True)
        
        # 最初の4サンプルを可視化
        num_samples = min(4, input_batch.shape[0])
        
        for i in range(num_samples):
            input_np = input_batch[i].cpu().squeeze().numpy()
            reconstructed_np = reconstructed_batch[i].cpu().squeeze().numpy()
            quantized_np = quantized_batch[i].cpu().squeeze().numpy()
            
            # 閾値処理を適用
            reconstructed_processed = self.apply_threshold_processing(reconstructed_np)
            
            sample_name = f"batch_{batch_idx}_sample_{i}"
            self.save_comparison_images(input_np, reconstructed_np, quantized_np, save_dir, sample_name, reconstructed_processed)
            
            # インタラクティブ可視化（最初のサンプルのみ）
            if i == 0:
                try:
                    print(f"Visualizing batch {batch_idx} sample {i}...")
                    visualize_with_timeout4voxel(input_np, timeout=10, title=f"Batch_{batch_idx}_Input_{i}")
                    visualize_with_timeout4voxel(reconstructed_processed, timeout=10, title=f"Batch_{batch_idx}_Recon_{i}")
                except Exception as e:
                    print(f"Interactive visualization failed for batch sample: {e}")
    
    def save_comparison_images(self, input_np, reconstructed_np, quantized_np, save_dir, sample_name, reconstructed_processed=None):
        """比較画像を保存"""
        # 各テンソルの中央スライスを取得
        input_mid_x = input_np.shape[0] // 2
        input_mid_y = input_np.shape[1] // 2
        input_mid_z = input_np.shape[2] // 2
        
        recon_mid_x = reconstructed_np.shape[0] // 2
        recon_mid_y = reconstructed_np.shape[1] // 2
        recon_mid_z = reconstructed_np.shape[2] // 2
        
        quant_mid_x = quantized_np.shape[0] // 2
        quant_mid_y = quantized_np.shape[1] // 2
        quant_mid_z = quantized_np.shape[2] // 2
        
        print(f"Input shape: {input_np.shape}")
        print(f"Reconstructed shape: {reconstructed_np.shape}")
        print(f"Quantized shape: {quantized_np.shape}")
        
        # 処理後の再構成結果がない場合は作成
        if reconstructed_processed is None:
            reconstructed_processed = self.apply_threshold_processing(reconstructed_np)
        
        # 4つのビューで比較画像を作成（Raw + Processed）
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        # XY plane (Z軸中央)
        axes[0, 0].imshow(input_np[:, :, input_mid_z], cmap='viridis')
        axes[0, 0].set_title(f'Input - XY plane (z={input_mid_z})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(reconstructed_np[:, :, recon_mid_z], cmap='viridis')
        axes[0, 1].set_title(f'Reconstructed Raw - XY plane (z={recon_mid_z})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(reconstructed_processed[:, :, recon_mid_z], cmap='viridis')
        axes[0, 2].set_title(f'Reconstructed Processed - XY plane (z={recon_mid_z})')
        axes[0, 2].axis('off')
        
        # XZ plane (Y軸中央)
        axes[1, 0].imshow(input_np[:, input_mid_y, :], cmap='viridis')
        axes[1, 0].set_title(f'Input - XZ plane (y={input_mid_y})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(reconstructed_np[:, recon_mid_y, :], cmap='viridis')
        axes[1, 1].set_title(f'Reconstructed Raw - XZ plane (y={recon_mid_y})')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(reconstructed_processed[:, recon_mid_y, :], cmap='viridis')
        axes[1, 2].set_title(f'Reconstructed Processed - XZ plane (y={recon_mid_y})')
        axes[1, 2].axis('off')
        
        # YZ plane (X軸中央)
        axes[2, 0].imshow(input_np[input_mid_x, :, :], cmap='viridis')
        axes[2, 0].set_title(f'Input - YZ plane (x={input_mid_x})')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(reconstructed_np[recon_mid_x, :, :], cmap='viridis')
        axes[2, 1].set_title(f'Reconstructed Raw - YZ plane (x={recon_mid_x})')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(reconstructed_processed[recon_mid_x, :, :], cmap='viridis')
        axes[2, 2].set_title(f'Reconstructed Processed - YZ plane (x={recon_mid_x})')
        axes[2, 2].axis('off')
        
        # Quantized features
        axes[3, 0].imshow(quantized_np[:, :, quant_mid_z], cmap='viridis')
        axes[3, 0].set_title(f'Quantized - XY plane (z={quant_mid_z})')
        axes[3, 0].axis('off')
        
        axes[3, 1].imshow(quantized_np[:, quant_mid_y, :], cmap='viridis')
        axes[3, 1].set_title(f'Quantized - XZ plane (y={quant_mid_y})')
        axes[3, 1].axis('off')
        
        axes[3, 2].imshow(quantized_np[quant_mid_x, :, :], cmap='viridis')
        axes[3, 2].set_title(f'Quantized - YZ plane (x={quant_mid_x})')
        axes[3, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{sample_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison images saved: {save_dir / f'{sample_name}_comparison.png'}")


def main():
    """メイン関数"""
    print("=== VQGAN Testing and Visualization ===")
    
    # 訓練済みモデルのパスを指定（存在しない場合はランダム重みでテスト）
    model_path = "/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqgan"
    
    # 最新のチェックポイントを探す
    checkpoint_path = None
    if os.path.exists(model_path):
        checkpoints = [f for f in os.listdir(model_path) if f.endswith('.ckpt')]
        if checkpoints:
            # epoch番号でソートして最新を選択
            checkpoints.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]))
            checkpoint_path = os.path.join(model_path, checkpoints[-1])
            print(f"Found checkpoint: {checkpoint_path}")
            print(f"Available checkpoints: {len(checkpoints)}")
            for ckpt in checkpoints[-3:]:  # 最後の3つを表示
                print(f"  - {ckpt}")
    
    # テスター初期化
    tester = VQGANTester(model_path=checkpoint_path)
    
    # テスト実行（VQVAEスタイル）
    print("\n1. VQGAN Batch testing (VQVAEスタイル)...")
    avg_loss = tester.test_batch(batch_size=1, num_samples=20)
    
    print(f"\n=== Testing Summary ===")
    print(f"Average VQ Loss: {avg_loss:.4f}")
    print("Check the 'vqgan_test_results' and 'vqgan_batch_results' directories for visualization results.")


if __name__ == "__main__":
    main()
