import sys

from sklearn.model_selection import learning_curve
import wandb

sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from model.svsae import SVSAE
from utils import npz2dense, dense2sparse,weighted_mse_loss

import random

import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from datetime import datetime
import save_train_result
from svs import voxel_distribution
from svs import visualize_voxel_data
from save_train_result import save_train_result
from multiprocessing import Process
import time
from my_dataset.svsdataset import SvsDataset
def chamfer_distance(p1, p2, 
                    voxel_size=1.0,
                    max_points_per_part=2000,
                    base_weights={'trunk':0.1, 'branch':1.0, 'leaf':1.0}):
    """
    """
    def process_tree_voxels(voxel):
        #(B,1,256,256,256)
        B, C, H, W, D = voxel.shape
        device = voxel.device
        
        # スムージング領域を考慮したマスク
        # チャンネル次元を削除する処理(B,256,256,256)
        trunk_mask = (voxel.squeeze(1) >= 0.8)
        branch_mask = (voxel.squeeze(1) >= 0.4) & (voxel.squeeze(1) <= 0.8)
        # 葉のマスクは0以上0.4
        leaf_mask = (voxel.squeeze(1) >= 0) & (voxel.squeeze(1) <= 0.4)
        
        # 座標正規化(GRID)
        x, y, z = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            torch.linspace(-1, 1, D, device=device),
            indexing='ij'
        )
        
        points_dict = {}
        for part, mask in [('trunk', trunk_mask),
                         ('branch', branch_mask),
                         ('leaf', leaf_mask)]:
            # マスクされた座標を取得
            indices = torch.nonzero(mask)
            if len(indices) == 0:
                points_dict[part] = None
                continue
                
            # 一様サンプリングに変更（中心からの距離に依存しない）
            if len(indices) > max_points_per_part:
                # 全点から均等にサンプリングするように修正
                prob = torch.ones(len(indices), device=device)
                prob /= prob.sum()
                selected = torch.multinomial(prob, max_points_per_part)
                indices = indices[selected]
            
            # 座標取得
            part_points = torch.stack([
                x[indices[:,0], indices[:,1], indices[:,2]],
                y[indices[:,0], indices[:,1], indices[:,2]],
                z[indices[:,0], indices[:,1], indices[:,2]]
            ], dim=1)
            points_dict[part] = part_points
            
        return points_dict

    def batch_cdist(x, y, chunk_size=512):
        """メモリ効率の良い距離計算"""
        # チャンクサイズを動的に調整するように変更
        chunks = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i:i+chunk_size]
            chunks.append(torch.cdist(x_chunk, y))
        return torch.cat(chunks, dim=0)

    total_loss = 0.0
    batch_size = p1.size(0)
    
    for b in range(batch_size):
        #座標を取得する
        pts1 = process_tree_voxels(p1[b:b+1])
        pts2 = process_tree_voxels(p2[b:b+1])
        
        part_loss = 0.0
        for part in ['trunk', 'branch', 'leaf']:
            if pts1[part] is None or pts2[part] is None:
                continue
                
            # 動的重み計算
            def calc_weight(pts):
                return len(pts) / max_points_per_part if len(pts) > 0 else 0.0
            weight_factor = 0.5 * (calc_weight(pts1[part]) + calc_weight(pts2[part]))
            part_weight = base_weights[part] * (1.0 + weight_factor)
            
            # 距離計算
            dist = batch_cdist(pts1[part], pts2[part])
            min_dist1 = torch.min(dist, dim=1)[0].mean()
            min_dist2 = torch.min(dist, dim=0)[0].mean()
            
            part_loss += part_weight * (min_dist1 + min_dist2) / 2
            
        total_loss += part_loss * voxel_size
        
    return total_loss / batch_size

def voxel_miou(pred_continuous, target_discrete):
    """
    Improved mIoU calculation for voxel data
    
    pred_continuous: モデルの連続値出力 (形状: [B,1,H,W,D])
    target_discrete: 離散化済み正解データ (値: -1.0, 0.0, 0.5, 1.0)
    
    Returns: 1 - mIoU (as a loss value)
    """
    # 予測値の離散化（しきい値に基づく分類）
    pred_discrete = torch.full_like(pred_continuous, -1.0)
    mask_trunk = (pred_continuous >= 0.8)
    mask_branch = (pred_continuous >= 0.4) & (pred_continuous < 0.8)
    mask_leaf = (pred_continuous >= 0.0) & (pred_continuous < 0.4)
    
    pred_discrete[mask_trunk] = 1.0
    pred_discrete[mask_branch] = 0.5
    pred_discrete[mask_leaf] = 0.0
    
    # 空白以外の部分のマスク（木の部分だけを対象にする）
    non_empty_pred = (pred_discrete >= 0.0)
    non_empty_target = (target_discrete >= 0.0)
    
    # クラス定義
    class_values = [1.0, 0.5, 0.0]  # trunk, branch, leaf
    class_weights = [3.0, 2.0, 1.0]  # 重要度に基づいた重み付け
    num_classes = len(class_values)
    
    # 予測と正解で木の部分（空白以外）が存在しない場合のチェック
    total_pred_non_empty = non_empty_pred.sum()
    total_target_non_empty = non_empty_target.sum()
    
    # 両方とも木の部分が存在しない場合は完全一致とみなす
    if total_pred_non_empty == 0 and total_target_non_empty == 0:
        return torch.tensor(0.0, device=pred_continuous.device)
    
    # 片方だけ木の部分が存在しない場合は完全不一致
    if total_pred_non_empty == 0 or total_target_non_empty == 0:
        return torch.tensor(1.0, device=pred_continuous.device)
    
    total_weight = sum(class_weights)
    weighted_iou_sum = 0.0
    
    for cls_idx, cls_val in enumerate(class_values):
        # 予測と正解のマスク作成
        pred_mask = (pred_discrete == cls_val).float()
        target_mask = (target_discrete == cls_val).float()
        
        # IoUの計算
        intersection = (pred_mask * target_mask).sum(dim=(1,2,3,4))
        union = torch.clamp((pred_mask + target_mask), 0, 1).sum(dim=(1,2,3,4))
        
        # マスクが存在しない場合のチェック
        valid_mask = (union > 0)
        
        # 有効なサンプルのみでIoUを計算
        if valid_mask.sum() > 0:
            iou = (intersection[valid_mask] / (union[valid_mask] + 1e-6)).mean()
        else:
            # このクラスが両方に存在しない場合は、完全一致とみなす
            iou = torch.tensor(1.0, device=pred_continuous.device)
        
        # 重み付け
        weighted_iou_sum += iou * class_weights[cls_idx]
    
    # 重み付き平均IoU
    weighted_miou = weighted_iou_sum / total_weight
    
    # 損失として返す (1 - mIoU)
    return 1.0 - weighted_miou


def wandb_setup():
    wandb.login()
    epochs=1
    batchsize=16
    learning_rate=0.001
    device="cuda"
    config = {
        "epochs": epochs,
        "batch_size": batchsize,
        "learning_rate": learning_rate,
        "device": device,
        "architecture": "SVSAE",
        "dataset_size": None,  # Will be set in main()
        "loss_weights": {
            "mse": 1.0,
            "iou": 0.5,
            "chamfer": 0.5
        }
    }
    wandb.init(
        project="Tree-Autoencoder",
        # entity="ryuichi",
        name=f"svsae_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
    )





config={
    "epochs": 100,
    "batchsize": 16,
    "learning_rate": 0.001,
    "device": "cuda",
    "loss_function":{
        "mse": True,
        "iou": True,
        "chamfer": True,
    }
}

def main():
    epochs = 1
    batchsize = 16
    device="cuda"
    losses = []
    test_train_flag=False
    use_pretained_model=True
    model_path=r"/home/ryuichi/tree/TREE_PROJ/data_dir/model2025-02-25/model_0.pth"
    
    # Initialize wandb
    wandb_setup()
    
    svs_path="/mnt/nas/rmjapan2000/tree/data_dir/svd_0.2"
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    print("データセットの読み込みを開始します")
    model=SVSAE().to("cuda")
    

    if use_pretained_model:
        model.load_state_dict(torch.load(model_path), strict=False)
    
    Svsdataset=SvsDataset(svs_path)
    if test_train_flag:
        subset_list= [i for i in range(0,64)]
        Svsdataset=torch.utils.data.Subset(Svsdataset,subset_list)
    
    Svsdataloader=torch.utils.data.DataLoader(Svsdataset,batchsize,shuffle=True)
    learning_rate=0.001
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        eps=1e-08,
    )
    
    #     # Update wandb config with dataset size
    # wandb.config.update({"dataset_size": len(Svsdataset)})
    
    print(f"データセットのサイズ: {len(Svsdataset)}")
    for epoch in range(epochs):
        loss_sum = 0.0
        mse_loss_sum = 0.0
        iou_loss_sum = 0.0
        chamfer_loss_sum = 0.0
        cnt = 0
        print(f"{epoch}回目の訓練を開始しました")
        
        for x in Svsdataloader:
            x = x.to(device)
            recon_x=model(x)
            
            mse_loss=torch.nn.functional.mse_loss(recon_x,x)
            iou_loss=voxel_miou(recon_x,x)
            chamfer_loss=chamfer_distance(recon_x,x)
            loss=mse_loss+0.5*iou_loss+0.5*chamfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "batch_mse_loss": mse_loss.item(),
                "batch_iou_loss": iou_loss.item(),
                "batch_chamfer_loss": chamfer_loss.item(),
                "epoch": epoch
            })

            loss_sum += loss.item()
            mse_loss_sum += mse_loss.item()
            iou_loss_sum += iou_loss.item()
            chamfer_loss_sum += chamfer_loss
            
            if (cnt+1) %100 == 0:
                avg_loss = loss_sum/(cnt+1)
                avg_mse = mse_loss_sum/(cnt+1)
                avg_iou = iou_loss_sum/(cnt+1)
                avg_chamfer = chamfer_loss_sum/(cnt+1)
                
                print(f"\n=== Loss Metrics (avg last 100 steps) ===")
                print(f"Total Loss:      {avg_loss:.3f}")
                print(f"MSE Loss:        {avg_mse:.3f}")
                print(f"IoU Loss:        {avg_iou:.3f}") 
                print(f"Chamfer Loss:    {avg_chamfer:.3f}\n")
                
                # Log average metrics to wandb
                wandb.log({
                    "avg_loss_100": avg_loss,
                    "avg_mse_loss_100": avg_mse,
                    "avg_iou_loss_100": avg_iou,
                    "avg_chamfer_loss_100": avg_chamfer,
                    "epoch": epoch
                })
                
                #recon_xの分布を可視化
                recon_x = torch.where(
                    recon_x >= 0.8,
                    torch.tensor(1.0, device=recon_x.device),
                    torch.where(
                        recon_x >= 0.4,
                        torch.tensor(0.5, device=recon_x.device),
                        torch.where(
                            recon_x >= 0,
                            torch.tensor(0.0, device=recon_x.device),
                            torch.tensor(-1.0, device=recon_x.device)
                        )
                    )
                )
                voxel_distribution(recon_x[0].cpu().detach().numpy().squeeze())
   


                def run_visualization(data):
                    visualize_voxel_data(data)

                p1 = Process(target=run_visualization, args=(x[0].cpu().detach().numpy().squeeze(),))
                p2 = Process(target=run_visualization, args=(recon_x[0].cpu().detach().numpy().squeeze(),))
                
                p1.start()
                p2.start()
                
                start_time = time.time()
                while time.time() - start_time < 10:  # 10秒待機
                    if not (p1.is_alive() or p2.is_alive()):
                        break
                    time.sleep(0.1)
                
                p1.terminate()
                p2.terminate()
                loss_sum =0
                mse_loss_sum =0
                iou_loss_sum =0
                chamfer_loss_sum =0
                cnt=0
            cnt += 1
        losses.append(loss_sum / cnt)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]}")

        # モデルの保存
        current_date = datetime.now().strftime("%Y-%m-%d")
        path = f"/home/ryuichi/tree/TREE_PROJ/data_dir/model{current_date}"
        save_path = f"{path}/model_{epoch}.pth"
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), save_path)

        # ランダムに選んだデータの分布を表示
        random_index = random.randint(0, len(Svsdataset) - 1)
        print("元の分布")
        voxel_distribution(Svsdataset[random_index].numpy())
        # visualize_voxel_data(Svsdataset[random_index].numpy().squeeze())

        # VAEの出力を取得
        ae_out = model(Svsdataset[random_index].unsqueeze(0).to(device))
        ae_out = ae_out.cpu().detach().numpy()

        # 結果を分類
        result = np.full_like(ae_out, -1)
        mask_trunk = (ae_out >= 0.8) & (ae_out <= 1.0)
        mask_branch = (ae_out >=0.4) & (ae_out <0.8)
        mask_leaf = (ae_out >=0) & (ae_out <0.4)
        result[mask_trunk] = 1
        result[mask_branch] = 0.5
        result[mask_leaf] = 0

        print("result shape:", result.shape)
        print("resultの分布")
        voxel_distribution(result.squeeze())

    # モデルの保存
    save_train_result(num_data=len(Svsdataset), epochs=epochs, learning_rate=learning_rate, loss=losses)
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = f"/home/ryuichi/tree/TREE_PROJ/data_dir/model{current_date}"
    save_path = f"{path}/model_{epochs}.pth"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)

    # Close wandb run at the end
    wandb.finish()

if __name__ == "__main__":
    main()
    
