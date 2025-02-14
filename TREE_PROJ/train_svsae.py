import sys


sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from model.svsae import SVSAE
from utils import npz2dense, dense2sparse,weighted_mse_loss

import random

import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from diffusers import AutoencoderKL
import os
from scipy.sparse import csr_matrix
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from datetime import datetime
import save_train_result
from svs import voxel_distribution
from svs import visualize_voxel_data
from save_train_result import save_train_result

def chamfer_distance(p1, p2):
    B, N, _ = p1.shape
    _, M, _ = p2.shape
    p1_sq = p1.pow(2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    p2_sq = p2.pow(2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, M)
    dist = p1_sq + p2_sq - 2 * torch.bmm(p1, p2.transpose(1, 2))  # (B, N, M)
    d1, _ = dist.min(dim=2)
    d2, _ = dist.min(dim=1)
    return d1.mean() + d2.mean()
def voxel_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3, 4))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2, 3, 4))
    iou = intersection / (union + 1e-6)
    return 1 - iou.mean()
class SvsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir
        self.transform=transform
    def __len__(self):
        return len(os.listdir(self.data_dir))
    def __getitem__(self,idx):
        svs=np.load(self.data_dir+f"/svs_{idx+1}.npz")
        # print(f"svs_{idx+1}.npzを読み込みました")
        svs=npz2dense(svs)
        svs[svs==0]=-1.0
        svs[svs==1]=0.0
        svs[svs==1.5]=0.5
        svs[svs==2]=1.0
        
        svs=torch.tensor(svs).float()
        svs=torch.unsqueeze(svs,0)
        return svs
    

def wandb_setup():
    wandb.login()
    wandb.init(
        project="Tree-Autoencoder",
        entity="ryuichi",
        name="svsae",
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
    model_path=r"/home/ryuichi/tree/TREE_PROJ/data_dir/model2025-01-26/model_1_last.pth"
    

    
    
    svs_path="/home/ryuichi/tree/TREE_PROJ/data_dir/svd_0.2"
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    print("データセットの読み込みを開始します")
    model=SVSAE().to("cuda")
    if use_pretained_model:
        model.load_state_dict(torch.load(model_path))
        # model.from_pretrained(model_path)
    Svsdataset=SvsDataset(svs_path)
    if test_train_flag:
        subset_list= [i for i in range(0,64)]
        Svsdataset=torch.utils.data.Subset(Svsdataset,subset_list)
    
    Svsdataloader=torch.utils.data.DataLoader(Svsdataset,batchsize,shuffle=True)
    learning_rate=0.001
    optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,         # 3D構造の学習には小さめの学習率が安定
    eps=1e-08,            # デフォルト値で問題なし
)
    print(f"データセットのサイズ: {len(Svsdataset)}")
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        print(f"{epoch}回目の訓練を開始しました")
        
        for x in Svsdataloader:
            x = x.to(device)
            recon_x=model(x)
            
            #再構成誤差（MSE)
            leaf_mask = (recon_x >= 0) & (recon_x < 0.4)
            branch_mask = (recon_x >= 0.4) & (recon_x < 0.8)
            trunk_mask = (recon_x >= 0.8) & (recon_x <= 1.0)
            blank_mask = (recon_x<0)
        
            
            mse_leaf = torch.nn.functional.mse_loss(recon_x[leaf_mask], x[leaf_mask])
            mse_branch = torch.nn.functional.mse_loss(recon_x[branch_mask], x[branch_mask])
            mse_trunk = torch.nn.functional.mse_loss(recon_x[trunk_mask], x[trunk_mask])
            mse_blank = torch.nn.functional.mse_loss(recon_x[blank_mask], x[blank_mask]) 
            
            len_leaf=len(recon_x[leaf_mask])
            len_branch=len(recon_x[branch_mask])
            len_trunk=len(recon_x[trunk_mask])
            len_blank=len(recon_x[blank_mask])

            leaf_weight=1.0*(len_leaf/(256*256*256*16))
            branch_weight=1.0*(len_branch/(256*256*256*16))
            trunk_weight=1.0*(len_trunk/(256*256*256*16))
            blank_weight=1.0*(len_blank/(256*256*256*16))
            
            mse_loss = leaf_weight * mse_leaf + branch_weight * mse_branch + trunk_weight * mse_trunk + blank_weight * mse_blank
            
            iou_loss=voxel_iou(recon_x,x)
            chamfer_loss=chamfer_distance(recon_x,x)

           
            
        
            # mse_loss = torch.nn.functional.mse_loss(recon_x, x)
            # weight_loss=weighted_mse_loss(x,recon_x )
            loss=mse_loss+0.1*iou_loss+0.1*chamfer_loss
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if cnt %10 == 0:
                print(f"cnt:{cnt}で、loss:{loss}")
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
                print("recon_xの分布")
                voxel_distribution(recon_x[0].cpu().detach().numpy().squeeze())
                print("Loss:", loss.item())
                # visualize_voxel_data(x[0].cpu().detach().numpy().squeeze())
                # visualize_voxel_data(recon_x[0].cpu().detach().numpy().squeeze())
            cnt += 1
            # print(f"cnt:{cnt}で、loss:{loss}")

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

        # # ボクセルデータの可視化
        # visualize_voxel_data(result.squeeze())
    # モデルの保存
    
    save_train_result(num_data=len(Svsdataset), epochs=epochs, learning_rate=learning_rate, loss=losses)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = f"/home/ryuichi/tree/TREE_PROJ/data_dir/model{current_date}"
    save_path = f"{path}/model_{epochs}.pth"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    # torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
    
