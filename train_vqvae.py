from email.policy import strict
import sys

from sklearn.model_selection import learning_curve
import wandb
import os 
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from model.svsae import SVSAE
import random
from schedulefree import RAdamScheduleFree
import torch
import numpy as np
import os
import torchvision
from datetime import datetime
import save_train_result
from svs import voxel_distribution
from svs import visualize_voxel_data
from save_train_result import save_train_result
from multiprocessing import Process
import time
from my_dataset.svsdataset import SvsDataset,SvsDataset_aug_trans,SvsDataset_aug_flip
from model.svsae import AutoencoderConfig
from torch.utils.data import ConcatDataset


def wandb_setup(device="cuda"):
    wandb.login()
    epochs=1
    batchsize=16
    learning_rate=0.001
    config = {
        "epochs": epochs,
        "batch_size": batchsize,
        "learning_rate": learning_rate,
        "device": device,
        "architecture": "SV-VQVAE",
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():
    epochs = 10
    batchsize = 4
    device="cuda"
    losses = []
    test_train_flag=False
    use_pretained_model=True

    # Initialize wandb with the detected device
    wandb_setup(device)
    
    svs_path="/mnt/nas/rmjapan2000/tree/data_dir/svd_0.2"
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    print("データセットの読み込みを開始します")
    config=AutoencoderConfig()
    config.encoder_type="ver2"
    config.decoder_type="ver1"
    # Update device in config
    print(f"Using device: {device}")
    config.device = device
    model=SVSAE(config).to(device)
    optimizer=RAdamScheduleFree(model.parameters(),lr=0.01,eps=1e-08)
    optimizer.train()

    if use_pretained_model:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    Svsdataset_normal=SvsDataset(svs_path)
    # Svsdataset_aug_trans=SvsDataset_aug_trans(svs_path)
    Svsdataset_aug_flip=SvsDataset_aug_flip(svs_path)
    # Svsdataset=ConcatDataset([Svsdataset_normal,Svsdataset_aug_flip,Svsdataset_aug_trans])
    Svsdataset=ConcatDataset([Svsdataset_normal,Svsdataset_aug_flip])
    if test_train_flag:
        subset_list= [i for i in range(0,64)]
        Svsdataset=torch.utils.data.Subset(Svsdataset,subset_list)
    
    Svsdataloader=torch.utils.data.DataLoader(Svsdataset,batchsize,shuffle=True)
  
    learning_rate=0.001 
    
    #     # Update wandb config with dataset size
    # wandb.config.update({"dataset_size": len(Svsdataset)})
    
    print(f"データセットのサイズ: {len(Svsdataset)}")
    for epoch in range(epochs):
        loss_sum = 0.0
        mae_loss_sum = 0.0
        iou_loss_sum = 0.0
        chamfer_loss_sum = 0.0
        weight_mae_loss_sum = 0.0
        weight_mae_loss_with_quantization_sum = 0.0
        cnt = 0
        print(f"{epoch}回目の訓練を開始しました")
        
        
        for x in Svsdataloader:
            x = x.to(device)
            recon_x=model(x)
            L1_Loss=torch.nn.L1Loss()
            mae_loss=L1_Loss(recon_x,x)
            iou_loss=voxel_miou(recon_x,x)
            # chamfer_loss=chamfer_distance(recon_x,x)
            chamfer_loss=0
            # weight_mae_loss=weighted_mae_loss(recon_x,x)
            weight_mae_loss=0
            weight_mae_loss_with_quantization=weighted_mae_loss_with_quantization(recon_x,x)
            
            loss=weight_mae_loss_with_quantization
            # loss=mae_loss+0.5*iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           

            # Log metrics to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "batch_mae_loss": mae_loss.item(),
                "batch_iou_loss": iou_loss.item(),
                "batch_chamfer_loss": chamfer_loss,
                "batch_weight_mae_loss": weight_mae_loss,
                "batch_weight_mae_loss_with_quantization": weight_mae_loss_with_quantization,
                "epoch": epoch
            })

            loss_sum += loss.item()
            mae_loss_sum += mae_loss.item()
            iou_loss_sum += iou_loss.item()
            chamfer_loss_sum += chamfer_loss
            weight_mae_loss_sum += weight_mae_loss
            weight_mae_loss_with_quantization_sum += weight_mae_loss_with_quantization

            if (cnt+1)%50 == 0:
                print(f"cnt:{cnt+1}")
                avg_loss = loss_sum/(cnt+1)
                avg_mae = mae_loss_sum/(cnt+1)
                avg_iou = iou_loss_sum/(cnt+1)
                avg_chamfer = chamfer_loss_sum/(cnt+1)
                avg_weight_mae = weight_mae_loss_sum/(cnt+1)
                avg_weight_mae_with_quantization = weight_mae_loss_with_quantization_sum/(cnt+1)

                print(f"\n=== Loss Metrics (avg last 100 steps) ===")
                print(f"Total Loss:      {avg_loss:.5f}")
                print(f"Weighted MAE Loss:      {avg_weight_mae:.5f}")
                print(f"MAE Loss:        {avg_mae:.5f}")
                print(f"IoU Loss:        {avg_iou:.5f}") 
                print(f"Chamfer Loss:    {avg_chamfer:.5f}\n")
                print(f"Weighted MAE Loss with Quantization:      {avg_weight_mae_with_quantization:.5f}")
                
                # Log average metrics to wandb
                wandb.log({
                    "avg_loss_100": avg_loss,
                    "avg_mae_loss_100": avg_mae,
                    "avg_iou_loss_100": avg_iou,
                    "avg_chamfer_loss_100": avg_chamfer,
                    "avg_weight_mae_loss_100": avg_weight_mae,
                    "avg_weight_mae_loss_with_quantization_100": avg_weight_mae_with_quantization,
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
                mae_loss_sum =0
                iou_loss_sum =0
                chamfer_loss_sum =0
                weight_mae_loss_sum =0
                weight_mae_loss_with_quantization_sum =0
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
    print("結果の保存")
    
    save_train_result(num_data=len(Svsdataset), epochs=epochs, learning_rate=learning_rate, loss=losses)
    current_date = datetime.now().strftime("%Y-%m-%d")
    path = f"/home/ryuichi/tree/TREE_PROJ/data_dir/model{current_date}"
    save_path = f"{path}/model_{epochs}.pth"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)

    # Close wandb run at the end
    wandb.finish()

    # CUDAが認識されているか確認
    print(f"CUDA available: {torch.cuda.is_available()}")
    # 認識されているGPUの数
    print(f"GPU count: {torch.cuda.device_count()}")
    # 現在のGPUメモリ使用状況を確認
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024**2:.2f}MB / {torch.cuda.memory_reserved(i) / 1024**2:.2f}MB")

if __name__ == "__main__":
    main()
    
