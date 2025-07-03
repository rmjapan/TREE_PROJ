
from model.svsae import AutoencoderConfig,SVSDecoder_ver2,SVSEncoder_ver2,VectorQuantizer
from my_dataset.svsdataset import SvsDataLoader
from train_vqvae import VQVAE
from visualize_func import visualize_with_timeout4voxel
import torch

config=AutoencoderConfig()
model = VQVAE(config)

ckpt = "/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqvae/epoch=03-train_total_loss=0.0030.ckpt"
# ckpt = "/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqvae/epoch=00-train_total_loss=0.0252.ckpt"
model.load_state_dict(torch.load(ckpt, map_location=torch.device("cuda"))["state_dict"])
dataload=SvsDataLoader(batch_size=1,sub_dataset=False)
dataload.setup()
train_data=dataload.train_dataloader()
count=0

for x in train_data:
    if count >20:
        break
    count=count+1
    recon_x=model(x)[0]
    #0.8以上は1.0,0.4以上は0.5,0.4未満は0.0,それ以外は-1.0
    recon_x=torch.where(
                recon_x>0.8,1.0,
                torch.where(
                    recon_x>0.4,0.5,
                    torch.where(
                        recon_x>0.0,0.0,-1.0
                    )
                )
            )
    print(recon_x.shape)
    recon_x=recon_x.squeeze(0).detach().cpu().numpy()
    print(recon_x.shape)
    visualize_with_timeout4voxel(recon_x,timeout=15,title=f"recon_x_{count:02d}")
    print(x.shape)
    x=x.squeeze(0).detach().cpu().numpy()
    visualize_with_timeout4voxel(x[0],timeout=15,title=f"x_{count:02d}")

#     
    
