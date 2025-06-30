

from model.svsae import AutoencoderConfig,SVSEncoder_ver2,SVSDecoder_ver2,VectorQuantizer
from my_dataset.svsdataset import SvsDataLoader
from lightning import LightningModule, Trainer
import torch.nn.functional as F
from schedulefree import RAdamScheduleFree
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from lightning.pytorch.loggers import WandbLogger


class VQVAE(LightningModule):
    def __init__(self,model_config):
        super().__init__()
        self.encode=SVSEncoder_ver2(model_config).to(model_config.device)
        self.decode=SVSDecoder_ver2(model_config).to(model_config.device)
        self.vq=VectorQuantizer(model_config,beta=0.25).to(model_config.device)
        
    def training_step(self,batch,batch_idx):
        #foward プロセス
        x=batch
        latent_feature=self.encode(x)
        vq_loss,vq_output,emb=self.vq(latent_feature)
        x_recon=self.decode(vq_output)
        
        #損失計算
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss   
        
        # ログ記録（個別の損失も記録すると分析しやすい）
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True,logger=True)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True,logger=True)
        return total_loss
    def validation_step(self,batch,batch_idx):
        x=batch
        latent_feature=self.encode(x)
        vq_loss,vq_output,emb=self.vq(latent_feature)
        x_recon=self.decode(vq_output)
        recon_loss=F.mse_loss(x_recon,x)
        self.log("val_total_loss",vq_loss+recon_loss,on_epoch=True,logger=True)
        self.log("val_recon_loss",recon_loss,on_epoch=True,logger=True)
        self.log("val_vq_loss",vq_loss,on_epoch=True,logger=True)
        return vq_loss+recon_loss
    def configure_optimizers(self):
        # RAdamのスケジュールフリー版を使用
        optimizer=RAdamScheduleFree(self.parameters(),lr=0.01,eps=1e-08)
        optimizer.train()
        return optimizer
    def forward(self,x):
        latent_feature=self.encode(x)
        vq_loss,vq_output,emb=self.vq(latent_feature)
        x_recon=self.decode(vq_output)
        return x_recon

        
        
def main():
    import torch
    import sys
    print(torch.__version__)
    print(torch.version.cuda)
    print(sys.version)
    dataloader=SvsDataLoader(batch_size=8,sub_dataset=False)
    config=AutoencoderConfig()
    config.device='cuda'
    model=VQVAE(config)

    os.makedirs("/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqvae",exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqvae",
        filename="{epoch:02d}-{train_total_loss:.4f}",  # ファイル名テンプレート
        save_top_k=5,  # ベスト3モデルだけ残す
        monitor="train_total_loss",  # 監視する指標
        mode="min",  # lossが小さい方が良い
        save_weights_only=True  # モデル全体を保存（Trueなら重みだけ）
    )
    logger = WandbLogger(project="tree_svs_vqvae")
    logger.watch(model,log="all",log_graph=True,log_freq=100)
    trainer= Trainer(
        accelerator="cuda",
        devices="auto",
        max_epochs=100,
        strategy="ddp",
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    trainer.fit(model,datamodule=dataloader)
    print("train_vqvae.py")
if __name__ == "__main__":
    main()
    