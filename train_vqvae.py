

from model.svsae import AutoencoderConfig,SVSEncoder_ver2,SVSDecoder_ver2,VectorQuantizer
from my_dataset.svsdataset import SvsDataLoader
from lightning import LightningModule, Trainer
import torch.nn.functional as F
from schedulefree import RAdamScheduleFree



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
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True)
        return total_loss
    def configure_optimizers(self):
        # RAdamのスケジュールフリー版を使用
        optimizer=RAdamScheduleFree(self.parameters(),lr=0.01,eps=1e-08)
        optimizer.train()
        return optimizer

        
        
def main():
    dataloader=SvsDataLoader(batch_size=8)
    config=AutoencoderConfig()
    config.device='cuda'
    model=VQVAE(config)
    trainer= Trainer(max_epochs=10)
    trainer.fit(model,datamodule=dataloader)
    print("train_vqvae.py")
if __name__ == "__main__":
    main()
    