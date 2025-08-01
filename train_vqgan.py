import sys
import os
sys.path.append("/home/ryuichi/tree/TREE_PROJ")

from model.svsae import AutoencoderConfig, SVSEncoder_ver2, SVSDecoder_ver2, VectorQuantizer, Discriminator, VQGANLoss
from my_dataset.svsdataset import SvsDataLoader
from lightning import LightningModule, Trainer
import torch.nn.functional as F
from schedulefree import RAdamScheduleFree
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger


class VQGAN(LightningModule):
    def __init__(self, model_config, beta=0.25, gan_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.encoder = SVSEncoder_ver2(model_config)
        self.decoder = SVSDecoder_ver2(model_config)
        self.vq = VectorQuantizer(model_config, beta=beta)
        self.discriminator = Discriminator(model_config)
        
        # Loss function
        self.vqgan_loss = VQGANLoss(model_config, perceptual_weight=perceptual_weight, gan_weight=gan_weight)
        
        # Automatic optimization を無効化（GeneratorとDiscriminatorを個別に最適化するため）
        self.automatic_optimization = False
        
        # GAN training balance parameters
        self.d_reg_every = 16  # Discriminator regularization frequency
        self.g_steps = 1  # Generator steps per iteration
        self.d_steps = 1  # Discriminator steps per iteration
        self.training_step_count = 0
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Vector Quantization
        vq_loss, quantized, encodings = self.vq(encoded)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        return reconstructed, vq_loss, quantized, encodings
    
    def training_step(self, batch, batch_idx):
        x = batch
        opt_g, opt_d = self.optimizers()
        self.training_step_count += 1
        
        # ==================== Train Generator ====================
        # Forward pass
        reconstructed, vq_loss, quantized, encodings = self(x)
        
        # Discriminator predictions on fake data
        disc_fake = self.discriminator(reconstructed)
        
        # Generator loss
        g_loss_dict = self.vqgan_loss.total_generator_loss(
            x, reconstructed, vq_loss, disc_fake
        )
        
        # Train generator every step
        opt_g.zero_grad()
        self.manual_backward(g_loss_dict['total_loss'])
        self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_g.step()
        
        # ==================== Train Discriminator ====================
        # Train discriminator less frequently or with lower learning rate
        d_loss = torch.tensor(0.0, device=self.device)
        
        # Adaptive discriminator training based on loss ratio
        g_loss_val = g_loss_dict['generator_loss'].item()
        
        # Train discriminator only if generator is doing well enough
        if g_loss_val > 0.5 or self.training_step_count % 2 == 0:
            # Discriminator predictions
            disc_real = self.discriminator(x)
            disc_fake_detached = self.discriminator(reconstructed.detach())
            
            # Discriminator loss with label smoothing
            d_loss = self.vqgan_loss.discriminator_loss_with_smoothing(disc_real, disc_fake_detached)
            
            # Optimize discriminator
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")  # Lower gradient clipping for D
            opt_d.step()
        
        # R1 regularization for discriminator every d_reg_every steps
        if self.training_step_count % self.d_reg_every == 0:
            x.requires_grad_(True)
            disc_real_reg = self.discriminator(x)
            r1_loss = self.compute_r1_penalty(disc_real_reg, x)
            
            opt_d.zero_grad()
            self.manual_backward(r1_loss * 10.0)  # R1 weight
            opt_d.step()
            
            x.requires_grad_(False)
            self.log('train_r1_loss', r1_loss, on_step=True, logger=True, sync_dist=True)
        
        # Logging
        self.log('train_g_total_loss', g_loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_g_recon_loss', g_loss_dict['reconstruction_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_g_vq_loss', g_loss_dict['vq_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_g_gan_loss', g_loss_dict['generator_loss'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return g_loss_dict['total_loss']
    
    def compute_r1_penalty(self, real_pred, real_img):
        """Compute R1 regularization penalty"""
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )[0]
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        # Forward pass
        reconstructed, vq_loss, quantized, encodings = self(x)
        
        # Discriminator predictions
        disc_real = self.discriminator(x)
        disc_fake = self.discriminator(reconstructed)
        
        # Losses
        g_loss_dict = self.vqgan_loss.total_generator_loss(
            x, reconstructed, vq_loss, disc_fake
        )
        d_loss = self.vqgan_loss.discriminator_loss(disc_real, disc_fake)
        
        # Logging
        self.log('val_g_total_loss', g_loss_dict['total_loss'], on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('val_g_recon_loss', g_loss_dict['reconstruction_loss'], on_epoch=True, logger=True, sync_dist=True)
        self.log('val_g_vq_loss', g_loss_dict['vq_loss'], on_epoch=True, logger=True, sync_dist=True)
        self.log('val_g_gan_loss', g_loss_dict['generator_loss'], on_epoch=True, logger=True, sync_dist=True)
        self.log('val_d_loss', d_loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        
        return g_loss_dict['total_loss']
    
    def configure_optimizers(self):
        # Generator optimizer (Encoder + Decoder + VQ) - Higher learning rate
        g_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.vq.parameters())
        optimizer_g = RAdamScheduleFree(g_params, lr=0.0002, eps=1e-08)  # Increased learning rate
        
        # Discriminator optimizer - Lower learning rate to prevent overfitting
        optimizer_d = RAdamScheduleFree(self.discriminator.parameters(), lr=0.00005, eps=1e-08)  # Decreased learning rate
        
        # Set to train mode
        optimizer_g.train()
        optimizer_d.train()
        
        return [optimizer_g, optimizer_d]


def main():
    import torch
    import sys
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Data loader
    dataloader = SvsDataLoader(batch_size=8, sub_dataset=True,sub_num=32)  # Smaller batch size for VQGAN
    
    # Model config
    config = AutoencoderConfig(
        latent_channels=1,
        encoder_type="ver2",
        decoder_type="ver2",
        device='cuda'
    )
    
    # Model
    model = VQGAN(
        model_config=config,
        beta=0.25,
        gan_weight=1.0,
        perceptual_weight=1.0
    )
    
    # Checkpoint callback
    os.makedirs("/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqgan", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqgan",
        filename="{epoch:02d}-{train_g_total_loss:.4f}",
        save_top_k=5,
        monitor="train_g_total_loss",
        mode="min",
        save_weights_only=True
    )
    
    # Logger
    logger = WandbLogger(project="tree_svs_vqgan", name="vqgan_experiment")
    logger.watch(model, log="all", log_graph=True, log_freq=100)
    
    # Trainer
    trainer = Trainer(
        accelerator="cuda",
        devices="auto",
        max_epochs=100,
        strategy="ddp",  # find_unused_parameters=Trueを削除
        callbacks=[checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=5,  # Validate every 5 epochs
        precision="16-mixed",  # 混合精度学習でメモリ効率とパフォーマンスを改善
    )
    
    # Train
    trainer.fit(model, datamodule=dataloader)
    print("VQGAN training completed!")


if __name__ == "__main__":
    main()
