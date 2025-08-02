import sys
import os
import math
sys.path.append("/home/ryuichi/tree/TREE_PROJ")

from model.svsae import AutoencoderConfig, SVSEncoder_ver2, SVSDecoder_ver2, VectorQuantizer, Discriminator, VQGANLoss
from my_dataset.svsdataset import SvsDataLoader
from lightning import LightningModule, Trainer
import torch.nn.functional as F
from schedulefree import RAdamScheduleFree
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class VQGAN(LightningModule):
    """
    改良されたVector Quantized Generative Adversarial Network (VQGAN)実装
    再構成品質と学習安定性向上のための高度な学習技術を搭載
    """
    
    def __init__(self, model_config, beta=0.25, gan_weight=0.5, perceptual_weight=1.0, 
                 learning_rate=1e-4, discriminator_lr_ratio=0.5, use_manual_optimization=False):
        super().__init__()
        self.save_hyperparameters()
        
        # モデルコンポーネント
        self.encoder = SVSEncoder_ver2(model_config)
        self.decoder = SVSDecoder_ver2(model_config)
        self.vq = VectorQuantizer(model_config, beta=beta)
        self.discriminator = Discriminator(model_config)
        
        # 損失関数
        self.vqgan_loss = VQGANLoss(
            model_config, 
            perceptual_weight=perceptual_weight, 
            gan_weight=gan_weight
        )
        
        # 最適化戦略の選択
        self.automatic_optimization = not use_manual_optimization
        
        # Lightning frameworkに手動最適化を通知
        if use_manual_optimization:
            self.automatic_optimization = False
        
        # 学習設定の初期化
        self._setup_training_parameters()
        
        # メトリクス追跡用
        self.train_step_outputs = []
        self.val_step_outputs = []
        
    def _setup_training_parameters(self):
        """自動最適化と手動最適化の両方に対応した改良学習パラメータの初期化"""
        # フェーズベースの学習アプローチ
        self.warmup_epochs = 10        # エンコーダー・デコーダーのウォームアップ
        self.vq_warmup_epochs = 20     # VQ層の追加ウォームアップ
        self.gan_start_epoch = 30      # GAN損失の開始
        
        # 手動最適化の互換性のためのレガシーパラメータ
        self.warmup_steps = 1000
        self.discriminator_start_step = 500
        self.d_reg_every = 16
        self.training_step_count = 0
        
        # 適応的重み付け
        self.recon_weight = 1.0
        self.vq_weight_start = 0.1
        self.vq_weight_max = 0.5
        self.gan_weight_max = self.hparams.gan_weight
        
        # 学習安定性
        self.gradient_clip_val = 1.0
        self.discriminator_steps = 1
        self.generator_steps = 1
        
        # 手動勾配蓄積
        self.accumulate_grad_batches = 2
        self.grad_accum_count = 0
        
        # 品質監視
        self.best_recon_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 5
    
    def configure_optimizers(self):
        """自動最適化と手動最適化の両方をサポートするオプティマイザ設定"""
        if self.automatic_optimization:
            # 自動最適化では単一オプティマイザを使用
            # 全パラメータ（エンコーダー + デコーダー + VQ + 識別器）
            all_params = (
                list(self.encoder.parameters()) + 
                list(self.decoder.parameters()) + 
                list(self.vq.parameters()) +
                list(self.discriminator.parameters())
            )
            optimizer = torch.optim.AdamW(
                all_params, 
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.99),
                weight_decay=1e-4,
                eps=1e-8
            )
            
            # 学習率スケジューラ
            scheduler = {
                'scheduler': CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6),
                'monitor': 'val_recon_loss',
                'frequency': 1,
                'name': 'unified_lr'
            }
            
            return [optimizer], [scheduler]
        else:
            # 手動最適化では複数オプティマイザを使用（レガシーモード）
            # ジェネレータオプティマイザ
            g_params = (
                list(self.encoder.parameters()) + 
                list(self.decoder.parameters()) + 
                list(self.vq.parameters())
            )
            optimizer_g = RAdamScheduleFree(
                g_params, 
                lr=5e-5,  # 保守的な学習率
                eps=1e-08,
                weight_decay=1e-4
            )
            
            # 識別器オプティマイザ
            optimizer_d = RAdamScheduleFree(
                self.discriminator.parameters(), 
                lr=1e-5,  # 識別器はさらに保守的
                eps=1e-08,
                weight_decay=1e-4
            )
            
            # 学習モードに設定
            optimizer_g.train()
            optimizer_d.train()
            
            return [optimizer_g, optimizer_d]
    
    def get_current_weights(self):
        """学習進行に基づく適応的重み付けの取得"""
        current_epoch = self.current_epoch
        
        # VQ重みのスケジューリング
        if current_epoch < self.warmup_epochs:
            vq_weight = 0.0  # 初期ウォームアップ中はVQ損失なし
        elif current_epoch < self.vq_warmup_epochs:
            # VQウォームアップ中は線形増加
            progress = (current_epoch - self.warmup_epochs) / (self.vq_warmup_epochs - self.warmup_epochs)
            vq_weight = self.vq_weight_start + progress * (self.vq_weight_max - self.vq_weight_start)
        else:
            vq_weight = self.vq_weight_max
            
        # GAN重みのスケジューリング
        if current_epoch < self.gan_start_epoch:
            gan_weight = 0.0  # 指定エポック前はGAN損失なし
        else:
            # GAN重みの段階的増加
            progress = min(1.0, (current_epoch - self.gan_start_epoch) / 20)
            gan_weight = progress * self.gan_weight_max
            
        return {
            'recon_weight': self.recon_weight,
            'vq_weight': vq_weight,
            'gan_weight': gan_weight
        }
    
    def forward(self, x):
        """モデルの前向き計算"""
        # エンコード
        encoded = self.encoder(x)
        
        # ベクトル量子化
        vq_loss, quantized, encodings = self.vq(encoded)
        
        # デコード
        reconstructed = self.decoder(quantized)
        
        return reconstructed, vq_loss, quantized, encodings
    
    def compute_losses(self, x, reconstructed, vq_loss, disc_fake=None, disc_real=None):
        """現在の適応的重み付けによる全損失の計算"""
        weights = self.get_current_weights()
        
        # 再構成損失
        recon_loss = self.vqgan_loss.reconstruction_loss(x, reconstructed)
        
        # VQ損失
        vq_loss_weighted = weights['vq_weight'] * vq_loss
        
        # ジェネレータ損失
        if disc_fake is not None and weights['gan_weight'] > 0:
            gen_loss = self.vqgan_loss.generator_loss(disc_fake)
            gen_loss_weighted = weights['gan_weight'] * gen_loss
        else:
            gen_loss = torch.tensor(0.0, device=self.device)
            gen_loss_weighted = torch.tensor(0.0, device=self.device)
        
        # 総ジェネレータ損失
        g_total_loss = (
            weights['recon_weight'] * recon_loss + 
            vq_loss_weighted + 
            gen_loss_weighted
        )
        
        # 識別器損失
        if disc_real is not None and disc_fake is not None and weights['gan_weight'] > 0:
            d_loss = self.vqgan_loss.discriminator_loss_with_smoothing(
                disc_real, disc_fake.detach()
            )
        else:
            d_loss = torch.tensor(0.0, device=self.device)
        
        return {
            'g_total_loss': g_total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'gen_loss': gen_loss,
            'd_loss': d_loss,
            'weights': weights
        }
    
    def training_step(self, batch, batch_idx):
        """統一された学習ステップ（自動最適化と手動最適化の両方をサポート）"""
        x = batch
        
        if self.automatic_optimization:
            # 自動最適化モード（単一オプティマイザ）
            return self._automatic_training_step(x, batch_idx)
        else:
            # 手動最適化モード（複数オプティマイザ）
            return self._manual_training_step(x, batch_idx)
    
    def _automatic_training_step(self, x, batch_idx):
        """自動最適化による学習ステップ（単一オプティマイザ使用）"""
        # 前向き計算
        reconstructed, vq_loss, quantized, encodings = self(x)
        
        # NaNチェック
        if torch.isnan(vq_loss) or torch.isinf(vq_loss):
            print(f"警告: ステップ {self.global_step} でVQ損失にNaN/Infが検出されました")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 現在の学習重みを取得
        weights = self.get_current_weights()
        
        # 識別器の予測（GANが有効な場合のみ）
        if weights['gan_weight'] > 0:
            disc_fake = self.discriminator(reconstructed)
            disc_real = self.discriminator(x)
        else:
            disc_fake = None
            disc_real = None
        
        # 損失計算
        losses = self.compute_losses(x, reconstructed, vq_loss, disc_fake=disc_fake, disc_real=disc_real)
        
        # NaNチェック（損失後）
        if torch.isnan(losses['recon_loss']) or torch.isinf(losses['recon_loss']):
            print(f"警告: ステップ {self.global_step} で再構成損失にNaN/Infが検出されました")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 統合損失（ジェネレータと識別器の両方）
        total_loss = losses['g_total_loss'] + 0.1 * losses['d_loss'] if weights['gan_weight'] > 0 else losses['g_total_loss']
        
        # 最終NaNチェック
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"警告: ステップ {self.global_step} で総損失にNaN/Infが検出されました")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ログ出力
        self.log('train_total_loss', total_loss, 
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_recon_loss', losses['recon_loss'], 
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_vq_loss', losses['vq_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_gen_loss', losses['gen_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_d_loss', losses['d_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        
        # 現在の重みをログ
        self.log('weight_recon', weights['recon_weight'], on_step=True, sync_dist=True)
        self.log('weight_vq', weights['vq_weight'], on_step=True, sync_dist=True)
        self.log('weight_gan', weights['gan_weight'], on_step=True, sync_dist=True)
        
        return total_loss
    
    def _manual_training_step(self, x, batch_idx):
        """手動最適化による学習ステップ（レガシーモード）"""
        opt_g, opt_d = self.optimizers()
        self.training_step_count += 1
        self.grad_accum_count += 1
        
        # 前向き計算
        reconstructed, vq_loss, quantized, encodings = self(x)
        
        # 学習進行に基づく適応的損失重み
        current_step = self.global_step
        
        # ウォームアップフェーズ: 再構成とVQ損失に集中
        if current_step < self.warmup_steps:
            gan_weight_current = 0.0
        else:
            gan_weight_current = self.hparams.gan_weight
        
        # シンプルなアプローチによるジェネレータ損失
        g_recon_loss = self.vqgan_loss.reconstruction_loss(x, reconstructed)
        
        if current_step >= self.discriminator_start_step:
            disc_fake = self.discriminator(reconstructed)
            g_gan_loss = self.vqgan_loss.generator_loss(disc_fake) if gan_weight_current > 0 else torch.tensor(0.0, device=self.device)
        else:
            g_gan_loss = torch.tensor(0.0, device=self.device)
        
        g_total_loss = g_recon_loss + 0.25 * vq_loss + gan_weight_current * g_gan_loss
        
        # NaNチェック
        if torch.isnan(g_total_loss) or torch.isinf(g_total_loss):
            print(f"警告: ステップ {current_step} でジェネレータ損失にNaNが検出されました")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 手動最適化ステップ（簡素化）
        g_total_loss = g_total_loss / self.accumulate_grad_batches
        self.manual_backward(g_total_loss)
        
        if self.grad_accum_count % self.accumulate_grad_batches == 0:
            self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_g.step()
            opt_g.zero_grad()
        
        # 識別器の学習（簡素化）
        if current_step >= self.discriminator_start_step and gan_weight_current > 0:
            disc_real = self.discriminator(x.detach())
            disc_fake_detached = self.discriminator(reconstructed.detach())
            d_loss = self.vqgan_loss.discriminator_loss_with_smoothing(disc_real, disc_fake_detached)
            
            if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                d_loss_scaled = d_loss / self.accumulate_grad_batches
                self.manual_backward(d_loss_scaled)
                
                if self.grad_accum_count % self.accumulate_grad_batches == 0:
                    self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                    opt_d.step()
                    opt_d.zero_grad()
        else:
            d_loss = torch.tensor(0.0, device=self.device)
        
        # ログ出力
        self.log('train_g_total_loss', g_total_loss * self.accumulate_grad_batches, 
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_recon_loss', g_recon_loss, 
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_gen_loss', g_gan_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_d_loss', d_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return g_total_loss * self.accumulate_grad_batches
    
    def validation_step(self, batch, batch_idx):
        """包括的メトリクスによる統一検証ステップ"""
        x = batch
        
        # 前向き計算
        reconstructed, vq_loss, quantized, encodings = self(x)
        
        # 現在の重みを取得
        weights = self.get_current_weights()
        
        # 識別器の予測
        if weights['gan_weight'] > 0:
            disc_real = self.discriminator(x)
            disc_fake = self.discriminator(reconstructed)
        else:
            disc_real = None
            disc_fake = None
        
        # 全損失の計算
        losses = self.compute_losses(
            x, reconstructed, vq_loss, 
            disc_fake=disc_fake, disc_real=disc_real
        )
        
        # 追加の品質メトリクス
        l1_loss = F.l1_loss(reconstructed, x)
        mse_loss = F.mse_loss(reconstructed, x)
        
        # 知覚的類似度
        cosine_sim = F.cosine_similarity(
            reconstructed.flatten(1), x.flatten(1), dim=1
        ).mean()
        
        # コードブック使用率
        unique_codes = torch.unique(torch.argmax(encodings, dim=1)).numel()
        codebook_usage = unique_codes / self.vq.K
        
        # ログ出力
        self.log('val_g_total_loss', losses['g_total_loss'], 
                on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', losses['recon_loss'], 
                on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_vq_loss', losses['vq_loss'], on_epoch=True, sync_dist=True)
        self.log('val_gen_loss', losses['gen_loss'], on_epoch=True, sync_dist=True)
        self.log('val_d_loss', losses['d_loss'], on_epoch=True, sync_dist=True)
        
        # 品質メトリクス
        self.log('val_l1_loss', l1_loss, on_epoch=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True, sync_dist=True)
        self.log('val_cosine_sim', cosine_sim, on_epoch=True, sync_dist=True)
        self.log('val_codebook_usage', codebook_usage, on_epoch=True, sync_dist=True)
        
        return losses['g_total_loss']
    
    def on_validation_epoch_end(self):
        """早期停止のための検証メトリクス監視"""
        current_recon_loss = self.trainer.callback_metrics.get('val_recon_loss', float('inf'))
        
        if current_recon_loss < self.best_recon_loss:
            self.best_recon_loss = current_recon_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        self.log('best_recon_loss', self.best_recon_loss, sync_dist=True)
        self.log('patience_counter', self.patience_counter, sync_dist=True)


def main():
    """自動最適化と手動最適化の両方をサポートする改良されたメイン学習関数"""
    import torch
    import sys
    
    # 最適化設定
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"CUDA バージョン: {torch.version.cuda}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    print(f"Float32 matmul 精度: {torch.get_float32_matmul_precision()}")
    
    # 最適化されたデータローダー設定
    dataloader = SvsDataLoader(
        batch_size=2,  # さらに小さなバッチサイズで安定性向上
        data_dir="/mnt/nas/rmjapan2000/tree/data_dir/train/svs_LDM_v1",
        sub_dataset=False
    )
    
    # モデル設定
    config = AutoencoderConfig(
        latent_channels=1,
        encoder_type="ver2",
        decoder_type="ver2",
        device='cuda'
    )
    
    # 学習モードの選択: True=自動最適化（改良版）、False=手動最適化（レガシー版）
    use_automatic_optimization = True
    
    if use_automatic_optimization:
        print("自動最適化による改良学習戦略を使用")
        # 自動最適化用の最適化されたハイパーパラメータ
        model = VQGAN(
            model_config=config,
            beta=0.05,  # より保守的なVQ commitment loss
            gan_weight=0.05,  # より保守的なGAN重み
            perceptual_weight=0.3,  # より保守的な知覚重み
            learning_rate=5e-5,  # より低い学習率
            discriminator_lr_ratio=0.5,  # 識別器はより遅く学習
            use_manual_optimization=False
        )
        
        # 自動最適化用のコールバック
        os.makedirs("/mnt/nas/rmjapan2000/tree/data_dir/train/model_improved_vqgan", exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath="/mnt/nas/rmjapan2000/tree/data_dir/train/model_improved_vqgan",
            filename="{epoch:02d}-{val_recon_loss:.4f}",
            save_top_k=3,
            monitor="val_recon_loss",
            mode="min",
            save_weights_only=True,
            every_n_epochs=1
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        early_stopping = EarlyStopping(
            monitor='val_recon_loss',
            patience=10,
            mode='min',
            min_delta=0.001
        )
        
        # ロガー
        logger = WandbLogger(
            project="tree_svs_vqgan", 
            name="improved_vqgan_automatic_training",
            log_model=True
        )
        
        # 自動最適化用の最適化されたTrainer設定
        trainer = Trainer(
            accelerator="cuda",
            devices="auto",
            max_epochs=100,
            strategy=DDPStrategy(find_unused_parameters=True),  # 未使用パラメータ検出を有効化
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            logger=logger,
            check_val_every_n_epoch=1,
            precision="32",  # AMPを無効化して安定性向上
            gradient_clip_val=1.0,
            log_every_n_steps=50,
            enable_model_summary=True,
            enable_progress_bar=True,
            deterministic=False  # パフォーマンス向上のため
        )
        
    else:
        print("手動最適化（レガシーモード）を使用")
        # 手動最適化用の保守的なハイパーパラメータ
        model = VQGAN(
            model_config=config,
            beta=0.05,  # 超保守的なVQ commitment loss
            gan_weight=0.05,  # 最小限のGAN重み
            perceptual_weight=0.3,  # 保守的な知覚重み
            use_manual_optimization=True
        )
        
        # 手動最適化用のコールバック
        os.makedirs("/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqgan", exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath="/mnt/nas/rmjapan2000/tree/data_dir/train/model_vqgan",
            filename="{epoch:02d}-{train_recon_loss:.4f}",
            save_top_k=5,
            monitor="train_recon_loss",
            mode="min",
            save_weights_only=True,
            every_n_epochs=2
        )
        
        # ロガー
        logger = WandbLogger(
            project="tree_svs_vqgan", 
            name="vqgan_manual_training",
            log_model=True
        )
        logger.watch(model, log="all", log_graph=True, log_freq=100)
        
        # 手動最適化用のTrainer
        trainer = Trainer(
            accelerator="cuda",
            devices="auto",
            max_epochs=150,
            strategy=DDPStrategy(find_unused_parameters=True),
            callbacks=[checkpoint_callback],
            logger=logger,
            check_val_every_n_epoch=3,
            precision="32",  # 手動最適化ではAMPなし
            log_every_n_steps=10
        )
    
    # 学習実行
    trainer.fit(model, datamodule=dataloader)
    print("VQGAN学習完了！")
    
    # 最良モデルのテスト
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"最良モデル保存先: {best_model_path}")
        trainer.test(model, datamodule=dataloader, ckpt_path=best_model_path)


if __name__ == "__main__":
    main()
