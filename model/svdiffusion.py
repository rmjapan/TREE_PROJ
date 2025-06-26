import os
from xml.sax.handler import all_features
from git import Tree
from numpy import save
import torch

from torch.optim import AdamW, Adam
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
import torchvision.models as models
from timm import create_model
import os
# from filecount import last_file_num
# from Octree.octfusion.datasets import dataloader
from svdutils import *
import copy
from schedulefree import RAdamScheduleFree
import torch
from torchmetrics.classification import Precision
print("=== CUDA INIT TEST ===")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
torch.cuda.current_device()  # 強制初期化
print("=== CUDA INIT DONE ===")

# import 
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ/")
from visualize_func import visualize_with_timeout4voxel
from utils import voxel2xyzfile, npz2dense
class UNetModel(nn.Module):
    def __init__(self,
                 img_backbone: str = 'Vit',
                 base_channels: int = 16,
                 dim_mults=(1, 2, 4, 8, 16),
                 dropout: float = 0.1,
                 img_size: int = 224,
                 image_condition_dim: int = 512,
                 with_attention: bool = False,
                 verbose: bool = False,
                 ):
        super().__init__()
        self.verbose = verbose
        if img_backbone == 'Resnet50':
            # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(2048, image_condition_dim).apply(weights_init_normal)
          
        elif img_backbone == 'Swin2':
            self.backbone = create_model("swinv2_base_window8_256", pretrained=True,
                                         pretrained_cfg_overlay=dict(file='./checkpoint'))
            # self.backbone = models.swin_v2_b(pretrained=True)
            self.backbone.head = nn.Linear(1024, image_condition_dim).apply(weights_init_normal)
            # print(self.backbone)
        elif img_backbone == 'Vit':
            # 之前的一版vit用的是models.vit_b_16 768维， 目前加入attention的一版vit用的是models.vit_l_16 1024维
            # self.backbone = create_model('vit_base_patch16_224', pretrained=True)
            # self.backbone.heads = nn.Linear(768, image_condition_dim).apply(weights_init_normal)
            self.backbone = models.vit_l_16(pretrained=True)
            self.backbone.heads = nn.Linear(1024, image_condition_dim).apply(weights_init_normal)
            # print(self.backbone)
        else:
            raise NotImplementedError

        self.img_size = img_size
        channels = [base_channels, *map(lambda m: base_channels * m, dim_mults)]
        if self.verbose:
            print(channels)
        emb_dim = base_channels * 4
        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        ).apply(weights_init_normal)

        self.input_emb = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        down_num = len(channels)-1
        size = 64
        for idx in range(down_num):
            if with_attention:
                # print("****************" + str(size))
                self.downs.append(nn.ModuleList([
                    ResnetBlock(dim_list=[channels[idx],
                                          channels[idx] // 2,
                                          channels[idx] // 2,
                                          channels[idx]],
                                time_dim=emb_dim,
                                dropout=dropout),
                    MixImgAttention(channels[idx],
                                    channels[idx + 1],
                                    img_dim=image_condition_dim,
                                    voxel_size=size,
                                    dropout=dropout)
                ]))
                size = size // 2
            else:
                self.downs.append(nn.ModuleList([
                    ResnetBlock(dim_list=[channels[idx],
                                          channels[idx]//2,
                                          channels[idx]//2,
                                          channels[idx]],
                                time_dim=emb_dim,
                                dropout=dropout),
                    MixImgFeature(channels[idx],
                                  channels[idx+1],
                                  img_dim=image_condition_dim,
                                  dropout=dropout)
                ]))

        self.mid_block = ResnetBlock(
            dim_list=[channels[-1], channels[-1] // 2, channels[-1] // 2, channels[-1]],
            time_dim=emb_dim,
            dropout=dropout)
        channels = channels[::-1]
        if self.verbose:
            print(channels)
        for idx in range(down_num):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[idx], channels[idx + 1],
                                       kernel_size=4, stride=2, padding=1),
                    normalization(channels[idx + 1]),
                    activation_function(),
                ).apply(weights_init_normal)
            )
        self.out = nn.Sequential(
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # todo: rebuttal
        # self.rebuttal = nn.Conv2d(5, 3, 1, 1).apply(weights_init_normal)

    def forward(self, x, t, img):
        # todo: rebuttal
        # img = self.rebuttal(img)  
        #画像に相当するスケッチを用意する
        
        

        img = self.backbone(img)#vit
        x = self.input_emb(x.to(torch.float32))
        t = self.time_emb(self.time_pos_emb(t))
        if self.verbose:
            print(f"xの形状: {x.shape}")
            print(f"tの形状: {t.shape}")
        h = []
        for resnet, mix in self.downs:
            x = resnet(x, t)
            x = mix(x, img)
            if self.verbose:
                print(x.shape)
            h.append(x)
        x = self.mid_block(x, t)
        if self.verbose:
            print(x.shape)
        for upblock in self.ups:
            x = upblock(x + h.pop())
            if self.verbose:
                print(x.shape)
        x = self.out(x)
        if self.verbose:
            print(x.shape)
        return x
class UNetModel_with_SE(nn.Module):
    def __init__(self,
                 img_backbone: str = 'Vit',
                 base_channels: int = 16,
                 dim_mults=(1, 2, 4, 8, 16),
                 dropout: float = 0.1,
                 img_size: int = 224,
                 image_condition_dim: int = 512,
                 with_attention: bool = False,
                 verbose: bool = False,
                 reduction: int = 16,
                 ):
        super().__init__()
        self.verbose = verbose
        if img_backbone == 'Resnet50':
            # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(2048, image_condition_dim).apply(weights_init_normal)
          
        elif img_backbone == 'Swin2':
            self.backbone = create_model("swinv2_base_window8_256", pretrained=True,
                                         pretrained_cfg_overlay=dict(file='./checkpoint'))
            # self.backbone = models.swin_v2_b(pretrained=True)
            self.backbone.head = nn.Linear(1024, image_condition_dim).apply(weights_init_normal)
            # print(self.backbone)
        elif img_backbone == 'Vit':
            # 之前的一版vit用的是models.vit_b_16 768维， 目前加入attention的一版vit用的是models.vit_l_16 1024维
            # self.backbone = create_model('vit_base_patch16_224', pretrained=True)
            # self.backbone.heads = nn.Linear(768, image_condition_dim).apply(weights_init_normal)
            self.backbone = models.vit_l_16(pretrained=True)
            self.backbone.heads = nn.Linear(1024, image_condition_dim).apply(weights_init_normal)
            # print(self.backbone)
        else:
            raise NotImplementedError

        self.img_size = img_size
        channels = [base_channels, *map(lambda m: base_channels * m, dim_mults)]
        if self.verbose:
            print(channels)
        emb_dim = base_channels * 4
        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        ).apply(weights_init_normal)

        self.input_emb = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        down_num = len(channels)-1
        size = 64
        for idx in range(down_num):
            if with_attention:
                # print("****************" + str(size))
               
                self.downs.append(nn.ModuleList([
                    SEResNetBlock(dim_list=[channels[idx],
                                          channels[idx] // 2,
                                          channels[idx] // 2,
                                          channels[idx]],
                                time_dim=emb_dim,
                                dropout=dropout,
                                reduction=reduction),
                    MixImgAttention(channels[idx],
                                    channels[idx + 1],
                                    img_dim=image_condition_dim,
                                    voxel_size=size,
                                    dropout=dropout)
                ]))
                size = size // 2
            else:
                self.downs.append(nn.ModuleList([
                    SEResNetBlock(dim_list=[channels[idx],
                                          channels[idx]//2,
                                          channels[idx]//2,
                                          channels[idx]],
                                time_dim=emb_dim,
                                dropout=dropout,
                                reduction=reduction),
                    MixImgFeature(channels[idx],
                                  channels[idx+1],
                                  img_dim=image_condition_dim,
                                  dropout=dropout)
                ]))

        self.mid_block = SEResNetBlock(
            dim_list=[channels[-1], channels[-1] // 2, channels[-1] // 2, channels[-1]],
            time_dim=emb_dim,
            dropout=dropout,
            reduction=reduction)
        channels = channels[::-1]
        if self.verbose:
            print(channels)
        for idx in range(down_num):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[idx], channels[idx + 1],
                                       kernel_size=4, stride=2, padding=1),
                    normalization(channels[idx + 1]),
                    activation_function(),
                ).apply(weights_init_normal)
            )
        self.out = nn.Sequential(
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # todo: rebuttal
        # self.rebuttal = nn.Conv2d(5, 3, 1, 1).apply(weights_init_normal)

    def forward(self, x, t, img):
        # todo: rebuttal
        # img = self.rebuttal(img)  
        #画像に相当するスケッチを用意する
        
        

        img = self.backbone(img)#vit
        x = self.input_emb(x.to(torch.float32))
        t = self.time_emb(self.time_pos_emb(t))
        if self.verbose:
            print(f"xの形状: {x.shape}")
            print(f"tの形状: {t.shape}")
        h = []
        for resnet, mix in self.downs:
            x = resnet(x, t)
            x = mix(x, img)
            if self.verbose:
                print(x.shape)
            h.append(x)
        x = self.mid_block(x, t)
        if self.verbose:
            print(x.shape)
        for upblock in self.ups:
            x = upblock(x + h.pop())
            if self.verbose:
                print(x.shape)
        x = self.out(x)
        if self.verbose:
            print(x.shape)
        return x   
class DiffusionModel(LightningModule):
    def __init__(
        self,
        base_channels: int = 64,
        lr: float = 2e-4,
        batch_size: int = 8,
        optimizier: str = "adamw",
        scheduler: str = "CosineAnnealingLR",
        ema_rate: float = 0.999,
        verbose: bool = False,
        img_backbone: str = 'Vit',
        dim_mults=(1, 2, 4, 8, 16),
        training_epoch: int = 100,
        gradient_clip_val: float = 0.5,
        noise_schedule: str = "linear",
        img_size: int = 8,
        image_condition_dim: int = 512,
        dropout: float = 0.1,
        with_attention: bool = False,
        eps: float = 1e-6,
        with_SE: bool = False,
        reduction: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if with_SE:
            self.model = UNetModel_with_SE(
                img_backbone=img_backbone,
                base_channels=base_channels,
                dim_mults=dim_mults,
                dropout=dropout,
                img_size=img_size,
                image_condition_dim=image_condition_dim,
                with_attention=with_attention,
                reduction=reduction,
                verbose=verbose)
        else:
            self.model = UNetModel(
                img_backbone=img_backbone,
                base_channels=base_channels,
                dim_mults=dim_mults,
                dropout=dropout,
                img_size=img_size,
                image_condition_dim=image_condition_dim,
                with_attention=with_attention,
                verbose=verbose)

        self.batch_size = batch_size
        self.lr = lr
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)

        self.image_feature_drop_out = dropout
        self.optim = optimizier
        self.scheduler = scheduler
        self.eps = eps
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

    def training_loss(self, img, img_features):
        from loss_functions import weighted_mae_loss
        batch = img.shape[0]

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times).to(torch.float32)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        pred = self.model(noised_img, noise_level, img_features)
        img = img.to(torch.float32)
        # return weighted_mae_loss(   pred,img),pred
        # return 100*F.l1_loss(pred, img), pred
        return 100 * F.mse_loss(pred, img),pred

    def training_step(self, batch, batch_idx):
        #Pytorchが訓練中に自動的に呼び出す関数
        #データローダーからバッチを取得
        
        voxel = batch["voxel"].unsqueeze(1)
        img_features = batch["img"]
        loss,pred_voxel = self.training_loss(voxel, img_features)
        loss = loss.mean()

        opt = self.optimizers()

        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(
        opt,
                gradient_clip_algorithm='norm',
                gradient_clip_val=self.gradient_clip_val
            )
 

        opt.step()
        self.update_EMA()
        self.log("train_loss", loss.clone().detach().item(), prog_bar=True)
        if (self.global_step+1)%1000==0:
            
            voxels=self.sample_with_img(img_features,batch_size=self.batch_size,steps=21,verbose=True)
            
            print(voxels.shape)
            voxel=voxels[0]
            #0.8以上は1.0,0.4以上は0.5,0.4未満は0.0,それ以外は-1.0
            voxel=torch.where(
                voxel>0.8,1.0,
                torch.where(
                    voxel>0.4,0.5,
                    torch.where(
                        voxel>0.0,0.0,-1.0
                    )
                )
            )
            voxel=voxel.squeeze(0).detach().cpu().numpy()
            visualize_with_timeout4voxel(voxel)
            
            voxel2xyzfile(voxel)
            #画像を保存
            image=img_features[0].squeeze(0).detach().cpu().numpy()
            image=image.transpose(1,2,0)
            image=image*255.0
            image=image.astype(np.uint8)
            import cv2
            cv2.imwrite("image.png",image)
            
    def voxel2svs(self,voxel):
        voxel=torch.where(
                    voxel>0.8,1.0,
                    torch.where(
                        voxel>0.4,0.5,
                        torch.where(
                            voxel>0.0,0.0,-1.0
                        )
                    )
                )
        return voxel
    
    def svs2integerclass(self,svs):
        svs_class=torch.where(svs == -1.0, 0,
                          torch.where(svs == 0.0, 1,
                          torch.where(svs == 0.5, 2, 3)))
        return svs_class
    def evaluate_filenametest(self, index):
        import numpy as np
        import os
        from utils import voxel2xyzfile, npz2dense
        from visualize_func import visualize_with_timeout4voxel
        import cv2
        
        # ファイルパスの設定
        voxel_path = f"/mnt/nas/rmjapan2000/tree/data_dir/train/svs_cgvi/svs_{index}.npz"
        sketch_path = f"/mnt/nas/rmjapan2000/tree/data_dir/train/sketch_cgvi/sketch_canny_{index}.png"
        
 
            
        try:
            print(f"=== Processing index {index} ===")
            
            # === GT (Ground Truth) データの処理 ===
            print(f"\n--- GT (Ground Truth) Processing ---")
            print(f"Loading GT voxel data from: {voxel_path}")
            gt_voxel_data = np.load(voxel_path)
            
            # npz2dense関数を使用してサイズを64x64x64に調整
            gt_voxel_data = npz2dense(gt_voxel_data, 64, 64, 64)
            
            # データローダーと同じ値の変換を適用
            gt_voxel_data[gt_voxel_data == 0] = -1.0   # 空白
            gt_voxel_data[gt_voxel_data == 1] = 0.0    # 葉
            gt_voxel_data[gt_voxel_data == 1.5] = 0.5  # 枝
            gt_voxel_data[gt_voxel_data == 2] = 1.0    # 幹
            
            print(f"GT after preprocessing unique values: {np.unique(gt_voxel_data)}")
            print(f"GT voxel data shape: {gt_voxel_data.shape}")
            
            # GT各値の分布を確認
            unique_values, counts = np.unique(gt_voxel_data, return_counts=True)
            total_voxels = gt_voxel_data.size
            print("GT Voxel distribution:")
            for val, count in zip(unique_values, counts):
                percentage = (count / total_voxels) * 100
                if val == -1.0:
                    print(f"  Empty (空白): {count} ({percentage:.2f}%)")
                elif val == 0.0:
                    print(f"  Leaf (葉): {count} ({percentage:.2f}%)")
                elif val == 0.5:
                    print(f"  Branch (枝): {count} ({percentage:.2f}%)")
                elif val == 1.0:
                    print(f"  Trunk (幹): {count} ({percentage:.2f}%)")
            
            # GTボクセルデータの可視化
            print("Visualizing GT voxel data...")
            visualize_with_timeout4voxel(gt_voxel_data, title=f"GT Voxel {index}")
            
            # GT XYZファイルとして保存
            gt_xyz_filename = f"gt_voxel_{index:04d}.xyz"
            print(f"Saving GT voxel data as XYZ format to: {gt_xyz_filename}")
            voxel2xyzfile(gt_voxel_data, gt_xyz_filename)
            
            # === GEN (Generated) データの処理 ===
            print(f"\n--- GEN (Generated) Processing ---")
            print(f"Loading sketch image from: {sketch_path}")
            
            # スケッチ画像の読み込みと前処理
            sketch_img = cv2.imread(sketch_path)
            if sketch_img is None:
                print(f"Error: Could not load sketch image: {sketch_path}")
                return
                
            sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
            sketch_img = cv2.resize(sketch_img, (224, 224))  # モデルの入力サイズに合わせる
            sketch_img = sketch_img.astype(np.float32) / 255.0
            sketch_img = sketch_img.transpose(2, 0, 1)  # HWC -> CHW
            
            # テンソルに変換してバッチ次元を追加
            import torch
            sketch_tensor = torch.from_numpy(sketch_img).unsqueeze(0).to(self.device)
            
            print("Generating voxel data using diffusion model...")
            # 拡散モデルで生成
            gen_voxels = self.sample_with_img(sketch_tensor, batch_size=1, steps=50, verbose=True)
            gen_voxel = gen_voxels[0]  # バッチから最初の要素を取得
            
            # 生成されたボクセルを適切な値に変換
            gen_voxel = self.voxel2svs(gen_voxel)
            gen_voxel_data = gen_voxel.squeeze(0).detach().cpu().numpy()
            
            print(f"GEN after preprocessing unique values: {np.unique(gen_voxel_data)}")
            print(f"GEN voxel data shape: {gen_voxel_data.shape}")
            
            # GEN各値の分布を確認
            unique_values, counts = np.unique(gen_voxel_data, return_counts=True)
            total_voxels = gen_voxel_data.size
            print("GEN Voxel distribution:")
            for val, count in zip(unique_values, counts):
                percentage = (count / total_voxels) * 100
                if val == -1.0:
                    print(f"  Empty (空白): {count} ({percentage:.2f}%)")
                elif val == 0.0:
                    print(f"  Leaf (葉): {count} ({percentage:.2f}%)")
                elif val == 0.5:
                    print(f"  Branch (枝): {count} ({percentage:.2f}%)")
                elif val == 1.0:
                    print(f"  Trunk (幹): {count} ({percentage:.2f}%)")
            
            # GENボクセルデータの可視化
            print("Visualizing GEN voxel data...")
            visualize_with_timeout4voxel(gen_voxel_data, title=f"GEN Voxel {index}")
            
            # GEN XYZファイルとして保存
            gen_xyz_filename = f"gen_voxel_{index:04d}.xyz"
            print(f"Saving GEN voxel data as XYZ format to: {gen_xyz_filename}")
            voxel2xyzfile(gen_voxel_data, gen_xyz_filename)
            
            # === 結果サマリー ===
            print(f"\n=== Successfully processed index {index} ===")
            print(f"GT:")
            print(f"  - Visualization completed")
            print(f"  - XYZ file saved: {gt_xyz_filename}")
            print(f"GEN:")
            print(f"  - Generation completed")
            print(f"  - Visualization completed") 
            print(f"  - XYZ file saved: {gen_xyz_filename}")
            
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            import traceback
            traceback.print_exc()
    def evaluate_test(self,
                      dataloader,
                      steps=21,
                      base_dir="/mnt/nas/rmjapan2000/tree/eval",
                      folder_name="test",
                      data_size=16
                      ):  
        import glob
        import os
        import pandas as pd
        import cv2
        
        folder_name=f"{folder_name}{steps}"
        save_folder = f"{base_dir}/{folder_name}"
        os.makedirs(save_folder,exist_ok=True)
        
        # GT (Ground Truth) 用のフォルダ
        gt_voxel_folder=f"{save_folder}/gt_voxeldataset"
        gt_image_folder=f"{save_folder}/gt_imagedataset"
        gt_xyz_folder=f"{save_folder}/gt_xyz_dataset"
        
        # GEN (Generated) 用のフォルダ
        gen_voxel_folder=f"{save_folder}/gen_voxeldataset"
        gen_image_folder=f"{save_folder}/gen_imagedataset"
        gen_xyz_folder=f"{save_folder}/gen_xyz_dataset"
        
        csv_filename=f"{save_folder}/metrics.csv"
        
        # フォルダ作成
        os.makedirs(gt_voxel_folder,exist_ok=True)
        os.makedirs(gt_image_folder,exist_ok=True)
        os.makedirs(gt_xyz_folder,exist_ok=True)
        os.makedirs(gen_voxel_folder,exist_ok=True)
        os.makedirs(gen_image_folder,exist_ok=True)
        os.makedirs(gen_xyz_folder,exist_ok=True)
        
        file_list=glob.glob(f"{gen_voxel_folder}/*.npy")
        file_count=len(file_list)
        
        # メトリクスデータを保存するリスト
        all_metrics_data = []
        count=0
        max_iter=data_size//self.batch_size
        
        for batch in dataloader:
            if count >= max_iter:
                break
                
            count += 1
            img_features=batch["img"]
            target_voxels=batch["voxel"].unsqueeze(1).to(torch.float32)
            print(f"batch_size{img_features.shape[0]}")
            
            voxels=self.sample_with_img(img_features,batch_size=img_features.shape[0],steps=steps,verbose=True)
            print(voxels.shape)
            file_list=glob.glob(f"{gen_voxel_folder}/*.npy")
            file_count=len(file_list)
        
            for i in range(voxels.shape[0]):
                
                #0.8以上は1.0,0.4以上は0.5,0.4未満は0.0,それ以外は-1.0
                gen_voxel=voxels[i]
                gen_voxel=self.voxel2svs(gen_voxel)
                gen_voxel_class=self.svs2integerclass(gen_voxel)
                target_voxel_class=self.svs2integerclass(target_voxels[i])
                
                blank_ac,leaf_ac,branch_ac,trunk_ac,mean_ac = self.calc_accuracy(gen_voxel_class, target_voxel_class)
                mae = F.l1_loss(gen_voxel, target_voxels[i].squeeze(0)).item()
                mse = F.mse_loss(gen_voxel, target_voxels[i].squeeze(0)).item()
                
                # メトリクスデータをリストに追加
                metrics_data = {
                    'file_index': i + file_count,
                    'blank_accuracy': blank_ac,
                    'leaf_accuracy': leaf_ac,
                    'branch_accuracy': branch_ac,
                    'trunk_accuracy': trunk_ac,
                    'mean_accuracy': mean_ac,
                    'mae': mae,
                    'mse': mse
                }
                print(metrics_data)
                all_metrics_data.append(metrics_data)
                
                # GEN (Generated) データの処理
                gen_voxel_data=gen_voxel.squeeze(0).detach().cpu().numpy()
                print(f"Visualizing GEN voxel {i+file_count}...")
                visualize_with_timeout4voxel(gen_voxel_data, title=f"GEN Voxel {i+file_count}")
                
                # GT (Ground Truth) データの処理
                gt_voxel_data=target_voxels[i].squeeze(0).detach().cpu().numpy()
                print(f"Visualizing GT voxel {i+file_count}...")
                visualize_with_timeout4voxel(gt_voxel_data, title=f"GT Voxel {i+file_count}")
    
                # 画像データの処理
                image=img_features[i].squeeze(0).detach().cpu().numpy()
                image=image.transpose(1,2,0)
                image=image*255.0
                image=image.astype(np.uint8)
            
                # Save GEN voxel data
                gen_voxel_filename = f"{gen_voxel_folder}/gen_voxel_{i+file_count:04d}.npy"
                np.save(gen_voxel_filename, gen_voxel_data)
                
                # Save GT voxel data
                gt_voxel_filename = f"{gt_voxel_folder}/gt_voxel_{i+file_count:04d}.npy"
                np.save(gt_voxel_filename, gt_voxel_data)
                
                # Save image data (both GT and GEN use same input image)
                gt_image_filename = f"{gt_image_folder}/gt_image_{i+file_count:04d}.png"
                gen_image_filename = f"{gen_image_folder}/gen_image_{i+file_count:04d}.png"
                cv2.imwrite(gt_image_filename, image)
                cv2.imwrite(gen_image_filename, image)
                
                # Save XYZ files
                gen_xyz_filename = f"{gen_xyz_folder}/gen_voxel_{i+file_count:04d}.xyz"
                gt_xyz_filename = f"{gt_xyz_folder}/gt_voxel_{i+file_count:04d}.xyz"
                voxel2xyzfile(gen_voxel_data, gen_xyz_filename)
                voxel2xyzfile(gt_voxel_data, gt_xyz_filename)
                
                print(f"Saved files for index {i+file_count}:")
                print(f"  GT:  {gt_voxel_filename}, {gt_image_filename}, {gt_xyz_filename}")
                print(f"  GEN: {gen_voxel_filename}, {gen_image_filename}, {gen_xyz_filename}")
        
        # Create DataFrame from all metrics data and save to CSV
        df = pd.DataFrame(all_metrics_data)
        df.to_csv(csv_filename, index=False)
        print(f"Metrics saved to: {csv_filename}")

    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or not self.validation_step_outputs:
            return
            
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_mae_loss = torch.stack([x['mae_loss'] for x in self.validation_step_outputs]).mean()
        avg_blank_precision = torch.stack([x['blank_precision'] for x in self.validation_step_outputs]).mean()
        avg_material_precision = torch.stack([x['material_precision'] for x in self.validation_step_outputs]).mean()
        avg_branch_precision = torch.stack([x['branch_precision'] for x in self.validation_step_outputs]).mean()
        avg_trunk_precision = torch.stack([x['trunk_precision'] for x in self.validation_step_outputs]).mean()
        avg_blank_accuracy = torch.stack([x['blank_accuracy'] for x in self.validation_step_outputs]).mean()
        avg_material_accuracy = torch.stack([x['material_accuracy'] for x in self.validation_step_outputs]).mean()
        avg_branch_accuracy = torch.stack([x['branch_accuracy'] for x in self.validation_step_outputs]).mean()
        avg_trunk_accuracy = torch.stack([x['trunk_accuracy'] for x in self.validation_step_outputs]).mean()
        
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_mae_loss', avg_mae_loss, prog_bar=True)
        self.log('val_blank_precision', avg_blank_precision, prog_bar=True)
        self.log('val_material_precision', avg_material_precision, prog_bar=True)
        self.log('val_branch_precision', avg_branch_precision, prog_bar=True)
        self.log('val_trunk_precision', avg_trunk_precision, prog_bar=True)
        self.log('val_blank_accuracy', avg_blank_accuracy, prog_bar=True)
        self.log('val_material_accuracy', avg_material_accuracy, prog_bar=True)
        self.log('val_branch_accuracy', avg_branch_accuracy, prog_bar=True)
        self.log('val_trunk_accuracy', avg_trunk_accuracy, prog_bar=True)
        
        # Clear the outputs for next epoch
        self.validation_step_outputs.clear()
        
    def calc_accuracy(self,pred_classes,target_classes):
         from torchmetrics import Accuracy
         device=pred_classes.device
         accuracy = Accuracy(task="multiclass", num_classes=4, average=None).to(device)
         accuracy_score = accuracy(pred_classes.flatten(), target_classes.flatten())
         blank_accuracy = accuracy_score[0].item()    # Class 0 (-1.0) の精度
         leaf_accuracy = accuracy_score[1].item() # Class 1 (0.0) の精度
         branch_accuracy = accuracy_score[2].item()   # Class 2 (0.5) の精度
         trunk_accuracy = accuracy_score[3].item()    # Class 3 (1.0) の精度
         mean_accuracy=accuracy_score.mean()
         return blank_accuracy,leaf_accuracy,branch_accuracy,trunk_accuracy,mean_accuracy
         
    def calc_precision(self,pred_classes,target_classes):
         from torchmetrics import Precision
         precision = Precision(task="multiclass", num_classes=4, average=None).to("cuda")
         precision_score = precision(pred_classes.flatten(), target_classes.flatten())
         blank_precision = precision_score[0].item()    # Class 0 (-1.0) の精度
         leaf_precision = precision_score[1].item() # Class 1 (0.0) の精度
         branch_precision = precision_score[2].item()   # Class 2 (0.5) の精度
         trunk_precision = precision_score[3].item()    # Class 3 (1.0) の精度
         mean_precision=precision_score.mean()
         return blank_precision,leaf_precision,branch_precision,trunk_precision,mean_precision
         
    def validation_step(self, batch, batch_idx):
        voxel = batch["voxel"].unsqueeze(1).to(torch.float32)
        img_features = batch["img"]
        
        # Calculate loss and prediction
        loss, pred_voxel = self.training_loss(voxel, img_features)
        mae_loss = F.l1_loss(pred_voxel, voxel)
        
        # Convert predictions to discrete values
        pred_voxel_discrete=self.voxel2svs(pred_voxel)
        

    
        # Convert floating point voxel values to integer class indices
        pred_classes = self.svs2integerclass(pred_voxel_discrete)
        
        target_classes =self.svs2integerclass(voxel)
        
        blank_accuracy, material_accuracy, branch_accuracy, trunk_accuracy,mean_accuracy = self.calc_accuracy(pred_classes, target_classes)
        blank_precision, material_precision, branch_precision, trunk_precision,mean_precision = self.calc_precision(pred_classes, target_classes)
        # Calculate voxel distribution ratios
        def calculate_voxel_distribution(voxel_data):
            """Calculate the distribution of voxel values"""
            voxel_data_flat = voxel_data.flatten()
            element_num = len(voxel_data_flat)
            
            result_blank = (voxel_data_flat == -1).sum().item()
            result_trunk = (voxel_data_flat == 1).sum().item()
            result_branch = (voxel_data_flat == 0.5).sum().item()
            result_leaf = (voxel_data_flat == 0).sum().item()
            
            blank_ratio = result_blank / element_num * 100
            trunk_ratio = result_trunk / element_num * 100
            branch_ratio = result_branch / element_num * 100
            leaf_ratio = result_leaf / element_num * 100
            
            return blank_ratio, trunk_ratio, branch_ratio, leaf_ratio
        
        # Calculate distributions for predicted and ground truth voxels
        pred_blank_ratio, pred_trunk_ratio, pred_branch_ratio, pred_leaf_ratio = calculate_voxel_distribution(pred_voxel_discrete)
        gt_blank_ratio, gt_trunk_ratio, gt_branch_ratio, gt_leaf_ratio = calculate_voxel_distribution(voxel)
        
        print(f"Predicted: blank={pred_blank_ratio:.2f}%, trunk={pred_trunk_ratio:.2f}%, branch={pred_branch_ratio:.2f}%, leaf={pred_leaf_ratio:.2f}%")
        print(f"Ground Truth: blank={gt_blank_ratio:.2f}%, trunk={gt_trunk_ratio:.2f}%, branch={gt_branch_ratio:.2f}%, leaf={gt_leaf_ratio:.2f}%")
        
        # Calculate mean values
        loss = loss.mean()
        mae_loss = mae_loss.mean()

        
        # Log metrics
        self.log("val_loss", loss.clone().detach().item(), prog_bar=True)
        self.log("val_mae_loss", mae_loss.clone().detach().item(), prog_bar=True)
        self.log("val_precision", mean_precision.clone().detach().item(), prog_bar=True)
        self.log("val_accuracy", mean_accuracy.clone().detach().item(), prog_bar=True)
        self.log("val_blank_precision", blank_precision, prog_bar=True)
        self.log("val_material_precision", material_precision, prog_bar=True)
        self.log("val_branch_precision", branch_precision, prog_bar=True)
        self.log("val_trunk_precision", trunk_precision, prog_bar=True)
        self.log("val_blank_accuracy", blank_accuracy, prog_bar=True)
        self.log("val_material_accuracy", material_accuracy, prog_bar=True)
        self.log("val_branch_accuracy", branch_accuracy, prog_bar=True)
        self.log("val_trunk_accuracy", trunk_accuracy, prog_bar=True)
        
        # Log predicted voxel distribution ratios
        self.log("val_blank_ratio", pred_blank_ratio, prog_bar=True)
        self.log("val_leaf_ratio", pred_leaf_ratio, prog_bar=True)
        self.log("val_branch_ratio", pred_branch_ratio, prog_bar=True)
        self.log("val_trunk_ratio", pred_trunk_ratio, prog_bar=True)
        
        # Log ground truth voxel distribution ratios
        self.log("gt_blank_ratio", gt_blank_ratio, prog_bar=True)
        self.log("gt_leaf_ratio", gt_leaf_ratio, prog_bar=True)
        self.log("gt_branch_ratio", gt_branch_ratio, prog_bar=True)
        self.log("gt_trunk_ratio", gt_trunk_ratio, prog_bar=True)
        
        return {
            # Loss metrics
            "loss": loss,
            "mae_loss": mae_loss,
            
            # Precision metrics
            "blank_score": blank_precision,
            "material_score": material_precision,
            "branch_score": branch_precision,
            "trunk_score": trunk_precision,
            
            # Accuracy metrics
            "blank_accuracy": blank_accuracy,
            "material_accuracy": material_accuracy,
            "branch_accuracy": branch_accuracy,
            "trunk_accuracy": trunk_accuracy,
            
            # Predicted voxel distribution ratios
            "val_blank_ratio": pred_blank_ratio,
            "val_leaf_ratio": pred_leaf_ratio,
            "val_branch_ratio": pred_branch_ratio,
            "val_trunk_ratio": pred_trunk_ratio,
            
            # Ground truth voxel distribution ratios
            "gt_blank_ratio": gt_blank_ratio,
            "gt_leaf_ratio": gt_leaf_ratio,
            "gt_branch_ratio": gt_branch_ratio,
            "gt_trunk_ratio": gt_trunk_ratio,
        }

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):

        if self.optim == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optim == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == "radam":
            optimizer = RAdamScheduleFree(self.model.parameters(), lr=self.lr)
            
        else:
            raise NotImplementedError

        if self.scheduler == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif self.scheduler == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        else:
            raise NotImplementedError
        return [optimizer], [lr_scheduler]

    @staticmethod
    def get_sampling_timesteps(batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

        #検証用データセットを用いてMAEｍ
    @torch.no_grad()
    
    def sample_with_img(self,
                        img,
                        batch_size=1,
                        steps=50,
                        truncated_index: float = 0.0,
                        img_weight: float = 1.0,
                        verbose: bool = False):
        vxl_size = 64
        shape = (batch_size, 1, vxl_size, vxl_size, vxl_size)
        time_pairs = self.get_sampling_timesteps(
            batch=batch_size, device=self.device, steps=steps)
        voxel = torch.randn(shape, device=self.device)
        x_start = None
        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs

        for time, time_next in _iter:
            log_snr = self.log_snr(time).type_as(time)
            log_snr_next = self.log_snr(time_next).type_as(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, voxel), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time).type_as(time)

            x_zero_none = self.model(voxel, noise_cond, img)
            x_start = x_zero_none + img_weight * \
                      (self.model(voxel, noise_cond, img) - x_zero_none)

            c = -torch.expm1(log_snr - log_snr_next)
            mean = alpha_next * (voxel * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(voxel),
                torch.zeros_like(voxel)
            )
            voxel = mean + torch.sqrt(variance) * noise
        return voxel

if __name__ == "__main__":
    from pytorch_lightning.trainer import Trainer
    import sys
    sys.path.append("/home/ryuichi/tree/TREE_PROJ/")
    from my_dataset.svddata_loader import direction_TreeDataLoader,TreeDataLoader
    model = DiffusionModel(batch_size=4,verbose=False,with_SE=True,with_attention=True,gradient_clip_val=1.0,reduction=4,img_backbone="Vit")
    use_pretrain = True
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=29-losstrain_loss=1.5480.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=13-losstrain_loss=1.7015.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=08-losstrain_loss=1.1489.ckpt"
    # # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi/epochepoch=08-losstrain_loss=0.8772.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi/epochepoch=30-losstrain_loss=0.9248.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=33-losstrain_loss=1.2695.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=01-losstrain_loss=1.7147.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=03-losstrain_loss=1.8162.ckpt"
    # ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=15-losstrain_loss=1.6484.ckpt"
    ckpt="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2/epochepoch=96-losstrain_loss=1.7216.ckpt"
    if use_pretrain:
        model.load_state_dict(torch.load(ckpt,map_location=torch.device('cuda'))['state_dict'])
    from pytorch_lightning.callbacks import ModelCheckpoint

    # 保存先パスとファイル名テンプレートを指定
    checkpoint_callback = ModelCheckpoint(
        dirpath="/mnt/nas/rmjapan2000/tree/data_dir/train/model_cgvi_ver2",
        filename="epoch{epoch:02d}-loss{train_loss:.4f}",  # ファイル名テンプレート
        save_top_k=5,  # ベスト3モデルだけ残す
        monitor="train_loss",  # 監視する指標
        mode="min",  # lossが小さい方が良い
        save_weights_only=True  # モデル全体を保存（Trueなら重みだけ）
    )


    # dataloader = direction_TreeDataLoader(batch_size=16)
    dataloader=TreeDataLoader(batch_size=4)
    train_data=dataloader.train_dataloader()
    val_data=dataloader.val_dataloader()
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project="tree_svs_diffusion")
    logger.watch(model.model,log="all",log_graph=True,log_freq=100)
    trainer = Trainer(
        accelerator="cuda",
        devices="auto",
        max_epochs=100,
        strategy="ddp",
        callbacks=[checkpoint_callback],
        precision=16,
        logger=logger,
        # gradient_clip_val=0.1
    )
    # サブデータローダーを作成（16個のサンプルに制限）
    model.eval()
    # model.evaluate_filenametest(29820)
    model.evaluate_test(
        steps=50,
        dataloader=val_data,
        data_size=32,
        folder_name="32_data_step50"
    )
    # trainer.validate(model,val_data)
    # trainer.fit(model, train_data, val_data)