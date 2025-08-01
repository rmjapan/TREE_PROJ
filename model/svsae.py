import sys
from turtle import distance

from git import Tree
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from flask import config
import safetensors
from torch.nn import Conv3d, ConvTranspose3d, ReLU,Sigmoid,LeakyReLU,Tanh
from utils import npz2dense
import numpy as np
import torch
import os
import json
import torch.nn as nn
from safetensors.torch import load_file
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from typing import Dict, Optional, Tuple, Union
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import DecoderOutput
from safetensors.torch import load_file, save_file
from typing import Callable, Optional, Union
from copy import deepcopy
from diffusers.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from torch.nn import AdaptiveAvgPool3d
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Original Squeeze-and-Excitation Block from CVPR 2018"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1, 1)
        return x * y

class SEResNetBlock(nn.Module):
    """Original implementation from 'Squeeze-and-Excitation Networks'"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # SE comes after conv2 but before residual
        x += residual
        return self.relu(x)
class DecoderSEResNetBlock(nn.Module):
    """Decoder用に調整したSEResNetBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvTranspose3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = ConvTranspose3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvTranspose3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        # print(f"In_channels: {self.in_channels}")
        # print(f"Out_channels: {self.out_channels}")
        residual = self.shortcut(x)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return self.relu(x)

class AutoencoderConfig(ConfigMixin):
    config_name="autoencoder_config"#必須
    def __init__(
        self,
        in_channels:int=1,
        out_channels:int=1,
        block_out_channels:Tuple[int]=(4,8,16),
        act_fn:str=["leakyrelu","sigmoid"],
        layers_per_block:int=3,#ブロックごとのレイヤー数(Conv3d+Batch+Act_fn-> 3)
        latent_channels:int=1,
        sample_size:int=256,
        scaling_factor:float=0.09366,#what is this?
        encoder_type:str="ver1",
        decoder_type="ver1",
        device="cuda"
    ):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.block_out_channels=block_out_channels
        self.latent_channels=latent_channels
        self.scaling_factor=scaling_factor
        self.sample_size=sample_size
        self.encoder_type=encoder_type
        self.decoder_type=decoder_type
        self.device=device
    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path:Union[str,os.PathLike],**kwargs):
        """
        
        AutoencoderConfigクラス自身を cls で参照し、config.jsonから読み込んだ設定を引数として渡して新しいインスタンスを生成するプログラム
        @classmethodでは、第一引数にclsが指定される
        これは、このクラスメソッドが呼び出されたクラス自身を参照します。

        """
        config_file=os.path.join(pretrained_model_name_or_path,"config.json")
        with open(config_file,"r") as f:
            config_dict=json.load(f)
            return cls(**config_dict)#キーワード引数展開
    def save_pretrained(self,save_directory:Union[str,os.PathLike]):
        config_file=os.path.join(save_directory,"config.json")
        with open(config_file,"w") as f:
            #self.__dict__はクラスの属性と値を辞書型で返すメソッド
            json.dump(self.__dict__,f)
#class SVSEncoderの一部だけ変えたい

class VectorQuantizer(nn.Module):
    def __init__(self,config:AutoencoderConfig,beta):
        super().__init__()
        self.D=config.latent_channels
        # コードブックサイズを現実的な値に変更（64^3は大きすぎる）
        self.K=512  # または1024, 2048など
        self.embedding=nn.Embedding(self.K,self.D)
        # print(f"Embedding shape: {self.embedding.weight.shape}")
        #重みの初期化を改善
        nn.init.uniform_(self.embedding.weight,-1/self.K,1/self.K)
        self.beta=beta
        
        # Commitment loss用の移動平均（EMAベースの更新を追加）
        self.decay = 0.99
        self.register_buffer('cluster_size', torch.zeros(self.K))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
    def CalcDist_x2codebook(self,x):
        """
        Encoderの出力ｘとCodeBookのベクトルの距離を計算する.
        """
        x_flat=x.view(-1,self.D)
        distances =(torch.sum(x_flat**2,dim=1,keepdim=True)
        + torch.sum(self.embedding.weight**2,dim=1)
        -2*torch.matmul(x_flat,self.embedding.weight.t()))
        return distances
        
    def vector_quantized(self,x):
        """
        Encoderの出力を量子化する.
        """
        x_shape=x.shape
        distances=self.CalcDist_x2codebook(x)
        
        # 最近傍のインデックスを取得(もっとも距離が近かったCodeBookのベクトル番号を取得）
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        #onehotベクトルをつくる.
        encodings=torch.zeros(encoding_indices.shape[0],self.K,device=x.device)
        #Kのところに1を入れる.encoding indicesで指定されたindexのところに
        encodings.scatter_(1,encoding_indices,1)
        
        #量子化する+形状をChange
        quantized=torch.matmul(encodings,self.embedding.weight).view(x_shape)
        
        # EMA更新（学習中のみ）
        if self.training:
            # NaN チェックを追加
            if torch.isnan(encodings.sum()) or torch.isinf(encodings.sum()):
                print("Warning: NaN/Inf detected in encodings, skipping EMA update")
                return quantized, encodings
                
            # クラスターサイズの更新
            self.cluster_size = self.cluster_size * self.decay + (1 - self.decay) * encodings.sum(0)
            
            # 埋め込みベクトルの平均の更新
            embed_sum = torch.matmul(encodings.t(), x.view(-1, self.D))
            
            # NaN チェック
            if torch.isnan(embed_sum).any() or torch.isinf(embed_sum).any():
                print("Warning: NaN/Inf detected in embed_sum, skipping EMA update")
                return quantized, encodings
            
            self.embed_avg = self.embed_avg * self.decay + (1 - self.decay) * embed_sum
            
            # 正規化された埋め込みベクトルの更新
            n = self.cluster_size.sum()
            if n > 0:  # ゼロ除算を防ぐ
                cluster_size = (self.cluster_size + 1e-5) / (n + self.K * 1e-5) * n
                embed_normalized = self.embed_avg / (cluster_size.unsqueeze(1) + 1e-8)  # ゼロ除算を防ぐ
                
                # NaN チェック
                if not torch.isnan(embed_normalized).any() and not torch.isinf(embed_normalized).any():
                    self.embedding.weight.data.copy_(embed_normalized)
                else:
                    print("Warning: NaN/Inf detected in embed_normalized, skipping weight update")
        
        return quantized,encodings
        
    def calc_loss(self,quantized,x,encodings):
        """
        Code Book Loss: q_latentLoss:埋め込みベクトルをEncoderベクトルに近づける.
        E_latent Loss : e_latentLoss:エンコーダベクトルを埋め込みベクトルに近づける
        """
        # VQ損失の改善（より安定な計算）
        e_latent_loss=F.mse_loss(quantized.detach(),x)#Encoder側に伝わるloss
        q_latent_loss=F.mse_loss(quantized,x.detach())#Codebook側に伝わるloss
        
        # NaN check
        if torch.isnan(e_latent_loss) or torch.isnan(q_latent_loss):
            print("Warning: NaN detected in VQ loss")
            return torch.tensor(0.25, device=x.device, requires_grad=True)
        
        # Clamp losses to prevent explosion
        e_latent_loss = torch.clamp(e_latent_loss, min=0.0, max=10.0)
        q_latent_loss = torch.clamp(q_latent_loss, min=0.0, max=10.0)
        
        loss = q_latent_loss + self.beta * e_latent_loss
        return loss
        
    def forward(self,x:torch.Tensor):
        x=x.permute(0,2,3,4,1).contiguous()
        quantized,encodings=self.vector_quantized(x)
        loss = self.calc_loss(quantized,x,encodings)
        
        quantized = x + (quantized - x).detach()
        return loss,quantized.permute(0,4,1,2,3).contiguous(),encodings


class SVSEncoder(torch.nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        # 新しいブロック構成 [(out_channels, stride)]
        self.blocks = nn.Sequential(
            nn.Sequential(
                Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
                SEResNetBlock(4, 4),
                nn.BatchNorm3d(4),
                LeakyReLU()
            ),
            nn.Sequential(
                Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
                SEResNetBlock(8, 8),
                nn.BatchNorm3d(8),
                LeakyReLU()
            ),
            nn.Sequential(
                Conv3d(8, 16, kernel_size=3, stride=1, padding=1),  # ダウンサンプリングしない
                SEResNetBlock(16, 16),
                nn.BatchNorm3d(16),
                LeakyReLU()
            ),
            nn.Sequential(
                Conv3d(16, config.latent_channels, kernel_size=1, stride=1),  # チャネル次元を1に圧縮
                nn.BatchNorm3d(config.latent_channels),
                LeakyReLU()
            )
        )

    def forward(self, x):
        return self.blocks(x)
    
class SVSDecoder(torch.nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        
        # 対称的なデコーダ構造
        self.blocks = nn.Sequential(
            nn.Sequential(
                ConvTranspose3d(config.latent_channels, 32, kernel_size=1, stride=1),
                DecoderSEResNetBlock(32, 32),
                nn.BatchNorm3d(32),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
                DecoderSEResNetBlock(16, 16),
                nn.BatchNorm3d(16),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=1),
                DecoderSEResNetBlock(8, 8),
                nn.BatchNorm3d(8),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1),
                DecoderSEResNetBlock(4, 4),
                nn.BatchNorm3d(4),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(4, 1, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(1),
                Tanh()
            )
        )

    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.blocks(x)
class SVSDecoder_ver2(torch.nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        # 対称的なデコーダ構造
        self.blocks = nn.Sequential(
            nn.Sequential(
                ConvTranspose3d(1, 32, kernel_size=1, stride=1),
                DecoderSEResNetBlock(32, 32),
                nn.BatchNorm3d(32),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
                DecoderSEResNetBlock(16, 16),
                nn.BatchNorm3d(16),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=1),
                DecoderSEResNetBlock(8, 8),
                nn.BatchNorm3d(8),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1),
                DecoderSEResNetBlock(4, 4),
                nn.BatchNorm3d(4),
                LeakyReLU()
            ),
            nn.Sequential(
                ConvTranspose3d(4, 1, kernel_size=4, stride=2, padding=1),
                DecoderSEResNetBlock(1,1),
                nn.BatchNorm3d(1),
                LeakyReLU(),
            ),
            nn.Sequential(
                ConvTranspose3d(1,1,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm3d(num_features=1),
                nn.Tanh()
            ),
        )
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # デバッグ用: 各ブロックの入出力形状を表示
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Decoder input shape: {x.shape}")
            intermediate = x
            for i, block in enumerate(self.blocks):
                intermediate = block(intermediate)
                print(f"Decoder block {i} output shape: {intermediate.shape}")
            return intermediate
        return self.blocks(x)
class SVSEncoder_ver2(torch.nn.Module):
    #カーネルサイズを7に変更
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        # 新しいブロック構成 [(out_channels, stride)]
        self.blocks = nn.Sequential(
            nn.Sequential(
                #256+2*1=258 258-7=251 251/2=125
                Conv3d(1, 4, kernel_size=7, stride=2, padding=3),
                SEResNetBlock(4, 4),
                nn.BatchNorm3d(4),
                LeakyReLU()
            ),
            nn.Sequential(
                Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
                SEResNetBlock(8, 8),
                nn.BatchNorm3d(8),
                LeakyReLU()
            ),
            nn.Sequential(
                Conv3d(8, 16, kernel_size=3, stride=1, padding=1),  # ダウンサンプリングしない
                SEResNetBlock(16, 16),
                nn.BatchNorm3d(16),
                LeakyReLU()
            ),
            nn.Sequential(
                Conv3d(16, 1, kernel_size=1, stride=1),  # チャネル次元を1に圧縮
                nn.BatchNorm3d(1),
                LeakyReLU()
            )
        )

    def forward(self, x):
        return self.blocks(x)
    

class Discriminator(nn.Module):
    """3D Discriminator for VQGAN"""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        # PatchGAN discriminator
        self.blocks = nn.Sequential(
            # 入力: 1 x 256 x 256 x 256
            Conv3d(1, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128 x 128
            LeakyReLU(0.2, inplace=True),
            
            Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64 x 64
            nn.BatchNorm3d(128),
            LeakyReLU(0.2, inplace=True),
            
            Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32 x 32
            nn.BatchNorm3d(256),
            LeakyReLU(0.2, inplace=True),
            
            Conv3d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 16 x 16 x 16
            nn.BatchNorm3d(512),
            LeakyReLU(0.2, inplace=True),
            
            Conv3d(512, 1, kernel_size=4, stride=1, padding=1),  # 1 x 15 x 15 x 15
        )
        
    def forward(self, x):
        return self.blocks(x)

class VQGAN(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    VQGAN implementation with Vector Quantization
    """
    def __init__(self, config: Optional[AutoencoderConfig] = None, beta: float = 0.25):
        super().__init__()
        if config is None:
            config = AutoencoderConfig()
            
        self.register_to_config(**config.__dict__)
        self.beta = beta
        
        # Encoder
        if config.encoder_type == "ver2":
            print("ver2のencoderを使用します")
            self.encoder = SVSEncoder_ver2(config)   
        else:
            print("ver1のencoderを使用します")
            self.encoder = SVSEncoder(config)
            
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(config, beta)
        
        # Decoder
        if config.decoder_type == "ver2":
            print("ver2のdecoderを使用します")
            self.decoder = SVSDecoder_ver2(config)
        else:
            print("ver1のdecoderを使用します")
            self.decoder = SVSDecoder(config)
            
        # Discriminator
        self.discriminator = Discriminator(config)
        
    def forward(self, x: torch.Tensor, return_dict: bool = True, generator: Optional[torch.Generator] = None):
        # Encode
        encoded = self.encoder(x)
        
        # Vector Quantization
        vq_loss, quantized, encodings = self.vector_quantizer(encoded)
        
        # Decode
        reconstructed = self.decoder(quantized, generator=generator)
        
        if return_dict:
            return {
                "reconstructed": reconstructed,
                "vq_loss": vq_loss,
                "quantized": quantized,
                "encodings": encodings,
                "encoded": encoded
            }
        return reconstructed, vq_loss, quantized, encodings
    
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        encoded = self.encoder(x)
        vq_loss, quantized, encodings = self.vector_quantizer(encoded)
        
        if return_dict:
            return {
                "latent_dist": quantized,
                "vq_loss": vq_loss,
                "encodings": encodings
            }
        return quantized, vq_loss, encodings
    
    def decode(self, z: torch.Tensor, return_dict: bool = True, generator: Optional[torch.Generator] = None):
        decoded = self.decoder(z, generator=generator)
        
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
    
    def discriminate(self, x: torch.Tensor):
        """Discriminator forward pass"""
        return self.discriminator(x)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config = AutoencoderConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config=config)
        model_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        state_dict = load_file(model_file)
        model.load_state_dict(state_dict)
        return model
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        save_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        
        # 設定ファイルの保存
        self.save_config(save_directory)
        
        # モデルの保存
        if save_serialization:
            safetensors_filename = SAFETENSORS_WEIGHTS_NAME if not variant else f"{SAFETENSORS_WEIGHTS_NAME.split('.')[0]}.{variant}.safetensors"
            weight_name = os.path.join(save_directory, safetensors_filename)
            save_file(self.state_dict(), weight_name)
        else:
            weights_filename = WEIGHTS_NAME if not variant else f"{WEIGHTS_NAME.split('.')[0]}.{variant}.bin"
            weight_name = os.path.join(save_directory, weights_filename)
            if save_function is None:
                torch.save(self.state_dict(), weight_name)
            else:
                save_function(self.state_dict(), weight_name)
    
    def save_config(self, save_directory: Union[str, os.PathLike]):
        config_dict = deepcopy(self.config.__dict__)
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        
        config_path = os.path.join(save_directory, CONFIG_NAME)
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

class SVSAE(ModelMixin,ConfigMixin,FromOriginalModelMixin):
    """
    
    FromOriginalModelMixin: 既存のモデルを読み込むためのMixin
    .ckpt または .safetensors 形式で保存された事前学習済みの重みをモデルに読み込むためのメソッドを提供するMixin
    
    """
    def __init__(self,config:Optional[AutoencoderConfig]=None):
        super().__init__()
        if config is None:
            config=AutoencoderConfig()#デフォルトの設定を使用
            
            #configに設定を登録（もしくは追加）(FlozenDictに登録)
        self.register_to_config(**config.__dict__)#キーワード引数展開
        if config.encoder_type=="ver2":
            print("ver2のencoderを使用します")
            self.encoder=SVSEncoder_ver2(config)   
        else:
            print("ver1のencoderを使用します")
            self.encoder=SVSEncoder(config)
        if config.decoder_type=="ver2":
            print("ver2のdecoderを使用します")
            self.decoder=SVSDecoder_ver2(config)
        else:
            print("ver1のdecoderを使用します")
            self.decoder=SVSDecoder(config)
    def forward(self,x:torch.Tensor,return_dict:bool=True,generator:Optional[torch.Generator]=None)->Union[torch.Tensor,dict]:
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    def encode(self,x:torch.Tensor,return_dict:bool=True)->Union[torch.Tensor,dict]:
        z=self.encoder(x)
        if return_dict:
            return {"latent_dist":z}
        return z
            
    def _decode(self,z:torch.Tensor,return_dict:bool=True)->Union[DecoderOutput,torch.Tensor]:
        x=self.decoder(z)
        if not return_dict:
            return (x,)
        return DecoderOutput(sample=x)
    
    @apply_forward_hook
    def decode(self,z:torch.Tensor,return_dict:bool=True,generator:Optional[torch.Generator]=None)->Union[torch.Tensor,dict]:
        decoded=self.decoder(z,generator=generator)
        if not return_dict:
            return (decoded,)#一要素のタプル result=(decoded,)->result[0]でdecodedを取得
        return DecoderOutput(sample=decoded)
        
    
    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path:Union[str,os.PathLike],**kwargs):
        """
        cls:クラス自身を参照するための引数(SVSAE)
        """
        #事前学習済みモデルのconfigを読み込む
        config=AutoencoderConfig.from_pretrained(pretrained_model_name_or_path)
        
        model=cls(config=config)#モデルのインスタンスを生成
        model_file=os.path.join(pretrained_model_name_or_path,"model.safetensors")
        state_dict=load_file(model_file)
        model.load_state_dict(state_dict)
        return model#事前学習済みモデルを返す
    def save_pretraind(
        self,
        save_directory:Union[str,os.PathLike],
        is_main_process:bool=True,
        save_function:Callable=None,
        save_serialization:bool=True,
        variant:Optional[str]=None,
    ):
        """
            モデルとその設定ファイルをディレクトリに保存します。

            #### 引数:
            - **`save_directory`** (`str` または `os.PathLike`):
            保存先のディレクトリ。存在しない場合は作成されます。

            - **`is_main_process`** (`bool`, *オプション*, デフォルト値は `True`):
            呼び出し元がメインプロセスかどうかを指定します。分散トレーニング時に便利です。

            - **`save_function`** (`Callable`):
            状態辞書（state dictionary）を保存するために使用する関数。分散トレーニング時にテンソルをプロセス間で同期させる際に便利です。

            - **`safe_serialization`** (`bool`, *オプション*, デフォルト値は `True`):
            モデルを `safetensors` を使用して保存するか、従来の PyTorch（`pickle` を使用）方式で保存するかを指定します。

            - **`variant`** (`str`, *オプション*):
            指定された場合、重みは `pytorch_model.<variant>.bin` という形式で保存されます。
        """ 
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory,exist_ok=True)
        
        #設定ファイルの保存
        self.save_config(save_directory)
        #モデルの保存
        if save_serialization:
            safetensors_filename=SAFETENSORS_WEIGHTS_NAME if not variant else f"{SAFETENSORS_WEIGHTS_NAME.split('.')[0]}.{variant}.safetensors"
            weight_name=os.path.join(save_directory,safetensors_filename)
            save_file(self.state_dict(),weight_name)
        else:
            weights_filename=WEIGHTS_NAME if not variant else f"{WEIGHTS_NAME.split('.')[0]}.{variant}.bin"
            weight_name=os.path.join(save_directory,weights_filename)
            if save_function is None:
                torch.save(self.state_dict(),weight_name)
            else:
                save_function(self.state_dict(),weight_name)
    def save_config(self,save_directory:Union[str,os.PathLike]):
        config_dict=deepcopy(self.config.__dict__)#ディープコピーは参照渡しを防ぐ（変更しても元のオブジェクトに影響を与えない）
        config_dict={k:v for k,v in config_dict.items() if not k.startswith('_')}#キーがアンダースコアで始まる要素を除外した新しい辞書を作成することを目的
        
        config_path=os.path.join(save_directory,CONFIG_NAME)
        with open(config_path,"w") as f:
            json.dump(config_dict,f)
            

class VQGANLoss(nn.Module):
    """VQGAN Loss Functions with improved reconstruction loss"""
    def __init__(self, config: AutoencoderConfig, perceptual_weight: float = 1.0, gan_weight: float = 1.0):
        super().__init__()
        self.config = config
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        
        # L1 Loss for reconstruction
        self.l1_loss = nn.L1Loss()
        
        # MSE Loss for reconstruction (alternative)
        self.mse_loss = nn.MSELoss()
        
        # Smooth L1 Loss for better gradients
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # BCE Loss for discriminator
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def reconstruction_loss(self, real, fake):
        """Improved reconstruction loss with multiple metrics"""
        # 複数の損失を組み合わせて再構成品質を向上
        l1 = self.l1_loss(fake, real)
        mse = self.mse_loss(fake, real)
        smooth_l1 = self.smooth_l1_loss(fake, real)
        
        # Structural Similarity component
        ssim_loss = self.structural_similarity_loss(real, fake)
        
        # NaN check for individual losses
        if torch.isnan(l1) or torch.isnan(mse) or torch.isnan(smooth_l1) or torch.isnan(ssim_loss):
            print("Warning: NaN detected in reconstruction loss components")
            return torch.tensor(1.0, device=real.device, requires_grad=True)
        
        # Combined loss with adaptive weights
        total_recon = l1 + 0.3 * mse + 0.2 * smooth_l1 + 0.1 * ssim_loss  # Reduced weights
        
        # Clamp to prevent explosion
        total_recon = torch.clamp(total_recon, min=0.0, max=10.0)
        
        return total_recon
    
    def structural_similarity_loss(self, real, fake):
        """Simple structural similarity loss for 3D data with stability checks"""
        try:
            # Normalize to [0, 1] for SSIM calculation
            real_norm = torch.clamp((real + 1) / 2, 0, 1)  # Assuming data is in [-1, 1]
            fake_norm = torch.clamp((fake + 1) / 2, 0, 1)
            
            # Calculate means
            mu_real = torch.mean(real_norm)
            mu_fake = torch.mean(fake_norm)
            
            # Calculate variances and covariance
            var_real = torch.var(real_norm)
            var_fake = torch.var(fake_norm)
            cov = torch.mean((real_norm - mu_real) * (fake_norm - mu_fake))
            
            # SSIM components with stability
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            numerator = (2 * mu_real * mu_fake + c1) * (2 * cov + c2)
            denominator = (mu_real ** 2 + mu_fake ** 2 + c1) * (var_real + var_fake + c2)
            
            # Prevent division by zero
            denominator = torch.clamp(denominator, min=1e-8)
            
            ssim = numerator / denominator
            
            # NaN check
            if torch.isnan(ssim) or torch.isinf(ssim):
                return torch.tensor(0.0, device=real.device, requires_grad=True)
            
            return torch.clamp(1 - ssim, min=0.0, max=2.0)  # Convert to loss (lower is better)
            
        except Exception as e:
            print(f"Warning: Error in SSIM calculation: {e}")
            return torch.tensor(0.0, device=real.device, requires_grad=True)
    
    def generator_loss(self, disc_fake):
        """Generator loss (fool the discriminator)"""
        # We want discriminator to output 1 for fake images
        real_labels = torch.ones_like(disc_fake)
        return self.bce_loss(disc_fake, real_labels)
    
    def discriminator_loss(self, disc_real, disc_fake):
        """Discriminator loss"""
        real_labels = torch.ones_like(disc_real)
        fake_labels = torch.zeros_like(disc_fake)
        
        real_loss = self.bce_loss(disc_real, real_labels)
        fake_loss = self.bce_loss(disc_fake, fake_labels)
        
        return (real_loss + fake_loss) / 2
    
    def discriminator_loss_with_smoothing(self, disc_real, disc_fake, smoothing=0.1):
        """Discriminator loss with label smoothing to prevent early convergence"""
        # Label smoothing: real labels become 0.9, fake labels become 0.1
        real_labels = torch.ones_like(disc_real) * (1.0 - smoothing)
        fake_labels = torch.zeros_like(disc_fake) + smoothing
        
        real_loss = self.bce_loss(disc_real, real_labels)
        fake_loss = self.bce_loss(disc_fake, fake_labels)
        
        return (real_loss + fake_loss) / 2
    
    def total_generator_loss(self, real, fake, vq_loss, disc_fake):
        """Total generator loss"""
        recon_loss = self.reconstruction_loss(real, fake)
        gen_loss = self.generator_loss(disc_fake)
        
        total_loss = recon_loss + self.perceptual_weight * vq_loss + self.gan_weight * gen_loss
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "vq_loss": vq_loss,
            "generator_loss": gen_loss
        }

def main():
    if torch.cuda.is_available():
        print("GPUが利用できます"   )
    else:
        print("GPUが利用できません")
    Acacia_Folder=r"/home/ryuichi/tree/TREE_PROJ/data_dir/svs/AcaciaClustered"
    model=SVSAE().to("cuda")
    x=np.load(Acacia_Folder+"/svs_1.npz")
    x=npz2dense(x)
    x=torch.tensor(x).float()
    x=torch.unsqueeze(x,0).to("cuda")
    print(f"xのshape:{x.shape}")
    output=model(x)
    print(f"outputのshape:{output.shape}")


def test_vqgan():
    """VQGAN test function"""
    if torch.cuda.is_available():
        print("GPUが利用できます")
        device = "cuda"
    else:
        print("GPUが利用できません")
        device = "cpu"
    
    # Create config
    config = AutoencoderConfig(device=device, encoder_type="ver2", decoder_type="ver2")
    
    # Create VQGAN model
    vqgan = VQGAN(config, beta=0.25).to(device)
    
    # Create loss function
    vqgan_loss = VQGANLoss(config, perceptual_weight=1.0, gan_weight=0.5)
    
    # Test input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 256, 256, 256).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        results = vqgan(input_tensor, return_dict=True)
        
        print(f"Reconstructed shape: {results['reconstructed'].shape}")
        print(f"VQ Loss: {results['vq_loss'].item():.4f}")
        print(f"Quantized shape: {results['quantized'].shape}")
        
        # Test discriminator
        disc_real = vqgan.discriminate(input_tensor)
        disc_fake = vqgan.discriminate(results['reconstructed'])
        
        print(f"Discriminator real output shape: {disc_real.shape}")
        print(f"Discriminator fake output shape: {disc_fake.shape}")
        
        # Test losses
        loss_dict = vqgan_loss.total_generator_loss(
            input_tensor, 
            results['reconstructed'], 
            results['vq_loss'], 
            disc_fake
        )
        
        print("Loss breakdown:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")
        
        disc_loss = vqgan_loss.discriminator_loss(disc_real, disc_fake)
        print(f"  discriminator_loss: {disc_loss.item():.4f}")
    
    print("VQGAN test completed successfully!")
        

if __name__ == "__main__":
    # main()
    # 既存のテストコード
    config=AutoencoderConfig(device="cuda")
    encoder=SVSEncoder_ver2(config).to(config.device)
    decoder=SVSDecoder_ver2(config).to(config.device)
    vq=VectorQuantizer(config,0.9).to(config.device)  # VectorQuantizerもGPUに移動
    input=torch.randn((1,1,256,256,256)).to(config.device)
    en_out=encoder(input)
    loss,vq_output,emb=vq(en_out)
    de_output=decoder(vq_output)
    
    print(f"Encoder output shape: {en_out.shape}")
    print(f"VQ loss: {loss.item():.4f}")
    print(f"VQ output shape: {vq_output.shape}")
    print(f"Decoder output shape: {de_output.shape}")
    
    print("\n" + "="*50)
    print("VQGAN Test")
    print("="*50)
    
    # VQGANのテスト
    test_vqgan()

