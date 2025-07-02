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
        #重みの初期化
        nn.init.uniform_(self.embedding.weight,-1/self.K,1/self.K)
        self.beta=beta
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
        # print(quantized.shape)
        return quantized,encodings
    def calc_loss(self,quantized,x):
        """
        Code Book Loss: q_latentLoss:埋め込みベクトルをEncoderベクトルに近づける.
        E_latent Loss : e_latentLoss:エンコーダベクトルを埋め込みベクトルに近づける
        """
        e_latent_loss=F.mse_loss(quantized.detach(),x)#Encoder側に伝わるloss
        q_latent_loss=F.mse_loss(quantized,x.detach())#Codebook側に伝わるloss
        loss =q_latent_loss+self.beta*e_latent_loss
        return loss
        
    def forward(self,x:torch.Tensor):
        x=x.permute(0,2,3,4,1).contiguous()
        quantized,encodings=self.vector_quantized(x)
        loss =self.calc_loss(quantized,x)
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
                Conv3d(16, 1, kernel_size=1, stride=1),  # チャネル次元を1に圧縮
                nn.BatchNorm3d(1),
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
    
        

if __name__ == "__main__":
    # main()
    config=AutoencoderConfig(device="cuda")
    encoder=SVSEncoder_ver2(config).to(config.device)
    decoder=SVSDecoder_ver2(config).to(config.device)
    vq=VectorQuantizer(config,0.9)
    input=torch.randn((1,1,256,256,256)).to(config.device)
    en_out=encoder(input)
    loss,vq_output,emb=vq(en_out)
    de_output=decoder(vq_output)

