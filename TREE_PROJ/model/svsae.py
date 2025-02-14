

import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
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
from diffusers.models.autoencoders.vae import DecoderOutput, BaseOutput
from safetensors.torch import load_file, save_file
from typing import Callable, Optional, Union
from copy import deepcopy
from diffusers.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME

class AutoencoderConfig(ConfigMixin):
    config_name="autoencoder_config"#必須
    def __init__(
        self,
        in_channels:int=1,
        out_channels:int=1,
        block_out_channels:Tuple[int]=(1,1),
        act_fn:str=["leakyrelu","sigmoid"],
        layers_per_block:int=3,#ブロックごとのレイヤー数(Conv3d+Batch+Act_fn-> 3)
        latent_channels:int=1,
        sample_size:int=256,
        scaling_factor:float=0.09366,#what is this?
    ):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.block_out_channels=block_out_channels
        self.latent_channels=latent_channels
        self.scaling_factor=scaling_factor
        self.sample_size=sample_size
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

class SVSEncoder(torch.nn.Module):
    def __init__(self,config:AutoencoderConfig):
        super().__init__()
        self.config=config
        c_hid=config.block_out_channels
        
        self.conv1=Conv3d(in_channels=config.in_channels,out_channels=c_hid[0],kernel_size=4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm3d(c_hid[0])
        self.conv2=Conv3d(in_channels=c_hid[0],out_channels=c_hid[1],kernel_size=4,stride=2,padding=1)
        self.bn2 = nn.BatchNorm3d(c_hid[1])
        self.conv3=Conv3d(in_channels=c_hid[1],out_channels=config.latent_channels,kernel_size=4,stride=2,padding=1)
        self.bn3 = nn.BatchNorm3d(config.latent_channels)
        self.relu=LeakyReLU()
        
    def forward(self,x):
        x_256=self.conv1(x)
        x_256=self.bn1(x_256)
        x_256=self.relu(x_256)
        x_128=self.conv2(x_256)
        x_128=self.bn2(x_128)
        x_128=self.relu(x_128)
        x_64=self.conv3(x_128)
        x_64=self.bn3(x_64)
        x_64=self.relu(x_64)
        return x_64
    
class SVSDecoder(torch.nn.Module):
    def __init__(self,config:AutoencoderConfig):
        super().__init__()
        c_hid=config.block_out_channels#隠れ層のチャンネル数
        self.conv1=ConvTranspose3d(in_channels=config.latent_channels,out_channels=c_hid[1],kernel_size=4,stride=2,padding=1)
        self.bn1=nn.BatchNorm3d(c_hid[1])
        self.conv2=ConvTranspose3d(in_channels=c_hid[1],out_channels=c_hid[0],kernel_size=4,stride=2,padding=1)
        self.bn2=nn.BatchNorm3d(c_hid[0])
        self.conv3=ConvTranspose3d(in_channels=c_hid[0],out_channels=config.out_channels,kernel_size=4,stride=2,padding=1)
        self.bn3=nn.BatchNorm3d(config.out_channels)
        self.relu=LeakyReLU()
        self.tanh=Tanh()
    def forward(self,x:torch.Tensor,generator:Optional[torch.Generator]=None)->torch.Tensor:
        x_128=self.conv1(x)
        x_128=self.bn1(x_128)
        x_128=self.relu(x_128)
        x_256=self.conv2(x_128)
        x_256=self.bn2(x_256)
        x_256=self.relu(x_256)
        x_512=self.conv3(x_256)
        x_512=self.bn3(x_512)
        x_512=self.tanh(x_512)
        return x_512
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
            self.encoder=SVSEncoder(config)
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
    main()
    

