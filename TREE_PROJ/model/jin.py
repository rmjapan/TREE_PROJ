from re import L
from diffusers import AutoencoderKL
import torch
from torch.nn import Conv3d, ConvTranspose3d, ReLU,Sigmoid,LeakyReLU
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from train_svsae import dense2sparse, npz2dense
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
    config_name = "autoencoder_config"
    # @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 2,
        act_fn: str = "relu",
        latent_channels: int = 4,
        sample_size: int = 32,
        # scaling_factor: float = 0.18215,
        scaling_factor: float = 0.09366,
        # scaling_factor: float = 1.0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.sample_size = sample_size
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.__dict__, f)
class encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        c_hid = config.block_out_channels[0]
        self.net = nn.Sequential(
            nn.Conv2d(config.in_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 512x512 => 256x256
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 256x256 => 128x128
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.ReLU(),
            nn.Conv2d(4 * c_hid, config.latent_channels, kernel_size=3, padding=1),  # 4*c_hid => latent_dim_channels
            nn.ReLU(),
        )
    def forward(self, x) -> torch.Tensor:
        return self.net(x)
class decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.ac_fn = nn.LeakyReLU(0.2)
        c_hid = config.block_out_channels[0]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(config.latent_channels, 4 * c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 64x64 => 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 128x128 => 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, stride=1, padding=1),  # 256x256 => 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 256x256 => 512x512
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),  # 512x512 => 512x512
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid, config.out_channels, kernel_size=3, stride=1, padding=1),
            # 512x512 => 512x512 (adjust the channels)
            nn.Tanh(),  # The input image is scaled between -1 and 1, hence the output has to be bounded as well
            # nn.ReLU(),
        )
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # x = self.conv_in(x)
        # for block in self.blocks:
        #     x = block(x)
        # return self.conv_out(x)
        x = self.net(x)
        return x
class Autoencoder(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    config_class = AutoencoderConfig
    # @register_to_config
    def __init__(self, config: Optional[AutoencoderConfig] = None):
        super().__init__()
        if config is None:
            config = AutoencoderConfig()
        self.register_to_config(**config.__dict__)
        self.encoder = encoder(config)
        self.decoder = decoder(config)
    def forward(self, x: torch.Tensor, return_dict: bool = True, generator: Optional[torch.Generator] = None) -> Union[torch.Tensor, dict]:
        x = self.encoder(x)
        x = self.decoder(x)
        if x.dim() != 4:
            raise ValueError(f"Expected output to have 4 dimensions, but got {x.dim()} dimensions")
        return x
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, dict]:
        z = self.encoder(x)
        if return_dict:
            return {"latent_dist": z}
        return z
    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True, generator: Optional[torch.Generator] = None) -> Union[torch.Tensor, dict]:
        decoded = self.decoder(z, generator=generator)
        if decoded.dim() != 4:
            raise ValueError(f"Expected decoded output to have 4 dimensions, but got {decoded.dim()} dimensions")
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config = AutoencoderConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        model = cls(config=config)
        model_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        state_dict = load_file(model_file)
        model.load_state_dict(state_dict)
        return model
    # def save_pretrained(self, save_directory: Union[str, os.PathLike]):
    #     os.makedirs(save_directory, exist_ok=True)
    #     self.config.save_pretrained(save_directory)
    #     model_file = os.path.join(save_directory, "model.safetensors")
    #     save_file(self.state_dict(), model_file)
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        Save the model and its configuration file to a directory.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training to sync tensors
                across processes.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        # Save the config
        self.save_config(save_directory)
        # Save the model
        if safe_serialization:
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
        print(f"Model weights saved in {weight_name}")
    def save_config(self, save_directory: Union[str, os.PathLike]):
        """Save the configuration of the autoencoder."""
        config_dict = deepcopy(self.config.__dict__)
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        config_path = os.path.join(save_directory, CONFIG_NAME)
        with open(config_path, "w") as f:
            json.dump(config_dict, f)