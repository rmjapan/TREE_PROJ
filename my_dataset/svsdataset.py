import os
import numpy as np
import torch
from utils import npz2dense
from data_augmentation.aug_trans import *
from data_augmentation.aug_flip import *
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split

class SvsDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/mnt/nas/rmjapan2000/tree/data_dir/train/svd_0.2",
        batch_size: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir= data_dir
        self.num_workers=4

    
    def setup(self,stage=None):
        if stage == "fit" or stage is None:
            full_dataset=SvsDataset(self.data_dir)
            total_length = len(full_dataset)
            train_length = int(total_length * 0.8)
            val_length = total_length - train_length
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_length,val_length],
                generator=torch.Generator().manual_seed(42)
                )
            
    def prepare_data(self):
        pass
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 訓練時はシャッフル
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 検証時はシャッフルしない
            num_workers=self.num_workers
        )               

class SvsDataset(Dataset):
    def __init__(self,data_dir=None,transform=None):
        if data_dir is None:
            self.data_dir="/mnt/nas/rmjapan2000/tree/data_dir/train/svd_0.2"
        else:
            self.data_dir=data_dir
        self.transform=transform
    def __len__(self):
        return len(os.listdir(self.data_dir))
    def __getitem__(self,idx):
        svs=np.load(self.data_dir+f"/svs_{idx+1}.npz")
        svs=npz2dense(svs)
        svs[svs==0]=-1.0
        svs[svs==1]=0.0
        svs[svs==1.5]=0.5
        svs[svs==2]=1.0
        svs=torch.tensor(svs).float()
        svs=torch.unsqueeze(svs,0)
        return svs
class SvsDataset_aug_trans(Dataset):
    def __init__(self,data_dir=None,transform=None):
        if data_dir is None:
            self.data_dir="/mnt/nas/rmjapan2000/tree/data_dir/train/svd_0.2"
        else:
            self.data_dir=data_dir
        self.transform=transform
    def __len__(self):
        return len(os.listdir(self.data_dir))
    def __getitem__(self,idx):
        svs=np.load(self.data_dir+f"/svs_{idx+1}.npz")
        svs=npz2dense(svs)
        svs[svs==0]=-1.0
        svs[svs==1]=0.0
        svs[svs==1.5]=0.5
        svs[svs==2]=1.0
        svs=torch.tensor(svs).float()
        translation_vector=torch.randint(-10,10,(1,3))
        svs=translate_voxel(svs,translation_vector)
        svs=torch.tensor(svs).float()
        svs=torch.unsqueeze(svs,0)
        return svs
class SvsDataset_aug_flip(Dataset):
    def __init__(self,data_dir=None,transform=None):
        if data_dir is None:
            self.data_dir="/mnt/nas/rmjapan2000/tree/data_dir/train/svd_0.2"
        else:
            self.data_dir=data_dir
        self.transform=transform
    def __len__(self):
        return len(os.listdir(self.data_dir))
    def __getitem__(self,idx):
        svs=np.load(self.data_dir+f"/svs_{idx+1}.npz")
        svs=npz2dense(svs)
        svs[svs==0]=-1.0
        svs[svs==1]=0.0
        svs[svs==1.5]=0.5
        svs[svs==2]=1.0
        svs=torch.tensor(svs).float()
        svs=torch.unsqueeze(svs,0)
        class_label=torch.randint(0,3,(1,))
        if class_label==0:
            svs=flip_column(svs)
        elif class_label==1:
            svs=flip_row(svs)
        elif class_label==2:
            svs=flip_depth(svs)
        return svs
            
            
     
            
        

    
    