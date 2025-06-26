import pytorch_lightning as pl
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import sys
sys.path.append("/home/ryuichi/tree/TREE_PROJ")
from utils import npz2dense 

class TreeDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str="/mnt/nas/rmjapan2000/tree/data_dir",
        batch_size: int=8,
        img_size: int=None,
        debug: bool=False,
    ):
        #transformsやデータの次元をクラス変数に設定する.
        super(TreeDataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_path = os.path.join(self.data_dir, "train")
        self.test_path = os.path.join(self.data_dir, "test")
        #debugモードではワーカーを1つに設定する.
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = 4

        train_dataset = voxel_sketch_Dataset(self.train_path, img_size)
        length = len(train_dataset)#
        train_length = int(length*0.8)
        val_length = length-train_length
        self.train_data, self.val_data = random_split(train_dataset, [train_length,val_length])
        self.train_data = train_dataset
        self.test_data = voxel_sketch_Dataset(self.train_path, img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )


class voxel_sketch_Dataset(Dataset):
    #スケッチとvoxelのペアを読み込む
    def __init__(self,
                 data_folder: str="/mnt/nas/rmjapan2000/tree/data_dir",
                 img_size: int = 224,
    ):
        super().__init__()
        self.img_folder = os.path.join(data_folder, "sketch_cgvi")
        self.vxl_folder = os.path.join(data_folder, "svs_cgvi")
        assert os.path.exists(self.img_folder), f"path '{self.img_folder}' does not exists."
        assert os.path.exists(self.vxl_folder), f"path '{self.vxl_folder}' does not exists."

        img_names = [i for i in os.listdir(self.img_folder) if i.endswith(".png")]
        self.img_list = [os.path.join(self.img_folder, i) for i in img_names]
        vxl_names = [i for i in os.listdir(self.vxl_folder) if i.endswith(".npz")]
        self.vxl_list = [os.path.join(self.vxl_folder, i) for i in vxl_names]

        self.transform = transforms.Compose([
            # transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB if needed
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
      

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        if not os.path.exists(self.img_list[index]):
            print(f"[ERROR] File does not exist: {self.img_list[index]}")
            # 安全に次のindexに回す（リスト末尾なら先頭に戻る）
            return self.__getitem__((index + 1) % len(self.img_list))
    
        img = Image.open(self.img_list[index])
        
        data["img"] = self.transform(img)
        voxel = np.load(self.vxl_list[index])
        voxel=npz2dense(voxel,64,64,64)
        voxel[voxel==0]=-1.0
        voxel[voxel==1]=0.0
        voxel[voxel==1.5]=0.5
        voxel[voxel==2]=1.0
        voxel=torch.tensor(voxel).float()
        data["voxel"] = voxel
        return data

class direction_TreeDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str="/mnt/nas/rmjapan2000/tree/data_dir",
        batch_size: int=8,
        img_size: int=224,
        debug: bool=False,
    ):
        #transformsやデータの次元をクラス変数に設定する.
        super(direction_TreeDataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_path = os.path.join(self.data_dir, "train")
        self.test_path = os.path.join(self.data_dir, "test")
        #debugモードではワーカーを1つに設定する.
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = 4

        back_train_dataset = direction_voxel_sketch_Dataset(self.train_path, img_size, "back")
        front_train_dataset = direction_voxel_sketch_Dataset(self.train_path, img_size, "front")
        left_train_dataset = direction_voxel_sketch_Dataset(self.train_path, img_size, "left")
        right_train_dataset = direction_voxel_sketch_Dataset(self.train_path, img_size, "right")
        train_dataset = torch.utils.data.ConcatDataset([back_train_dataset, front_train_dataset, left_train_dataset, right_train_dataset])
        length = len(train_dataset)#
        train_length = int(length*0.75)
        val_length = length-train_length
        self.train_data, self.val_data = random_split(train_dataset, [train_length,val_length])
        self.train_data = train_dataset
        self.test_data = voxel_sketch_Dataset(self.train_path, img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )


class direction_voxel_sketch_Dataset(Dataset):
    #スケッチとvoxelのペアを読み込む
    def __init__(self,
                 data_folder: str="/mnt/nas/rmjapan2000/tree/data_dir",
                 img_size: int = 224,
                 direction: str = "back",
    ):
        super().__init__()
        self.img_folder = os.path.join(data_folder, "sketch_cgvi_ver2", direction)
        self.vxl_folder = os.path.join(data_folder, "svs_cgvi_ver2", direction)
        assert os.path.exists(self.img_folder), f"path '{self.img_folder}' does not exists."
        assert os.path.exists(self.vxl_folder), f"path '{self.vxl_folder}' does not exists."

        img_names = [i for i in os.listdir(self.img_folder) if i.endswith(".png")]
        self.img_list = [os.path.join(self.img_folder, i) for i in img_names]
        vxl_names = [i for i in os.listdir(self.vxl_folder) if i.endswith(".npz")]
        self.vxl_list = [os.path.join(self.vxl_folder, i) for i in vxl_names]

        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB if needed
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
      

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        if not os.path.exists(self.img_list[index]):
            print(f"[ERROR] File does not exist: {self.img_list[index]}")
            # 安全に次のindexに回す（リスト末尾なら先頭に戻る）
            return self.__getitem__((index + 1) % len(self.img_list))
    
        img = Image.open(self.img_list[index])
        
        data["img"] = self.transform(img)
        voxel = np.load(self.vxl_list[index%len(self.vxl_list)])
        voxel=npz2dense(voxel,64,64,64)
        voxel[voxel==0]=-1.0
        voxel[voxel==1]=0.0
        voxel[voxel==1.5]=0.5
        voxel[voxel==2]=1.0
        voxel=torch.tensor(voxel).float()
        data["voxel"] = voxel
        return data

