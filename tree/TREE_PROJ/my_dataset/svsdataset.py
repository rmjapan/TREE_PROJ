import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import npz2dense
class SvsDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir
        self.transform=transform
    def __len__(self):
        return len(os.listdir(self.data_dir))
    def __getitem__(self,idx):
        svs=np.load(self.data_dir+f"/svs_{idx+1}.npz")
        # print(f"svs_{idx+1}.npzを読み込みました")
        svs=npz2dense(svs)
        svs[svs==0]=-1.0
        svs[svs==1]=0.0
        svs[svs==1.5]=0.5
        svs[svs==2]=1.0
        
        svs=torch.tensor(svs).float()
        svs=torch.unsqueeze(svs,0)
        return svs
    