"""

木の3dボクセルから木の3dボクセルを出力するオートエンコーダモデルの入力から出力までのテストをする関数はここで作成する

オートエンコーダ：
入力(Input)：木の3dボクセル（256,256,256）
中間出力(Encoder)：木の3dボクセル（64,64,64）
出力(Decoder)：木の3dボクセル（256,256,256）
"""

import torch
from model.svsae import SVSAE#基本的にはmodelフォルダー内でモデルを定義している
from my_dataset.svsdataset import SvsDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from visualize_func import visualize_voxel_data,visualize_and_save_volume
import random
from svs import voxel_distribution
import torch.nn.functional as nn
from datetime import datetime


def load_model(model_path=None):
    """
    事前学習済みの重みをモデルにロードする
    「Pytorch入門7. モデルの保存と読み込み」にチュートリアルベースでモデルの読み込みに関するあれこれについて書いてあるからおすすめ！
    """
    if model_path is None:
        model_path=r"/home/ryuichi/tree/TREE_PROJ/data_dir/model2025-02-25/model_0.pth"

    
    #1:モデルの初期化(nn.Moduleを継承したモデルをcudaというGPUに載せることでGPUによる高速な計算が可能になる)
    model = SVSAE().to("cuda")

    #2:事前学習済みの重みをモデルにロードする
    model.load_state_dict(torch.load(model_path))

    #3:モデルを評価モードにする(BatchNormalizationやDropoutの動作を変えるため)
    model.eval()#BN:バッチ統計量→移動平均,Dropout:ドロップアウト率→0

    return model


def create_subset(dataset,num_data,random_index=True):
    """
    データセットのサブセットを作成する
    """
    total_size=len(dataset)
    
    #もしnum_dataがtotal_sizeより大きい場合は全データを返す
    if num_data>=total_size:
        return dataset
    else:
        if random_index:
            #ランダムにデータセットを選択
            indices=random.sample(range(total_size),num_data)
            return Subset(dataset,indices)
        else:
            #最初からnum_data個のデータを返す
            return Subset(dataset,range(num_data))
        

def prepare_dataset(dataset_path=None):
    """
    データセットクラスに関しては
    「Pytorch入門 2: データセットとデータローダー（Datasets & DataLoaders）」と
    データの前処理関連の
    「Pytorch入門 3: データ変換(Transform)」がおすすめ
    """
    if dataset_path is None:
        dataset_path=r"/mnt/nas/rmjapan2000/tree/data_dir/svd_0.2"
    dataset = SvsDataset(dataset_path)#データの読み込みや基本的な前処理関係
    return dataset
def prepare_dataloader(dataset, batch_size):
    """
    データローダーを作成する
    データをバッチサイズ毎に読み込んだり,後は順番をシャッフルしてくれたりなどしてくれる機能
    「Pytorch入門 2: データセットとデータローダー（Datasets & DataLoaders）」がおすすめ
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#離散値への変換
def svs_change_value(voxel_data):
    """
    0.8<=x<=1.0　⇒1.0(幹)
    0.4<=x<0.8　⇒0.5(枝)
    0<=x<0.4　⇒0.0(葉)
    その他　⇒-1.0(空白)
    """
    voxel_data=torch.where(voxel_data>=0.8,torch.tensor(1.0,device=voxel_data.device),
                           torch.where(voxel_data>=0.4,torch.tensor(0.5,device=voxel_data.device),
                                       torch.where(voxel_data>=0,torch.tensor(0.0,device=voxel_data.device),
                                                   torch.tensor(-1.0,device=voxel_data.device))))
    return voxel_data
def test_svsae(numdata=0,random_index=True):
    """
    モデルのテストを行う
    """
    model=load_model()
    dataset=prepare_dataset()
    dataloader=None
    #もしnumdataが0より大きい場合はサブセットを作成
    if numdata>0:
        subset=create_subset(dataset,num_data=numdata,random_index=random_index)
        dataloader=prepare_dataloader(subset,batch_size=1)
    else:
        dataloader=prepare_dataloader(dataset,batch_size=1)

    
    for idx,voxeldata in enumerate(dataloader):
        print(voxeldata.shape)
        input_voxel=voxeldata
        input_voxel=input_voxel.to("cuda")
        #Encoderの出力
        latent_voxel=model.encode(input_voxel,return_dict=False)
        print(latent_voxel.shape)
    
        #Decoderの出力
        output_voxel=model.decode(latent_voxel,return_dict=False)[0]
        print(output_voxel.shape)



        # 入力、潜在表現、出力の可視化と保存
        base_dir = r"/home/ryuichi/tree/TREE_PROJ/data_dir/visualize"
        date=datetime.now().strftime("%Y-%m-%d")
        save_dir = f"{base_dir}/{date}"
        visualize_and_save_volume(input_voxel, f"{save_dir}/input_voxel_{idx}.png")
        visualize_and_save_volume(latent_voxel, f"{save_dir}/latent_voxel_{idx}.png")
        visualize_and_save_volume(output_voxel, f"{save_dir}/output_voxel_{idx}.png")



if __name__=="__main__":
    numdata=10
    random_index=True
    test_svsae(numdata=numdata,random_index=random_index)