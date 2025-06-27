

from model.svsae import AutoencoderConfig
from lightning import LightningModule
from Octree.octfusion.datasets import dataloader


def vq_train_loop()
def setup_dataset
class VQVAE(LightningModule):
    def __init__(self,model_config):
        super().__init__()
    def training_step(self,batch,batch_idx):
        #dataloaderをまず実装する必要があるね.
        
        #損失計算    
        
        
        
        
def main()
    setup_dataset(dataset_path)
    train_data=train_dataset
    test_data= test_dataset
    transforms=Mytransform
    dataloader=Mydataloader
    model=
    vq_train_loop()#ここで訓練する
    vq_test_loop()
    vq_save_model()