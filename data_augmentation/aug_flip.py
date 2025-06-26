#水平反転
import torch

#奥行反転(奥行軸反転)
def flip_depth(tensor):
    return torch.flip(tensor, dims=[0])

#行軸（横軸）反転
def flip_row(tensor):
    return torch.flip(tensor, dims=[1])

#列軸（縦軸）反転
def flip_column(tensor):
    return torch.flip(tensor, dims=[2])















