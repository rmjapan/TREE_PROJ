# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    #お気持ち：小さいテンソルを大きいテンソルと同じサイズに拡張する
    #同じ次元ならそのまま返される
    #other.dim()=３
    #src.dim()=1
    #dim=-1の場合の例
    if dim < 0:
        dim = other.dim() + dim# dim=2
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    #src.dim()=3

    #srcがOtherと同じサイズになるように末尾を拡張
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)

    #srcをotherと同じサイズに拡張
    src = src.expand_as(other)
    return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    #indexをsrcと同じサイズに拡張
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        #dim_sizeが存在する場合
        if dim_size is not None:
            size[dim] = dim_size
        #indexのテンソル要素数（index.numel()）が0の場合
        elif index.numel() == 0:
            size[dim] = 0
        #indexのテンソル要素数が0でない場合
        else:
            size[dim] = int(index.max()) + 1
        #sizeが
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
				weights: Optional[torch.Tensor] = None,
				out: Optional[torch.Tensor] = None,
				dim_size: Optional[int] = None) -> torch.Tensor:
    #weightsが存在する場合（デフォルトNone）
    if weights is not None:
        src = src * broadcast(weights, src, dim)
    #srcをindexに従って足し合わせる
    out = scatter_add(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    if weights is None:
        weights = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_add(weights, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out
