import torch
import numpy as np
import torch.nn.functional as F
from utils import process_tree_voxels, batch_cdist, discretize_voxel, discretize_voxel_ste



def weighted_mae_loss(output, target):
    """
    Weighted Mean Absolute Error loss using torch.nn.functional for faster computation
    本来あるべき箇所にある要素が無い場合に損失を大きくしたい。
    本来は葉っぱがあるべきなのに、葉っぱじゃないものがあるときも損失を大きくしたいよね。
    本来は枝があるべきなのに、枝じゃないものがあるときも損失を大きくしたいよね。
    本来は幹があるべきなのに、幹じゃないものがあるときも損失を大きくしたいよね。
    本来は空白の場所に空白じゃないものがあるときも損失を大きくしたいよね。
    
    """

    weights = torch.ones_like(target)
    weights[target < 0] = 100#空白
    weights[(target >= 0) & (target < 0.4)] = 140
    weights[(target >= 0.4) & (target < 0.8)] = 450
    weights[target >= 0.8] = 250
    
    # F.l1_lossを使用して絶対誤差を計算し、重みを適用
    loss = (weights * F.l1_loss(output, target, reduction='none')).mean()
    return loss

def weighted_mae_loss_with_quantization(output, target):
    """
    
    """
    weights = torch.ones_like(target)
    weights[target < 0] = 1#空白
    weights[(target >= 0) & (target < 0.4)] = 500
    weights[(target >= 0.4) & (target < 0.8)] = 500
    weights[target >= 0.8] = 50
    
    quantized_output = discretize_voxel_ste(output)

    for i in range(quantized_output.size(0)):
        quantized_output[i] = discretize_voxel(output[i])
    
    loss = (weights * F.l1_loss(quantized_output, target, reduction='none')).mean()
    return loss

def chamfer_distance(p1, p2, 
                    voxel_size=1.0,
                    max_points_per_part=2000,
                    base_weights={'trunk':0.1, 'branch':1.0, 'leaf':1.0}):
    """
    """
    total_loss = 0.0
    batch_size = p1.size(0)
    
    for b in range(batch_size):
        #座標を取得する
        pts1 = process_tree_voxels(p1[b:b+1], max_points_per_part)
        pts2 = process_tree_voxels(p2[b:b+1], max_points_per_part)
        
        part_loss = 0.0
        for part in ['trunk', 'branch', 'leaf']:
            if pts1[part] is None or pts2[part] is None:
                continue
                
            # 動的重み計算
            def calc_weight(pts):
                return len(pts) / max_points_per_part if len(pts) > 0 else 0.0
            weight_factor = 0.5 * (calc_weight(pts1[part]) + calc_weight(pts2[part]))
            part_weight = base_weights[part] * (1.0 + weight_factor)
            
            # 距離計算
            dist = batch_cdist(pts1[part], pts2[part])
            min_dist1 = torch.min(dist, dim=1)[0].mean()
            min_dist2 = torch.min(dist, dim=0)[0].mean()
            
            part_loss += part_weight * (min_dist1 + min_dist2) / 2
            
        total_loss += part_loss * voxel_size
        
    return total_loss / batch_size

def voxel_miou(pred_continuous, target_discrete):
    """
    Improved mIoU calculation for voxel data
    
    pred_continuous: モデルの連続値出力 (形状: [B,1,H,W,D])
    target_discrete: 離散化済み正解データ (値: -1.0, 0.0, 0.5, 1.0)
    
    Returns: 1 - mIoU (as a loss value)
    """
    # 予測値の離散化（しきい値に基づく分類）
    pred_discrete = discretize_voxel(pred_continuous)
    
    # 空白以外の部分のマスク（木の部分だけを対象にする）
    non_empty_pred = (pred_discrete >= 0.0)
    non_empty_target = (target_discrete >= 0.0)
    
    # クラス定義
    class_values = [1.0, 0.5, 0.0]  # trunk, branch, leaf
    class_weights = [3.0, 2.0, 1.0]  # 重要度に基づいた重み付け
    num_classes = len(class_values)
    
    # 予測と正解で木の部分（空白以外）が存在しない場合のチェック
    total_pred_non_empty = non_empty_pred.sum()
    total_target_non_empty = non_empty_target.sum()
    
    # 両方とも木の部分が存在しない場合は完全一致とみなす
    if total_pred_non_empty == 0 and total_target_non_empty == 0:
        return torch.tensor(0.0, device=pred_continuous.device)
    
    # 片方だけ木の部分が存在しない場合は完全不一致
    if total_pred_non_empty == 0 or total_target_non_empty == 0:
        return torch.tensor(1.0, device=pred_continuous.device)
    
    total_weight = sum(class_weights)
    weighted_iou_sum = 0.0
    
    for cls_idx, cls_val in enumerate(class_values):
        # 予測と正解のマスク作成
        pred_mask = (pred_discrete == cls_val).float()
        target_mask = (target_discrete == cls_val).float()
        
        # IoUの計算
        intersection = (pred_mask * target_mask).sum(dim=(1,2,3,4))
        union = torch.clamp((pred_mask + target_mask), 0, 1).sum(dim=(1,2,3,4))
        
        # マスクが存在しない場合のチェック
        valid_mask = (union > 0)
        
        # 有効なサンプルのみでIoUを計算
        if valid_mask.sum() > 0:
            iou = (intersection[valid_mask] / (union[valid_mask] + 1e-6)).mean()
        else:
            # このクラスが両方に存在しない場合は、完全一致とみなす
            iou = torch.tensor(1.0, device=pred_continuous.device)
        
        # 重み付け
        weighted_iou_sum += iou * class_weights[cls_idx]
    
    # 重み付き平均IoU
    weighted_miou = weighted_iou_sum / total_weight
    
    # 損失として返す (1 - mIoU)
    return 1.0 - weighted_miou