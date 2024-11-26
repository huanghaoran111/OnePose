import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    # 在分类任务中，如果某个类别的样本占比远高于其他类别（类别不平衡），传统的交叉熵损失函数可能会过度关注多数类，而忽视少数类
    # Focal Loss 是一种改进的损失函数，通过动态调整难样本的权重，减小易分类样本对损失的影响，从而缓解类别不平衡问题
    def __init__(self, alpha=1, gamma=2, neg_weights=0.5, pos_weights=0.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha                              # 平衡因子，用于调整正负样本的权重
        self.gamma = gamma                              # 聚焦因子，越大则对易分类样本的惩罚越小
        # 分别控制负样本和正样本对损失的贡献比例
        self.neg_weights = neg_weights
        self.pos_weights = pos_weights

    def forward(self, pred, target):
        # 正样本损失: 适用于目标值为 1, 公式: -alpha * (1 - p) ^ gamma * log(p)
        loss_pos = - self.alpha * torch.pow(1 - pred[target==1], self.gamma) * (pred[target==1]).log()
        # 负样本损失: 适用于目标值为 0, 公式: -(1 - alpha) * p ^ gamma * log(1 - p)
        loss_neg = - (1 - self.alpha) * torch.pow(pred[target==0], self.gamma) * (1 - pred[target==0]).log()

        assert len(loss_pos) != 0 or len(loss_neg) != 0, 'Invalid loss.'
        # operate mean operation on an empty list will lead to NaN
        # 空样本处理: 如果正样本或负样本为空（例如批次内没有对应类别的样本），直接返回剩余部分的平均损失，避免计算空列表的均值导致 NaN
        if len(loss_pos) == 0:
            loss_mean = self.neg_weights * loss_neg.mean()
        elif len(loss_neg) == 0:
            loss_mean = self.pos_weights * loss_pos.mean()
        else:
            # 合并损失: 加权组合正负样本的损失
            loss_mean = self.pos_weights * loss_pos.mean() + self.neg_weights * loss_neg.mean()
        
        return loss_mean