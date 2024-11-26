import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码实现了一个 图注意力层（Graph Attention Layer, GAT），用于处理图结构数据。
# 具体来说，这个实现支持 2D 和 3D 特征图的交互操作，并允许用户自定义是否包括自节点、额外加权或线性变换。


class GraphAttentionLayer(nn.Module):
    # in_features 和 out_features:
    # 1. 输入和输出特征维度。
    # 2. 输入维度决定了每个节点的特征数量，输出维度是该层的特征输出大小。
    #
    # dropout:
    # 在计算注意力得分时的随机丢弃比例，用于防止过拟合。
    #
    # alpha:
    # LeakyReLU 激活函数的负斜率，用于引入非线性。
    #
    # concat:
    # 是否对输出应用激活函数 F.elu。
    #
    # include_self:
    # 决定是否在注意力机制中包含自节点的特征（self-loop）。
    #
    # additional:
    # 如果设置为 True，会将 3D 节点的特征添加到输出特征中。
    #
    # with_linear_transform:
    # 决定是否在输出前对节点特征进行线性变换（通过权重 W）。
    def __init__(
        self, 
        in_features, 
        out_features, 
        dropout, 
        alpha, 
        concat=True, 
        include_self=True,
        additional=False,
        with_linear_transform=True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 权重和注意力机制的初始化
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.include_self = include_self
        self.with_linear_transform = with_linear_transform
        self.additional = additional

    # h_2d: 2D 特征图，形状 [batch_size, num_2d_nodes, in_features]。
    # h_3d: 3D 特征图，形状 [batch_size, num_3d_nodes, in_features]。
    def forward(self, h_2d, h_3d):
        b, n1, dim = h_3d.shape
        b, n2, dim = h_2d.shape
        num_leaf = int(n2 / n1)

        # 线性变换
        wh_2d = torch.matmul(h_2d, self.W)
        wh_3d = torch.matmul(h_3d, self.W)

        # 注意力计算
        e = self._prepare_attentional_mechanism_input(wh_2d, wh_3d, num_leaf, self.include_self)
        attention = F.softmax(e, dim=2)
        # 调用 _prepare_attentional_mechanism_input 计算 2D 和 3D 特征的注意力权重。

        h_2d = torch.reshape(h_2d, (b, n1, num_leaf, dim))
        wh_2d = torch.reshape(wh_2d, (b, n1, num_leaf, dim))
        if self.include_self:
            wh_2d = torch.cat(
                [wh_3d.unsqueeze(-2), wh_2d], dim=-2
            ) # [b, N, 1+num_leaf, d_out]
            h_2d = torch.cat(
                [h_3d.unsqueeze(-2), h_2d], dim=-2
            )

            # 特征组合
            # 利用注意力权重将 2D 和 3D 特征融合
            if self.with_linear_transform:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d)
            else:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d)

            if self.additional:
                h_prime = h_prime + h_3d
        else:
            if self.with_linear_transform:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d) / 2. + wh_3d
            else:
                h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d) / 2. + h_3d

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # 注意力机制:
    # 计算注意力系数 e，由 2D 和 3D 特征的线性变换的加和表示。
    # 使用 LeakyReLU 处理，保证非线性。
    def _prepare_attentional_mechanism_input(self, wh_2d, wh_3d, num_leaf, include_self=False):
        b, n1, dim = wh_3d.shape
        b, n2, dim = wh_2d.shape

        wh_2d_ = torch.matmul(wh_2d, self.a[:self.out_features, :]) # [b, N2, 1]
        wh_2d_ = torch.reshape(wh_2d_, (b, n1, num_leaf, -1)) # [b, n1, 6, 1]
        wh_3d_ = torch.matmul(wh_3d, self.a[self.out_features:, :]) # [b, N1, 1]

        if include_self:
            wh_2d_ = torch.cat(
                [wh_3d_.unsqueeze(2), wh_2d_], dim=-2
            ) # [b, N1, 1 + num_leaf, 1]

        e = wh_3d_.unsqueeze(2) + wh_2d_
        return self.leakyrelu(e)
