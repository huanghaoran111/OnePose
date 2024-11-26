# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    # 执行非极大值抑制（NMS）以过滤掉过于密集的关键点，保留局部的显著点
    assert(nms_radius >= 0)

    # 最大池化
    def max_pool(x):
        # 使用 max_pool2d 在局部窗口内执行最大池化操作，找到局部最大值
        # 窗口大小为 nms_radius*2 + 1，窗口中心即为当前像素点
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)       # 掩码初始化
    for _ in range(2):
        # 抑制非最大值的周围点，并重新计算局部最大值，进一步精细化选择。
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    # 将非极大值位置的分数置零，返回抑制后的分数
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    # 删除距离图像边界太近的关键点，避免边界点影响后续处理

    # 高度和宽度掩码: 对关键点的 x 和 y 坐标分别检查是否在边界范围内
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w

    # 保留在边界范围内的关键点和对应的分数
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    #  筛选出置信分数最高的 k 个关键点
    if k >= len(keypoints):
        # 如果关键点数量小于  k，直接返回所有关键点和分数
        return keypoints, scores
    # 使用 torch.topk 对分数排序，获取前 k 个分数及其对应索引
    scores, indices = torch.topk(scores, k, dim=0)
    # 根据索引选出对应的关键点
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    # 采样描述符: 根据关键点的位置，从密集描述符图中采样对应的描述符
    b, c, h, w = descriptors.shape

    # 关键点归一化
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    # 将关键点坐标从像素尺度映射到特征图尺度
    # 进一步归一化到 [−1,1]，以便后续的双线性插值

    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

# 这段代码实现了 SuperPoint 模型，这是一个用于图像关键点检测和描述符提取的深度学习模型
# SuperPoint 模型在输入图像中检测关键点，并为这些关键点生成描述符
# 这些关键点和描述符可以用于诸如特征匹配、SLAM 和 3D 重建等任务。

class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    # SuperPoint 模型包含两个主要部分
    # 关键点检测器：找到图像中的显著点（关键点），并计算每个点的置信分数
    # 描述符生成器：为每个关键点生成一个高维特征向量，用于匹配任务

    # default_config 定义了模型的默认参数
    default_config = {
        'descriptor_dim': 256,              # 描述符的维度（默认为 256）
        'nms_radius': 4,                    # 非极大值抑制的半径，用于过滤关键点
        'keypoint_threshold': 0.005,        # 关键点置信分数的阈值
        'max_keypoints': -1,                # 最多保留的关键点数量 -1表示不限制
        'remove_borders': 4,                # 删除距离图像边缘太近的关键点的像素数
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # convPa 和 convPb 负责生成关键点的密集置信分数图
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # convDa 和 convDb 负责生成关键点的密集描述符
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)
        # 每个关键点会得到一个高维的描述符向量，描述图像中关键点的局部特征
        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

    def forward(self, inp):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        # 将输入图像 inp 通过卷积网络，提取低级到高级的特征
        # 卷积层使用 ReLU 激活函数，部分层使用 MaxPool 下采样
        x = self.relu(self.conv1a(inp))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])
        '''
            生成置信分数图
                convPa 和 convPb 提取每个像素点的置信分数
                使用 Softmax 正规化分数
            非极大值抑制（NMS）
                使用 simple_nms 函数抑制局部的非极值点，保留显著点作为候选关键点
        '''

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
        # 置信度阈值筛选: 仅保留置信分数大于 keypoint_threshold 的点作为关键点

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))
        # 边界点剔除: 删除离图像边界太近的关键点，避免特征提取不准确

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:       # 如果设置了 max_keypoints 参数（即最大关键点数量），则保留置信分数最高的关键点
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]         # torch.flip(k, [1]) 翻转关键点的第二个维度（即高度和宽度交换）
        # 将关键点的坐标从 (h, w) 格式转换为 (x, y) 格式

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        '''
        生成密集描述符
            使用 convDa 和 convDb 生成每个像素点的描述符
            对描述符进行 L2 归一化
        '''

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        # 采样描述符: 根据检测到的关键点位置，从密集描述符图中提取对应位置的描述符
        return {
            'keypoints': keypoints,             # 图像中的关键点位置，表示为 (𝑥,𝑦) 坐标
            'scores': scores,                   # 每个关键点的置信分数
            'descriptors': descriptors,         # 每个关键点的描述符向量
        }
