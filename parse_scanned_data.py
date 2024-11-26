import os
import cv2
import tqdm
import numpy as np
import os.path as osp
import argparse
from pathlib import Path
from transforms3d import affines, quaternions
from src.utils import data_utils
import shutil

def get_arkit_default_path(data_dir):
    video_file = osp.join(data_dir, 'Frames.m4v')

    color_dir = osp.join(data_dir, 'color')
    Path(color_dir).mkdir(parents=True, exist_ok=True)

    box_file = osp.join(data_dir, 'Box.txt')
    assert Path(box_file).exists()
    out_3D_box_dir = osp.join(osp.dirname(data_dir), 'box3d_corners.txt')

    out_pose_dir = osp.join(data_dir, 'poses')
    Path(out_pose_dir).mkdir(parents=True, exist_ok=True)
    pose_file = osp.join(data_dir, 'ARposes.txt')
    assert Path(pose_file).exists()

    reproj_box_dir = osp.join(data_dir, 'reproj_box')
    Path(reproj_box_dir).mkdir(parents=True, exist_ok=True)
    out_box_dir = osp.join(data_dir, 'bbox')
    Path(out_box_dir).mkdir(parents=True, exist_ok=True)

    orig_intrin_file = osp.join(data_dir, 'Frames.txt')
    assert Path(orig_intrin_file).exists()

    final_intrin_file = osp.join(data_dir, 'intrinsics.txt')

    intrin_dir = osp.join(data_dir, 'intrin')
    Path(intrin_dir).mkdir(parents=True, exist_ok=True)

    M_dir = osp.join(data_dir, 'M')
    Path(M_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        'video_file': video_file,
        'color_dir': color_dir,
        'box_path': box_file,
        'pose_file': pose_file,
        'out_box_dir': out_box_dir,
        'out_3D_box_dir': out_3D_box_dir,
        'reproj_box_dir': reproj_box_dir,
        'out_pose_dir': out_pose_dir,
        'orig_intrin_file': orig_intrin_file,
        'final_intrin_file': final_intrin_file,
        'intrin_dir': intrin_dir,
        'M_dir': M_dir
    }

    return paths


def get_test_default_path(data_dir):
    video_file = osp.join(data_dir, 'Frames.m4v')

    # box_file = osp.join(data_dir, 'RefinedBox.txt')
    box_file = osp.join(data_dir, 'Box.txt')
    if osp.exists(box_file):
        os.remove(box_file)

    color_full_dir = osp.join(data_dir, 'color_full')
    Path(color_full_dir).mkdir(parents=True, exist_ok=True)

    pose_file = osp.join(data_dir, 'ARposes.txt')
    if osp.exists(pose_file):
        os.remove(pose_file)

    orig_intrin_file = osp.join(data_dir, 'Frames.txt')
    final_intrin_file = osp.join(data_dir, 'intrinsics.txt')

    paths = {
        'video_file': video_file,
        'color_full_dir': color_full_dir,
        'orig_intrin_file': orig_intrin_file,
        'final_intrin_file': final_intrin_file,
    }

    return paths


def get_bbox3d(box_path):
    assert Path(box_path).exists()
    with open(box_path, 'r') as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(',')]
    ex, ey, ez = box_data[3: 6]
    bbox_3d = np.array([
        [-ex, -ey, -ez],
        [ex,  -ey, -ez],
        [ex,  -ey, ez],
        [-ex, -ey, ez],
        [-ex,  ey, -ez],
        [ ex,  ey, -ez],
        [ ex,  ey, ez],
        [-ex,  ey, ez]
    ]) * 0.5
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((8, 1))], axis=1)
    return bbox_3d, bbox_3d_homo

def parse_box(box_path):
    """
    parse_box 函数读取一个 3D 包围盒的位姿描述文件，解析其中的位置信息和四元数旋转信息
    生成一个 4×4 的变换矩阵 T_ow（对象到世界坐标的变换矩阵）。
    @box_path: 一个文件路径，指向包含对象位姿信息的文本文件
    """
    # 读取文件内容: 读取 box_path 中的所有行内容
    with open(box_path, 'r') as f:
        lines = f.readlines()

    '''
    解析位置信息和四元数旋转
        提取数值:
            从第二行（lines[1]）中解析浮点数值。
            位置信息 (position): 前 3 个数值（data[:3]）是对象的 (x, y, z) 位置。
            旋转信息 (quaternion): 第 7 至最后的数值（data[6:]）是表示对象旋转的四元数 (qw, qx, qy, qz)
    '''
    data = [float(e) for e in lines[1].strip().split(',')]
    position = data[:3]
    quaternion = data[6:]
    rot_mat = quaternions.quat2mat(quaternion)                  # 将四元数转换为旋转矩阵

    '''
    使用 affines.compose 生成变换矩阵：
            position: 位置 (x, y, z)，定义平移部分。
            rot_mat: 旋转矩阵，定义旋转部分。
            np.ones(3): 单位尺度，表示对象没有缩放。
        输出的 T_ow 是一个 4×4 的变换矩阵：
    
    T_ow = [ R   t ]
           [ 0   1 ]
    '''
    T_ow = affines.compose(position, rot_mat, np.ones(3))       # 生成 4×4 的变换矩阵
    return T_ow     # 返回一个 4×4 的变换矩阵 T_ow，描述对象从对象坐标系到世界坐标系的变换。


def reproj(K_homo, pose, points3d_homo):
    """
    @K_homo: 齐次形式的相机内参矩阵，形状为 (3, 4)，通常由相机内参矩阵扩展一列零组成
    @pose: 相机的 4×4 世界到相机的变换矩阵，通常由旋转矩阵和平移向量构成
    @points3d_homo: 齐次形式的 3D 点云，形状为 [4, n]，包含 3D 点坐标和齐次坐标
    """
    assert K_homo.shape == (3, 4)
    assert pose.shape == (4, 4)
    assert points3d_homo.shape[0] == 4 # [4 ,n]

    '''
    pose @ points3d_homo: 将 3D 点云从世界坐标系转换到相机坐标系
    K_homo @ (上一步骤结果): 使用相机内参将点从相机坐标系投影到图像平面
    '''
    reproj_points = K_homo @ pose @ points3d_homo
    reproj_points = reproj_points[:]  / reproj_points[2:]       # 将齐次坐标的每一列除以其第三行的值（深度 z），得到归一化的 2D 图像坐标
    reproj_points = reproj_points[:2, :].T                      # 提取前两行（x, y 坐标），并转置得到最终结果，形状为 [n, 2]，其中 n 是输入 3D 点的数量
    return reproj_points # [n, 2]           返回 [n, 2] 的 2D 投影点坐标矩阵



def parse_video(paths, downsample_rate=5, bbox_3d_homo=None, hw=512):
    """
    parse_video 函数的主要目的是从输入视频中逐帧读取图像，裁剪、调整图像尺寸，并保存处理后的图像和对应的内参信息
    同时，还会对 3D 边界框投影和图像坐标变换矩阵进行处理，并保存相应的信息。
    """

    '''
    初始化变量
        读取原始内参文件，解析得到相机内参矩阵 K 和齐次形式内参矩阵 K_homo
            K: 3×3 相机内参矩阵
            K_homo: 4×4 齐次形式的相机内参矩阵
    '''
    orig_intrin_file = paths['final_intrin_file']
    K, K_homo = data_utils.get_K(orig_intrin_file)

    '''
    打开视频文件
        打开输入视频文件 video_file
        初始化帧索引 index 为 0，用于记录当前处理的帧
    '''
    intrin_dir = paths['intrin_dir']
    cap = cv2.VideoCapture(paths['video_file'])
    index = 0
    while True:
        # 逐帧处理视频: 使用 OpenCV 的 cap.read() 方法逐帧读取视频。如果视频读取失败（ret == False），退出循环。
        ret, image = cap.read()
        if not ret:
            break
        if index % downsample_rate == 0:        # 每隔 downsample_rate 帧处理一次帧图像，降低计算量
            # 帧的下采样:

            img_name = osp.join(paths['color_dir'], '{}.png'.format(index))
            save_intrin_path = osp.join(intrin_dir, '{}.txt'.format(index))

            '''
            读取投影后的 3D 边界框
                检查当前帧是否存在对应的投影边界框文件，如果不存在，则跳过该帧。
                加载投影后的 3D 边界框坐标，并提取边界框的左上角 (x0, y0) 和右下角 (x1, y1)。
            '''
            reproj_box3d_file = osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))
            if not osp.isfile(reproj_box3d_file):
                continue
            reproj_box3d = np.loadtxt(osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))).astype(int)
            x0, y0 = reproj_box3d.min(0)
            x1, y1 = reproj_box3d.max(0)

            '''
            裁剪并调整图像尺寸
                第一步裁剪:
                    根据 3D 边界框计算裁剪的区域 box 和目标大小 resize_shape
                    调用工具函数 data_utils.get_K_crop_resize 计算裁剪后的内参矩阵 K_crop 和齐次内参矩阵 K_crop_homo
                    调用工具函数 data_utils.get_image_crop_resize 裁剪并调整图像大小，同时返回裁剪的仿射变换矩阵 trans1
            '''
            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
            image_crop, trans1 = data_utils.get_image_crop_resize(image, box, resize_shape)

            '''
            裁剪并调整图像尺寸
                第二步裁剪:
                    将图像进一步调整到固定尺寸（如 512×512），更新 box_new 和 resize_shape
                    重新计算内参矩阵 K_crop 和裁剪变换矩阵 trans2
            '''
            box_new = np.array([0, 0, x1-x0, y1-y0])
            resize_shape = np.array([hw, hw])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
            image_crop, trans2 = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)

            '''
            保存裁剪变换矩阵
                合并两次裁剪的变换矩阵 trans1 和 trans2，得到完整的裁剪变换矩阵 trans_full_to_crop
                计算其逆矩阵 trans_crop_to_full，并保存
            '''
            trans_full_to_crop = trans2 @ trans1
            trans_crop_to_full = np.linalg.inv(trans_full_to_crop)

            np.savetxt(osp.join(paths['M_dir'], '{}.txt'.format(index)), trans_crop_to_full)

            '''
            投影 3D 边界框到裁剪图像坐标
                加载当前帧的相机姿态 pose
                将 3D 边界框 bbox_3d_homo 投影到当前裁剪图像的坐标系中，得到新的投影边界框 box_new
                保存新的投影边界框到文件中
            '''
            pose = np.loadtxt(osp.join(paths['out_pose_dir'], '{}.txt'.format(index)))
            reproj_crop = reproj(K_crop_homo, pose, bbox_3d_homo.T)
            x0_new, y0_new = reproj_crop.min(0)
            x1_new, y1_new = reproj_crop.max(0)
            box_new = np.array([x0_new, y0_new, x1_new, y1_new])

            '''
            保存图像和内参
                保存裁剪后的图像和全尺寸原始图像
                保存裁剪后更新的内参矩阵
            '''
            np.savetxt(osp.join(paths['out_box_dir'], '{}.txt'.format(index)), box_new)
            cv2.imwrite(img_name, image_crop)
            # cv2.imwrite(out_mask_file, mask_crop)
            full_img_dir = paths['color_dir'] + '_full'
            Path(full_img_dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(osp.join(full_img_dir, '{}.png'.format(index)), image)
            np.savetxt(save_intrin_path, K_crop)

        index += 1
    cap.release()


def data_process_anno(data_dir, downsample_rate=1, hw=512):
    paths = get_arkit_default_path(data_dir)
    '''
    解析相机内参 (orig_intrin_file)
        从原始内参文件中读取相机内参，并通过去掉注释行（# 开头）和空行清洗数据。
        将内参数据转化为浮点数，并取平均值以生成最终内参。
    '''
    with open(paths['orig_intrin_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    eles = [[float(e) for e in l.split(',')] for l in lines]
    data = np.array(eles)
    fx, fy, cx, cy = np.average(data, axis=0)[2:]
    '''生成的内参写入文件'''
    with open(paths['final_intrin_file'], 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))

    '''
    生成3D边界框
        调用 get_bbox3d 函数从指定路径加载3D边界框信息
        将3D边界框保存到 out_3D_box_dir
    '''
    bbox_3d, bbox_3d_homo = get_bbox3d(paths['box_path'])
    np.savetxt(paths['out_3D_box_dir'], bbox_3d)

    '''
    相机内参矩阵扩展
        在2D相机内参矩阵的基础上扩展为4×4矩阵，以支持齐次坐标。
    '''
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])

    '''
    处理位姿文件
        加载位姿文件，逐行解析位姿信息
        使用四元数（quaternion）计算旋转矩阵
        计算齐次变换矩阵 T_cw 和 T_oc（物体到相机的变换）
    '''
    with open(paths['pose_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        index = 0
        for line in tqdm.tqdm(lines):
            if len(line) == 0 or line[0] == '#':
                continue

            if index % downsample_rate == 0:
                eles = line.split(',')
                data = [float(e) for e in eles]

                position = data[1:4]
                quaternion = data[4:]
                # 使用四元数（quaternion）计算旋转矩阵
                rot_mat = quaternions.quat2mat(quaternion)
                rot_mat = rot_mat @ np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])

                T_ow = parse_box(paths['box_path'])
                # 计算齐次变换矩阵
                T_cw = affines.compose(position, rot_mat, np.ones(3))
                T_wc = np.linalg.inv(T_cw)
                T_oc = T_wc @ T_ow
                pose_save_path = osp.join(paths['out_pose_dir'], '{}.txt'.format(index))
                box_save_path = osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))
                '''
                投影3D边界框到2D
                    使用变换矩阵和相机内参将3D边界框投影到2D
                    检查投影边界是否在合理范围内
                '''
                reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)     # 使用变换矩阵和相机内参将3D边界框投影到2D

                x0, y0 = reproj_box3d.min(0)
                x1, y1 = reproj_box3d.max(0)

                if x0 < -1000 or y0 < -1000 or x1 > 3000 or y1 > 3000:      # 检查投影边界是否在合理范围内
                    continue

                # 保存位姿和投影结果
                np.savetxt(pose_save_path, T_oc)
                np.savetxt(box_save_path, reproj_box3d)
            index += 1

    # 调用 parse_video 对视频帧进行解析和下采样
    parse_video(paths, downsample_rate, bbox_3d_homo, hw=hw)

    # Make fake data for demo annotate video without BA:
    # 生成伪数据以便在未进行BA（Bundle Adjustment）情况下的注释视频
    if osp.exists(osp.join(osp.dirname(paths['intrin_dir']), 'intrin_ba')):
        shutil.rmtree(osp.join(osp.dirname(paths['intrin_dir']), 'intrin_ba'), ignore_errors=True)
    shutil.copytree(paths['intrin_dir'], osp.join(osp.dirname(paths['intrin_dir']), 'intrin_ba'))

    if osp.exists(osp.join(osp.dirname(paths['out_pose_dir']), 'poses_ba')):
        shutil.rmtree(osp.join(osp.dirname(paths['out_pose_dir']), 'poses_ba'), ignore_errors=True)
    shutil.copytree(paths['out_pose_dir'], osp.join(osp.dirname(paths['out_pose_dir']), 'poses_ba'))

def data_process_test(data_dir, downsample_rate=1):
    # data_process_test 函数的目的是对测试数据集进行预处理
    # 包括解析相机内参文件、提取视频帧并保存帧图像，支持对帧进行下采样（downsample_rate）
    paths = get_test_default_path(data_dir)         # 根据输入的 data_dir 获取测试数据集的各类文件路径

    # Parse intrinsic:
    #
    '''
    读取并解析原始内参文件，提取和保存相机内参（焦距和光心坐标）操作如下：
        打开 orig_intrin_file，读取非空且非注释行（以 # 开头的行为注释行）。
        将每行的内容按逗号分隔并转换为浮点数，存储为二维数组 data。
        使用 np.average(data, axis=0) 对数据的每列取平均值，取第3列至第6列的值作为相机内参的 fx、fy（焦距）和 cx、cy（光心坐标）。
        将内参写入 final_intrin_file，格式为
            fx: value
            fy: value
            cx: value
            cy: value
    '''
    with open(paths['orig_intrin_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    eles = [[float(e) for e in l.split(',')] for l in lines]
    data = np.array(eles)
    fx, fy, cx, cy = np.average(data, axis=0)[2:]
    with open(paths['final_intrin_file'], 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))
    
    # Parse video:
    '''
    逐帧解析视频文件并保存帧图像, 操作如下:
        使用 OpenCV 的 cv2.VideoCapture 打开输入视频文件 video_file
        遍历视频的每一帧
            ret, image = cap.read()
                ret: 表示读取成功与否
                image: 当前帧的图像数据
            如果视频读取失败（ret == False），退出循环
        如果帧索引 index 是 downsample_rate 的倍数，表示该帧需要被保存
            使用 cv2.imwrite 将图像保存到 color_full_dir，文件名以帧索引命名，例如 0.png、1.png 等
        增加帧索引 index
        完成解析后释放视频资源 cap.release()
    '''
    cap = cv2.VideoCapture(paths['video_file'])
    index = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if index % downsample_rate == 0:
            full_img_dir = paths['color_full_dir']
            cv2.imwrite(osp.join(full_img_dir, '{}.png'.format(index)), image)
        index += 1
    cap.release()

def parse_args():
    # 模块的核心类，用于定义需要解析的命令行参数
    parser = argparse.ArgumentParser(
        # 自动在帮助信息中显示参数的默认值
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    '''
    添加一个名为 --scanned_object_path 的命令行参数
        参数类型为字符串（type=str）
        参数是必须提供的（required=True）
    '''
    parser.add_argument("--scanned_object_path", type=str, required=True)

    # 从命令行解析用户提供的参数，并将其存储到 args 对象中
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # 调用 parse_args 函数解析命令行参数，其中 scanned_object_path 指定了扫描对象的根目录
    data_dir = args.scanned_object_path
    # 使用 assert 检查指定路径是否存在，不存在时抛出异常
    assert osp.exists(data_dir), f"Scanned object path:{data_dir} not exists!"

    # 遍历 data_dir 中的所有子目录（或文件）
    seq_dirs = os.listdir(data_dir)
    for seq_dir in seq_dirs:
        # seq_dir 是当前子目录的名称

        '''
        根据命名规则选择处理方式
            如果子目录名称包含 -annotate，调用 data_process_anno 函数进行注释序列处理。
            如果名称包含 -test，调用 data_process_test 函数处理测试序列
        '''
        if '-annotate' in seq_dir:
            print('=> Processing annotate sequence: ', seq_dir)
            # 下采样率 downsample_rate 默认为 1，不执行下采样
            data_process_anno(osp.join(data_dir, seq_dir), downsample_rate=1, hw=512)       # 用于处理注释序列，传入完整路径、下采样率、和图像目标分辨率。
        elif '-test' in seq_dir:
            # Parse scanned test sequence
            print('=> Processing test sequence: ', seq_dir)
            # 下采样率 downsample_rate 默认为 1，不执行下采样
            data_process_test(osp.join(data_dir, seq_dir), downsample_rate=1)               # 用于处理测试序列，传入完整路径和下采样率。
        else:
            # 如果目录名称既不包含 -annotate 也不包含 -test，跳过该目录
            continue

