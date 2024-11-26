import json
import os
import glob
import hydra

import os.path as osp
from loguru import logger
import shutil
from pathlib import Path
from omegaconf import DictConfig


def merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
           idxs_file, img_id, ann_id, images, annotations):
    """ To prepare training and test objects, we merge annotations about difference objs"""
    '''将2D注释文件、3D注释文件路径和图像索引文件信息整合为统一格式，并更新全局注释数据结构（images 和 annotations）'''
    with open(anno_2d_file, 'r') as f:              # 包含图像的 2D 注释信息
        annos_2d = json.load(f)                     # 将其解析为 Python 字典列表 annos_2d
    
    for anno_2d in annos_2d:
        # 遍历 annos_2d 中的每条注释数据
        # 每条数据代表一张图像的相关注释

        img_id += 1
        # 为每张图像分配一个全局唯一的 img_id
        info = {
            'id': img_id,                           # 图像的唯一标识符
            'img_file': anno_2d['img_file'],        # 图像文件路径
        }
        images.append(info)                         # 将图像信息存入全局列表 images

        ann_id += 1
        anno = {
            'image_id': img_id,                             # 与该注释相关联的图像标识符
            'id': ann_id,                                   # 注释的唯一标识符
            'pose_file': anno_2d['pose_file'],              # 注释的位姿文件路径
            'anno2d_file': anno_2d['anno_file'],            # 对应的2D注释文件路径
            'avg_anno3d_file': avg_anno_3d_file,            # 平均3D注释文件路径
            'collect_anno3d_file': collect_anno_3d_file,    # 3D点集合注释文件路径
            'idxs_file': idxs_file                          # 图像索引文件路径
        }
        annotations.append(anno)                            # 将注释信息存入全局列表 annotations
    return img_id, ann_id


def merge_anno(cfg):
    """ Merge different objects' anno file into one anno file """
    '''该函数的功能是合并多个对象的注释文件（anno_2d 和 anno_3d），并将它们保存为一个统一的注释文件'''
    anno_dirs = []

    # 根据数据集的划分类型（train 或 val），从配置文件中获取对象名称列表（names）
    if cfg.split == 'train':
        names = cfg.train.names
    elif cfg.split == 'val':
        names = cfg.val.names

    # 遍历每个对象名称，为每个对象构造其注释目录路径，并存储在 anno_dirs 列表中
    for name in names:
        anno_dir = osp.join(cfg.datamodule.data_dir, name, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
        anno_dirs.append(anno_dir)


    img_id = 0                  # 用于为图像赋予全局唯一的编号
    ann_id = 0                  # 用于为注释赋予全局唯一的编号
    images = []                 # 存储所有对象的图像信息
    annotations = []            # 存储所有对象的注释信息
    for anno_dir in anno_dirs:
        logger.info(f'Merging anno dir: {anno_dir}')
        anno_2d_file = osp.join(anno_dir, 'anno_2d.json')                   # 2D注释文件，包含图像中的目标信息
        avg_anno_3d_file = osp.join(anno_dir, 'anno_3d_average.npz')        # 3D注释文件（平均值形式）
        collect_anno_3d_file = osp.join(anno_dir, 'anno_3d_collect.npz')    # 3D注释文件（所有点集合）
        idxs_file = osp.join(anno_dir, 'idxs.npy')                          # 图像索引文件

        # 如果注释文件缺失，则跳过该目录的处理
        if not osp.isfile(anno_2d_file) or not osp.isfile(avg_anno_3d_file) or not osp.isfile(collect_anno_3d_file):
            logger.info(f'No annotation in: {anno_dir}')
            continue

        # 合并注释数据
        img_id, ann_id = merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
                                idxs_file, img_id, ann_id, images, annotations)
        # 将当前目录中的注释数据合并到全局的 images 和 annotations 中
        # 同时更新全局的 img_id 和 ann_id

    logger.info(f'Total num: {len(images)}')
    # 将 images 和 annotations 封装为字典 instance
    instance = {'images': images, 'annotations': annotations}

    out_dir = osp.dirname(cfg.datamodule.out_path)
    # 确保输出路径存在，如果不存在则创建
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(cfg.datamodule.out_path, 'w') as f:
        json.dump(instance, f)                          # 使用 json.dump 将 instance 写入输出文件


def sfm(cfg):
    """ Reconstruct and postprocess sparse object point cloud, and store point cloud features"""
    '''
    这段代码实现了一个 Structure from Motion (SfM) 流程
    结合后处理来生成稀疏的三维点云，并提取点云特征
    '''
    data_dirs = cfg.dataset.data_dir        # 包含数据的路径，可以是字符串或列表
    down_ratio = cfg.sfm.down_ratio         # 控制图像的下采样比例，用于选择特定帧进行处理
    data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs

    # 遍历数据集目录
    for data_dir in data_dirs:
        logger.info(f"Processing {data_dir}.")
        # 每个元素形如 "root_dir sub_dir1 sub_dir2 ..."
        root_dir, sub_dirs = data_dir.split(' ')[0], data_dir.split(' ')[1:]

        # Parse image directory and downsample images:
        img_lists = []
        for sub_dir in sub_dirs:
            # 寻找图像文件: 在每个子目录中查找 color/ 文件夹下的 .png 图像
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(str(Path(seq_dir)) + '\\color\\*.png', recursive=True)

        down_img_lists = []
        # 下采样: 对图像文件的索引按 down_ratio 取模，仅保留符合条件的图像（减少计算量）
        for img_file in img_lists:
            index = int(osp.basename(img_file).split('.')[0])
            if index % down_ratio == 0:
                down_img_lists.append(img_file)
        img_lists = down_img_lists

        if len(img_lists) == 0:
            # 无图像文件: 如果没有图像文件，记录日志并跳过当前目录
            logger.info(f"No png image in {root_dir}")
            continue
        
        obj_name = root_dir.split('/')[-1]                              # 提取当前对象的名称，作为后续处理的标识。
        outputs_dir_root = cfg.dataset.outputs_dir.format(obj_name)     # 根据配置生成输出路径

        # Begin SfM and postprocess:
        sfm_core(cfg, img_lists, outputs_dir_root, obj_name)                      # 核心 SfM 流程，生成稀疏三维点云, 输入：配置、图像路径、输出目录
        postprocess(cfg, img_lists, root_dir, outputs_dir_root)         # 后处理步骤，包括点云特征提取或优化


def sfm_core(cfg, img_lists, outputs_dir_root, obj_name):
    """ Sparse reconstruction: extract features, match features, triangulation"""
    from src.sfm import extract_features, match_features, \
                         generate_empty, triangulation, pairs_from_poses

    # Construct output directory structure:
    # 初始化输出目录
    outputs_dir = osp.join(outputs_dir_root, 'outputs' + '_' + cfg.network.detection + '_' + cfg.network.matching)
    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')            # 存储提取的特征
    covis_pairs_out = osp.join(outputs_dir, f'pairs-covis{cfg.sfm.covis_num}.txt')      # 存储基于协视关系的图像对
    matches_out = osp.join(outputs_dir, f'matches-{cfg.network.matching}.h5')           # 存储特征匹配结果
    empty_dir = osp.join(outputs_dir, 'sfm_empty')                                      # 生成空白模型，用于初始化重建
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')                                      # 存储最终重建的工作空间

    if cfg.redo:
        # 删除现有输出目录及其内容
        if os.path.exists(outputs_dir):
            shutil.rmtree(outputs_dir)
        # 创建新的目录
        Path(outputs_dir).mkdir(exist_ok=True, parents=True)
        # Extract image features, construct image pairs and then match:
        # 从输入的图像列表 img_lists 中提取局部特征，并存储在 feature_out 文件中
        # cfg 提供特征提取的配置（如检测器类型和参数）
        extract_features.main(img_lists, feature_out, cfg)
        '''
        基于图像的协视关系（视点相邻或重叠）构造图像对
            covis_pairs_out：输出协视图像对文件
            cfg.sfm.covis_num：控制生成的图像对数量
            cfg.sfm.rotation_thresh：设定视角旋转的阈值
        '''
        pairs_from_poses.covis_from_pose(img_lists, covis_pairs_out, cfg.sfm.covis_num, max_rotation=cfg.sfm.rotation_thresh)

        '''
        调用 match_features
            对提取的特征进行匹配
            输入：特征文件、协视图像对
            输出：匹配结果存储在 matches_out 文件中
        '''
        match_features.main(cfg, feature_out, covis_pairs_out, matches_out, obj_name,vis_match=False)

        # Reconstruct 3D point cloud with known image poses:
        # 调用 generate_empty: 根据图像列表生成一个空白的初始模型，作为三角化的基础
        generate_empty.generate_model(img_lists, empty_dir)
        '''
        调用 triangulation:
            基于特征匹配和已知的图像位姿，重建稀疏三维点云
            输入:
                deep_sfm_dir：用于存储三角化结果的目录
                empty_dir：初始化的空模型
                covis_pairs_out：协视图像对
                feature_out：图像特征
                matches_out：匹配结果
        '''
        triangulation.main(deep_sfm_dir, empty_dir, outputs_dir, covis_pairs_out, feature_out, matches_out, image_dir=f"E:/githubCode/OnePose/data/sfm_model/{obj_name}/")
    
    
def postprocess(cfg, img_lists, root_dir, outputs_dir_root):
    """ Filter points and average feature"""
    '''
    这段代码从 src.sfm.postprocess 中导入了用于后处理的三个主要模块:
        filter_points: 用于过滤和合并3D点
        feature_process: 用于处理点云的特征
        filter_tkl: 用于通过轨迹长度限制点云的数量
    '''
    from src.sfm.postprocess import filter_points, feature_process, filter_tkl

    bbox_path = osp.join(root_dir, "box3d_corners.txt")                                                                 # 包含3D包围盒（bounding box）信息的文件路径
    # Construct output directory structure:
    outputs_dir = osp.join(outputs_dir_root, 'outputs' + '_' + cfg.network.detection + '_' + cfg.network.matching)      # 保存后处理结果的根目录
    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')                                            # 保存点云特征的文件路径
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')                                                                      # 稀疏重建工作空间的路径
    model_path = osp.join(deep_sfm_dir, 'model')                                                                        # 存放3D点云模型的路径

    # Select feature track length to limit the number of 3D points below the 'max_num_kp3d' threshold:

    '''
    filter_tkl.get_tkl:
        计算点云的轨迹长度（track length），并限制轨迹长度以使得最终3D点云数量不超过配置文件中的阈值 cfg.dataset.max_num_kp3d
        返回满足条件的轨迹长度以及点云数量列表
    '''
    track_length, points_count_list = filter_tkl.get_tkl(model_path, thres=cfg.dataset.max_num_kp3d, show=False)

    '''
    filter_tkl.vis_tkl_filtered_pcds:
        可视化经过轨迹长度过滤后的点云，用于调试和验证
    '''
    filter_tkl.vis_tkl_filtered_pcds(model_path, points_count_list, track_length, outputs_dir) # For visualization only

    # Leverage the selected feature track length threshold and 3D BBox to filter 3D points:
    '''
    filter_points.filter_3d:
        使用轨迹长度阈值和3D包围盒，过滤掉无效或离群的3D点
        返回过滤后的3D点 (xyzs) 及其对应的索引 (points_idxs)
    '''
    xyzs, points_idxs = filter_points.filter_3d(model_path, track_length, bbox_path)

    # Merge 3d points by distance between points
    '''
    filter_points.merge
        按照距离阈值 (dist_threshold=1e-3)，将距离过近的点合并为一个点
        返回合并后的点 (merge_xyzs) 和更新后的索引 (merge_idxs)
    '''
    merge_xyzs, merge_idxs = filter_points.merge(xyzs, points_idxs, dist_threshold=1e-3) 

    # Save features of the filtered point cloud:
    '''
    feature_process.get_kpt_ann
        结合点云特征和图像列表，提取关键点信息并保存到输出目录中
    '''
    feature_process.get_kpt_ann(cfg, img_lists, feature_out, outputs_dir, merge_idxs, merge_xyzs)
    

@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)            # 执行配置里面的type的方法(反射)


if __name__ == "__main__":
    main()