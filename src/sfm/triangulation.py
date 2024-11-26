import os
import h5py
import logging
import tqdm
import subprocess
import os.path as osp
import numpy as np

from pathlib import Path
from src.utils.colmap.read_write_model import CAMERA_MODEL_NAMES, Image, read_cameras_binary, read_images_binary
from src.utils.colmap.database import COLMAPDatabase


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def geometric_verification(colmap_path, database_path, pairs_path):
    """ Geometric verfication """
    logging.info('Performing geometric verification of the matches...')

    colmap_path = Path(colmap_path)
    database_path = Path(database_path)
    pairs_path = Path(pairs_path)

    cmd = [
        colmap_path, 'matches_importer',
        '--database_path', database_path,
        '--match_list_path', pairs_path,
        '--match_type', 'pairs'
    ]
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.warning('Problem with matches_importer, existing.')
        exit(ret)


def create_db_from_model(empty_model, database_path):
    """ Create COLMAP database file from empty COLMAP binary file. """
    if database_path.exists():
        logging.warning('Database already exists.')
    
    cameras = read_cameras_binary(str(empty_model / 'cameras.bin'))
    images = read_images_binary(str(empty_model / 'images.bin'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    
    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(model_id, camera.width, camera.height, camera.params,
                      camera_id=i, prior_focal_length=True)
    
    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)
    
    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}


def import_features(image_ids, database_path, feature_path):
    """ Import keypoints info into COLMAP database. """
    logging.info("Importing features into the database...")
    feature_file = h5py.File(str(feature_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm.tqdm(image_ids.items()):
        keypoints = feature_file[image_name]['keypoints'].__array__()
        keypoints += 0.5
        db.add_keypoints(image_id, keypoints)
    
    feature_file.close()
    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path, feature_path,
                   min_match_score=None, skip_geometric_verification=False):
    """ Import matches info into COLMAP database. """
    logging.info("Importing matches into the database...")

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]
    
    match_file = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)
    
    matched = set()
    for name0, name1 in tqdm.tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
    
        pair = names_to_pair(name0, name1)
        if pair not in match_file:
            raise ValueError(
                f'Could not find pair {(name0, name1)}... '
                'Maybe you matched with a different list of pairs? '
                f'Reverse in file: {names_to_pair(name0, name1) in match_file}.'
            )
        
        matches = match_file[pair]['matches0'].__array__()
        valid = matches > -1
        if min_match_score:
            scores = match_file[pair]['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)

        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)
    
    match_file.close()
    db.commit()
    db.close()


def run_triangulation(colmap_path, model_path, database_path, image_dir, empty_model):
    """ run triangulation on given database """
    logging.info('Running the triangulation...')
    # if not image_dir:
    #     image_dir = '/'
    # 确保 image_dir 格式化为统一的正斜杠路径
    # image_dir = Path(image_dir).as_posix()

    # print(f"Image path: {image_dir}")
    # print(f"Image directory: {image_dir}")
    # print(f"Final path: {Path(image_dir) / image_dir}")

    cmd = [
        colmap_path, 'point_triangulator',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--input_path', empty_model,
        '--output_path', model_path,
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0'
    ]
    logging.info(' '.join(cmd))
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.warning('Problem with point_triangulator, existing.')
        exit(ret)
    
    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', model_path]
    )
    stats_raw = stats_raw.decode().split('\n')
    stats = dict()
    for stat in stats_raw:
        if stat.startswith('Register images'):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith('Points'):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith('Observation'):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith('Mean track length'):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith('Mean observation per image'):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith('Mean reprojection error'):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])
    return stats

def convert_colmap_model(colmap_path, model, outputs_dir):
    """Convert COLMAP model to PLY format on Windows."""
    # 确保输出路径存在
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # 构造命令
    cmd = [
        colmap_path, 'model_converter',
        '--input_path', model,
        '--output_path', os.path.join(outputs_dir, 'model.ply'),
        '--output_type', 'PLY'
    ]

    # 打印命令用于调试
    print(f"Running command: {' '.join(cmd)}")

    # 调用 subprocess.run 执行命令
    try:
        result = subprocess.run(cmd, shell=True)
        print(result.stdout)  # 打印标准输出
        print("Model conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during model conversion: {e}")
        print(e.stderr)  # 打印标准错误

def main(sfm_dir, empty_sfm_model, outputs_dir, pairs, features, matches, \
         colmap_path='E:\\githubCode\\OnePose\\colmap-x64-windows-cuda\\bin\\colmap.exe', skip_geometric_verification=False, min_match_score=None, image_dir=None):
    """ 
        Import keypoints, matches.
        Given keypoints and matches, reconstruct sparse model from given camera poses.
    """
    assert Path(empty_sfm_model).exists(), empty_sfm_model
    assert Path(features).exists(), features
    assert Path(pairs).exists(), pairs
    assert Path(matches).exists(), matches 

    Path(sfm_dir).mkdir(parents=True, exist_ok=True)
    database = osp.join(sfm_dir, 'database.db')
    model = osp.join(sfm_dir, 'model')
    Path(model).mkdir(exist_ok=True)

    image_ids = create_db_from_model(Path(empty_sfm_model), Path(database))
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches, features,
                   min_match_score, skip_geometric_verification)
    
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs)

    # img_list = []
    # with open(pairs, 'r') as f:
    #     for line in f:
    #         name0, name1 = line.split(' ')
    #         img_list.append((name0.split('\\')[-1], name1.split('\\')[-1]))
    #
    # with open(outputs_dir+'/match.txt', 'w') as f:
    #     f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
    #
    if not image_dir:
        image_dir = '\"\"'
    stats = run_triangulation(colmap_path, model, database, image_dir, empty_sfm_model)
    convert_colmap_model(colmap_path, model, outputs_dir)
    # os.system(f'colmap model_converter --input_path {model} --output_path {outputs_dir}/model.ply --output_type PLY')