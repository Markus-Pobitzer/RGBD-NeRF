import numpy as np
import os, imageio
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2

coord_trans_world = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
coord_trans_cam = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def load_4dor_data(path, half_res=False, load_depth=False, far=6.0):
    path = Path(path)
    imgdir = os.path.join(path, 'colorimage')
    img_names = sorted(os.listdir(imgdir))
    img_paths = [os.path.join(imgdir, f) for f in img_names if
                 f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in img_paths]
    imgs = np.stack(imgs, 0)
    num = imgs.shape[0]

    focal_rgb = 0
    focal_depth = 0

    cam_infos = load_cam_infos(path)

    poses_rgb = []
    for idx in range(num):
        img = img_names[idx]
        camera_name = img[:8]
        cam_params = cam_infos[camera_name]
        focal_rgb += (cam_params['fov_x'] + cam_params['fov_y']) / 2.  # Estimate focal length over all RGB-Cameras
        poses_rgb.append(cam_params["pose_rgb"][:3, :4])

    poses_rgb = np.stack(poses_rgb)
    print('RGB poses shape:', poses_rgb.shape)
    focal_rgb /= num
    H_rgb, W_rgb = imgs[0].shape[:2]
    print("RGB HWF:", H_rgb, W_rgb, focal_rgb)

    depth_maps = None
    poses_depth = None

    return poses_rgb, imgs, [H_rgb, W_rgb, focal_rgb], depth_maps, poses_depth, [None, None, focal_depth]


# Code taken from https://github.com/egeozsoy/4D-OR/blob/master/helpers/utils.py
def load_cam_infos(root_path: Path, cam_count=6):
    cam_infos = {}
    for c_idx in range(1, cam_count + 1):
        cam_json_path = root_path / f'camera0{c_idx}.json'
        with cam_json_path.open() as f:
            cam_info = json.load(f)['value0']
            intrinsics_json = cam_info['color_parameters']['intrinsics_matrix']
            intrinsics = np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                                     [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                                     [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

            extrinsics_json = cam_info['camera_pose']
            trans = extrinsics_json['translation']
            rot = extrinsics_json['rotation']
            extrinsics = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            extrinsics[:3, :3] = rot_matrix
            extrinsics[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]

            color2depth_json = cam_info['color2depth_transform']
            trans = color2depth_json['translation']
            rot = color2depth_json['rotation']
            color2depth_transform = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            color2depth_transform[:3, :3] = rot_matrix
            color2depth_transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
            depth_extrinsics = np.copy(extrinsics)
            extrinsics = np.matmul(extrinsics,
                                   color2depth_transform)  # Extrinsics were given for the depth camera, convert them to color camera

            fov_x_depth = cam_info['depth_parameters']['fov_x']
            fov_y_depth = cam_info['depth_parameters']['fov_y']
            c_x_depth = cam_info['depth_parameters']['c_x']
            c_y_depth = cam_info['depth_parameters']['c_y']
            width_depth = cam_info['depth_parameters']['width']
            height_depth = cam_info['depth_parameters']['height']
            depth_cam_params = {'fov_x': fov_x_depth, 'fov_y': fov_y_depth, 'c_x': c_x_depth, 'c_y': c_y_depth,
                                'width': width_depth, 'height': height_depth}

            fov_x = cam_info['color_parameters']['fov_x']
            fov_y = cam_info['color_parameters']['fov_y']
            c_x = cam_info['color_parameters']['c_x']
            c_y = cam_info['color_parameters']['c_y']
            width = cam_info['color_parameters']['width']
            height = cam_info['color_parameters']['height']

            params = cam_info['color_parameters']['radial_distortion']
            radial_params = params['m00'], params['m10'], params['m20'], params['m30'], params['m40'], params['m50']
            params = cam_info['color_parameters']['tangential_distortion']
            tangential_params = params['m00'], params['m10']

            # Computing the pose of the RGB camera
            pose_rgb = np.copy(extrinsics)
            # pose_rgb = np.eye(4)
            pose_rgb[:3, :3] = extrinsics[:3, :3].transpose()  # Transposing the rotation matrix
            # TODO: is he pose correct? What with the intrinsic
            # TODO: consider including radial_params and tangential_params
            pose_rgb = (
                    coord_trans_world
                    @ pose_rgb
                    @ coord_trans_cam
            )

            # Computing the pose of the RGB camera
            pose_depth = np.copy(depth_extrinsics)
            # pose_rgb = np.eye(4)
            pose_depth[:3, :3] = depth_extrinsics[:3, :3].transpose()  # Transposing the rotation matrix
            # TODO: is he pose correct? What with the intrinsic
            # TODO: consider including radial_params and tangential_params
            pose_depth = (
                    coord_trans_world
                    @ pose_depth
                    @ coord_trans_cam
            )

            cam_infos[f'camera0{c_idx}'] = {'intrinsics': intrinsics, 'extrinsics': extrinsics, 'pose_rgb': pose_rgb,
                                            'pose_depth': pose_depth, 'fov_x': fov_x,
                                            'fov_y': fov_y, 'c_x': c_x, 'c_y': c_y, 'width': width, 'height': height,
                                            'radial_params': radial_params, 'tangential_params': tangential_params,
                                            'depth_extrinsics': depth_extrinsics, 'depth_cam_params': depth_cam_params}

    return cam_infos


if __name__ == '__main__':
    load_4dor_data("../data/4D-OR/export_holistic_take1_processed/")
