#!/usr/bin/env python

"""
Usage:
    $ python3 path/to/pose.py --cfg calib.yaml

"""

import os
import cv2
from pupil_apriltags import Detector #https://github.com/pupil-labs/apriltags
from calibrate import FisheyeModel, PinholeModel
from bundle import *
from utils import RM2EA, vis_tag_detections, vis_bundle_detections, save_extrinsic
import numpy as np
import argparse
from pathlib import Path
import yaml
import time
import glob

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


def apriltag_pose(cfg):
    """Estimate AprilTags bundles relative pose.

    Args: 
        cfg (dict): Configurations.

    Returns: 
        The rotation and translation matrices.
    """

    # Dimensions of checkerboard
    hcorners = 9
    vcorners = 6

    # AprilTag size in meters
    tag_size = 0.2032

    bt_detector = AprilTagBundles3x3(tag_size=tag_size)

    cam_model = cfg.get('cam_model')
    if cam_model == 'fisheye':
        cam = FisheyeModel(hcorners=hcorners, vcorners=vcorners)
        refDir = './images/fisheye_chkbd'
        imgPath = './images/fisheye.png'
        bundleA_id = 1
        bundleB_id = 0

    elif cam_model == 'pinhole':
        cam = PinholeModel(hcorners=hcorners, vcorners=vcorners)
        refDir = './images/pinhole_chkbd'
        imgPath = './images/pinhole.png'
        bundleA_id = 1
        bundleB_id = 6
    else:
        raise ValueError(f'{cam_model} - Invalid camera model. Use fisheye or pinhole.\n')

    cam.calibrate(images=glob.glob(refDir + "/*.png"), process=f'Calibrating {cam_model} camera')

    undimg, K_new = cam.undistort(img_path=imgPath)
    gray_img = cv2.cvtColor(undimg, cv2.COLOR_BGR2GRAY)
    camera_params = (K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2])

    tags_info = at_detector.detect(gray_img, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    bundles_info = bt_detector.detect_bundles(tags_info=tags_info, K=K_new, D=cam.D)

    #save annotated images
    if cfg.get('debug'):
        vis_tag_detections(undimg, tags_info, camera_params, tag_size)
        vis_bundle_detections(undimg, bundles_info, camera_params, tag_size)

        output_path = os.path.join(cfg.get('savedir'),'undistorted_'+str(os.path.split(imgPath)[1]))
        output_path = output_path.replace(str(os.path.splitext(imgPath)[1]), '.png')
        cv2.imwrite(output_path, undimg)

    for bundle in bundles_info:
        bundle_id = bundle.bundle_id
        print('Detected aprilTag bundle (id={})'.format(bundle_id))

        if bundle_id == bundleA_id:
            R_xa = bundle.pose_R
            T_xa = bundle.pose_t
        elif bundle_id == bundleB_id:
            R_xb = bundle.pose_R
            T_xb = bundle.pose_t
        else:
            assert False,'Bundle IDs do not match.'

    assert R_xa is not None, 'R_xa and T_xa are undefined.'
    assert R_xb is not None, 'R_xb and T_xb are undefined.'

    #calculate the rotation and translation of bundle B w.r.t bundle A.
    R_ab = np.matmul(np.transpose(R_xa),R_xb)
    T_ab = np.matmul(np.transpose(R_xa),np.subtract(T_xb, T_xa))
    print('Estimated pose B with respect to A from undistored {}'.format(imgPath))

    return R_ab, T_ab


def run(cfg='calib.yaml'):
    """Camera calibration and pose estimation.

    Args: 
        cfg (str): Configuration file.

    """
    t = time.time()

    # Read yaml
    if isinstance(cfg, (str, Path)):
        with open(cfg, errors='ignore') as f:
            cfg = yaml.safe_load(f)

    save_dir = cfg.get('savedir')
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)

    R_ab, T_ab = apriltag_pose(cfg)

    #save the pose
    save_extrinsic(R_ab, T_ab, os.path.join(save_dir,'poseAB.txt'))

    # Finish
    print('\nCalibration complete ({:.2f}s)'.format(time.time() - t))
    print('\nResults saved to {}'.format(save_dir))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='calib.yaml', help='provide the calib.yaml path')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)