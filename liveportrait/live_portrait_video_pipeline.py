# coding: utf-8

"""
Pipeline of LivePortrait
"""

# TODO:
# 1. 当前假定所有的模板都是已经裁好的，需要修改下
# 2. pick样例图 source + driving

import cv2
import numpy as np
import pickle
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.retargeting_utils import calc_lip_close_ratio
from .utils.io import load_image_rgb, load_driving_info, resize_to_limit
from .utils.helper import mkdir, basename, dct2cuda, is_video, is_template
from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def execute(self, args: ArgumentConfig):
        inference_cfg = self.live_portrait_wrapper.cfg  # for convenience

        # Load the source video
        source_video = cv2.VideoCapture(args.source_video)
        
        # Load the driving video
        driving_video = cv2.VideoCapture(args.driving_info)
        
        # Get video properties
        fps = source_video.get(cv2.CAP_PROP_FPS)
        width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(osp.join(args.output_dir, f'{basename(args.source_video)}_edited.mp4'), 
                                    fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret_source, source_frame = source_video.read()
            ret_driving, driving_frame = driving_video.read()
            
            if not ret_source or not ret_driving:
                break  # End of video
            
            frame_count += 1
            log(f"Processing frame {frame_count}")
            
            # Process source frame
            source_frame_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
            source_frame_rgb = resize_to_limit(source_frame_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
            crop_info = self.cropper.crop_single_image(source_frame_rgb)
            source_lmk = crop_info['lmk_crop']
            img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
            
            if inference_cfg.flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            else:
                I_s = self.live_portrait_wrapper.prepare_source(source_frame_rgb)
            
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            # Process driving frame
            driving_frame_rgb = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
            driving_frame_256 = cv2.resize(driving_frame_rgb, (256, 256))
            I_d = self.live_portrait_wrapper.prepare_driving_videos([driving_frame_256])[0]
            x_d_info = self.live_portrait_wrapper.get_kp_info(I_d)
            R_d = get_rotation_matrix(x_d_info['pitch'], x_d_info['yaw'], x_d_info['roll'])

            # Apply transformations
            R_new = R_d @ R_s
            delta_new = x_d_info['exp']
            scale_new = x_s_info['scale']
            t_new = x_d_info['t']
            t_new[..., 2].fill_(0)  # zero tz
            x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # Apply stitching if enabled
            if inference_cfg.flag_stitching:
                x_d_new = self.live_portrait_wrapper.stitching(x_s, x_d_new)

            # Generate output frame
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_new)
            I_p = self.live_portrait_wrapper.parse_output(out['out'])[0]

            # Paste back if enabled
            if inference_cfg.flag_pasteback:
                mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_frame_rgb.shape[1], source_frame_rgb.shape[0]))
                I_p = paste_back(I_p, crop_info['M_c2o'], source_frame_rgb, mask_ori)

            # Convert back to BGR for writing
            I_p_bgr = cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)
            out_video.write(I_p_bgr)

        # Release resources
        source_video.release()
        driving_video.release()
        out_video.release()

        log(f"Edited video saved to: {osp.join(args.output_dir, f'{basename(args.source_video)}_edited.mp4')}")
        return osp.join(args.output_dir, f'{basename(args.source_video)}_edited.mp4'), None
