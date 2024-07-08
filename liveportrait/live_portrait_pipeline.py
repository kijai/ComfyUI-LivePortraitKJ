# coding: utf-8

"""
Pipeline of LivePortrait
"""

import cv2
import comfy.utils

import os.path as osp
import numpy as np
from .config.inference_config import InferenceConfig

# from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.crop import _transform_img

# from .utils.video import images2video, concat_frames

# from .utils.retargeting_utils import calc_lip_close_ratio
# from .utils.io import load_image_rgb, load_driving_info
# from .utils.helper import mkdir, basename, dct2cuda, is_video, is_template, resize_to_limit
# from .utils.helper import resize_to_limit

# from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):
    def __init__(
        self,
        appearance_feature_extractor,
        motion_extractor,
        warping_module,
        spade_generator,
        stitching_retargeting_module,
        inference_cfg: InferenceConfig,
    ):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(
            appearance_feature_extractor,
            motion_extractor,
            warping_module,
            spade_generator,
            stitching_retargeting_module,
            cfg=inference_cfg,
        )

    def _get_source_frame(self, source_np, idx, total_frames, method):
        if source_np.shape[0] == 1:
            return source_np[0]

        if method == "repeat":
            return source_np[min(idx, source_np.shape[0] - 1)]
        elif method == "cycle":
            return source_np[idx % source_np.shape[0]]
        elif method == "mirror":
            cycle_length = 2 * source_np.shape[0] - 2
            mirror_idx = idx % cycle_length
            if mirror_idx >= source_np.shape[0]:
                mirror_idx = cycle_length - mirror_idx
            return source_np[mirror_idx]
        elif method == "nearest":
            ratio = idx / (total_frames - 1)
            return source_np[
                min(int(ratio * (source_np.shape[0] - 1)), source_np.shape[0] - 1)
            ]

    def execute(
        self, source_np, driving_images_np, mismatch_method="repeat", reference_frame=0
    ):
        inference_cfg = self.live_portrait_wrapper.cfg
        is_video = source_np.shape[0] > 1

        I_p_lst = []
        I_p_paste_lst = []

        total_frames = driving_images_np.shape[0]

        pbar = comfy.utils.ProgressBar(total_frames)

        ref_frame = self._get_source_frame(
            source_np, reference_frame, total_frames, mismatch_method
        )
        rcrop_info = self.cropper.crop_single_image(ref_frame)
        rsource_lmk = rcrop_info["lmk_crop"]
        rimg_crop, ref_crop_256x256 = (
            rcrop_info["img_crop"],
            rcrop_info["img_crop_256x256"],
        )

        for i in range(total_frames):
            source_frame_rgb = self._get_source_frame(
                source_np, i, total_frames, mismatch_method
            )
            driving_frame = driving_images_np[i]

            crop_info = self.cropper.crop_single_image(source_frame_rgb)
            source_lmk = crop_info["lmk_crop"]
            img_crop, img_crop_256x256 = (
                crop_info["img_crop"],
                crop_info["img_crop_256x256"],
            )

            if inference_cfg.flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            else:
                I_s = self.live_portrait_wrapper.prepare_source(source_frame_rgb)

            rel_s_info = self.live_portrait_wrapper.get_kp_info(
                self.live_portrait_wrapper.prepare_source(ref_crop_256x256)
            )
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info["kp"]
            R_s = get_rotation_matrix(
                x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
            )
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            if inference_cfg.flag_lip_zero:
                c_d_lip_before_animation = [0.0]
                combined_lip_ratio_tensor_before_animation = (
                    self.live_portrait_wrapper.calc_combined_lip_ratio(
                        c_d_lip_before_animation, source_lmk
                    )
                )
                # TODO: expose lip_zero_threshold
                if (
                    combined_lip_ratio_tensor_before_animation[0][0]
                    < inference_cfg.lip_zero_threshold
                ):
                    inference_cfg.flag_lip_zero = False
                else:
                    lip_delta_before_animation = (
                        self.live_portrait_wrapper.retarget_lip(
                            x_s, combined_lip_ratio_tensor_before_animation
                        )
                    )

            # driving_frame_rgb = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
            driving_frame_256 = cv2.resize(driving_frame, (256, 256))
            I_d = self.live_portrait_wrapper.prepare_driving_videos(
                [driving_frame_256]
            )[0]

            if inference_cfg.flag_eye_retargeting or inference_cfg.flag_lip_retargeting:
                driving_lmk_lst = self.cropper.get_retargeting_lmk_info([driving_frame])
                input_eye_ratio_lst, input_lip_ratio_lst = (
                    self.live_portrait_wrapper.calc_retargeting_ratio(
                        source_lmk, driving_lmk_lst
                    )
                )

            x_d_info = self.live_portrait_wrapper.get_kp_info(I_d)
            R_d = get_rotation_matrix(
                x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"]
            )

            if inference_cfg.flag_relative:
                R_new = R_d @ R_s
                delta_new = rel_s_info["exp"] + (x_d_info["exp"] - rel_s_info["exp"])
                scale_new = rel_s_info["scale"] * (
                    x_d_info["scale"] / rel_s_info["scale"]
                )
                t_new = rel_s_info["t"] + (x_d_info["t"] - rel_s_info["t"])
            else:
                R_new = R_d
                delta_new = x_d_info["exp"]
                scale_new = x_s_info["scale"]
                t_new = x_d_info["t"]

            t_new[..., 2].fill_(0)  # zero tz
            x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if inference_cfg.flag_stitching:
                x_d_new = self.live_portrait_wrapper.stitching(x_s, x_d_new)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out["out"])[0]
            I_p_lst.append(I_p_i)

            # Transform and blend
            I_p_i_to_ori = _transform_img(
                I_p_i,
                crop_info["M_c2o"],
                dsize=(source_frame_rgb.shape[1], source_frame_rgb.shape[0]),
            )

            if inference_cfg.flag_pasteback:
                if inference_cfg.mask_crop is None:
                    inference_cfg.mask_crop = cv2.imread(
                        make_abs_path("./utils/resources/mask_template.png"),
                        cv2.IMREAD_COLOR,
                    )
                mask_ori = _transform_img(
                    inference_cfg.mask_crop,
                    crop_info["M_c2o"],
                    dsize=(source_frame_rgb.shape[1], source_frame_rgb.shape[0]),
                )
                mask_ori = mask_ori.astype(np.float32) / 255.0
                I_p_i_to_ori_blend = np.clip(
                    mask_ori * I_p_i_to_ori + (1 - mask_ori) * source_frame_rgb, 0, 255
                ).astype(np.uint8)
            else:
                I_p_i_to_ori_blend = I_p_i_to_ori

            I_p_paste_lst.append(I_p_i_to_ori_blend)
            pbar.update(1)

        return I_p_lst, I_p_paste_lst
