# coding: utf-8

"""
Pipeline of LivePortrait
"""

import comfy.utils
from tqdm import tqdm
import numpy as np
from .config.inference_config import InferenceConfig
import torch
from .utils.camera import get_rotation_matrix
from .utils.crop import _transform_img
from .live_portrait_wrapper import LivePortraitWrapper
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

import os
script_directory = os.path.dirname(os.path.abspath(__file__))

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

    def _get_source_frame(self, source_np, idx, method):
        if source_np.shape[0] == 1:
            return source_np[0]

        if method == "constant":
            return source_np[min(idx, source_np.shape[0] - 1)]
        elif method == "cycle":
            return source_np[idx % source_np.shape[0]]
        elif method == "mirror":
            cycle_length = 2 * source_np.shape[0] - 2
            mirror_idx = idx % cycle_length
            if mirror_idx >= source_np.shape[0]:
                mirror_idx = cycle_length - mirror_idx
            return source_np[mirror_idx]

    def execute(
        self, source_np, driving_images, crop_info, driving_landmarks, delta_multiplier = 1.0, mismatch_method="constant"
    ):
        inference_cfg = self.live_portrait_wrapper.cfg
        device = inference_cfg.device_id

        cropped_image_list = []
        composited_image_list = []
        out_mask_list = []
        R_d_0, x_d_0_info = None, None

        if mismatch_method == "cut" or inference_cfg.flag_eye_retargeting or inference_cfg.flag_lip_retargeting:
            total_frames = source_np.shape[0]
        else:
            total_frames = driving_images.shape[0]

        pbar = comfy.utils.ProgressBar(total_frames)

        for i in tqdm(range(total_frames), desc='Animating...', total=total_frames):

            safe_index = min(i, len(crop_info["crop_info_list"]) - 1)

            # skip and return empty frames if no crop due to no face detected
            if not crop_info["crop_info_list"][safe_index]:
                composited_image_list.append(source_np[safe_index])
                cropped_image_list.append(torch.zeros(1, 512, 512, 3, dtype=torch.float32, device = device))
                out_mask_list.append(np.zeros((source_np.shape[1], source_np.shape[2], 3), dtype=np.uint8))
                continue

            source_lmk = crop_info["crop_info_list"][safe_index]["lmk_crop"]
            img_crop_256x256 = crop_info["crop_info_list"][safe_index]["img_crop_256x256"]

            x_d_info = self.live_portrait_wrapper.get_kp_info(driving_images[i].unsqueeze(0).to(device))

            if mismatch_method == "cut" or inference_cfg.flag_eye_retargeting or inference_cfg.flag_lip_retargeting:
                source_frame_rgb = source_np[safe_index]
            else:
                source_frame_rgb = self._get_source_frame(source_np, i, mismatch_method)

            if inference_cfg.flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
            else:
                I_s = self.live_portrait_wrapper.prepare_source(source_frame_rgb)

            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info["kp"]
            R_s = get_rotation_matrix(
                x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
            )
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            #lip zero
            if inference_cfg.flag_lip_zero:
                c_d_lip_before_animation = [0.0]
                combined_lip_ratio_tensor_before_animation = (self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk))

                if (combined_lip_ratio_tensor_before_animation[0][0] < inference_cfg.lip_zero_threshold):
                    inference_cfg.flag_lip_zero = False
                else:
                    lip_delta_before_animation = (self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation))

            R_d = get_rotation_matrix(
                x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"]
            )

            if i == 0:
                R_d_0 = R_d
                x_d_0_info = x_d_info

            if inference_cfg.flag_relative:
                R_new = (R_d @ R_d_0.permute(0, 2, 1)) @ R_s
                delta_new = x_s_info["exp"] + (x_d_info["exp"] - x_d_0_info["exp"])
                scale_new = x_s_info["scale"] * (
                    x_d_info["scale"] / x_d_0_info["scale"]
                )
                t_new = x_s_info["t"] + (x_d_info["t"] - x_d_0_info["t"])
            else:
                R_new = R_d
                delta_new = x_d_info["exp"]
                scale_new = x_s_info["scale"]
                t_new = x_d_info["t"]

            t_new[..., 2].fill_(0)  # zero tz

            delta_new = delta_new * delta_multiplier
            
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            if (
                not inference_cfg.flag_stitching
                and not inference_cfg.flag_eye_retargeting
                and not inference_cfg.flag_lip_retargeting
            ):
                # without stitching or retargeting
                if inference_cfg.flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    pass
            elif (
                inference_cfg.flag_stitching
                and not inference_cfg.flag_eye_retargeting
                and not inference_cfg.flag_lip_retargeting
            ):
                # with stitching and without retargeting
                if inference_cfg.flag_lip_zero:
                    x_d_i_new = self.live_portrait_wrapper.stitching(
                        x_s, x_d_i_new
                    ) + lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            else:
                eyes_delta, lip_delta = None, None
                if inference_cfg.flag_eye_retargeting:
                    c_d_eyes_i = calc_eye_close_ratio(driving_landmarks[i][None])
                    combined_eye_ratio_tensor = (
                        self.live_portrait_wrapper.calc_combined_eye_ratio(
                            c_d_eyes_i, source_lmk
                        )
                    )
                    combined_eye_ratio_tensor = (
                        combined_eye_ratio_tensor
                        * inference_cfg.eyes_retargeting_multiplier
                    )
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(
                        x_s, combined_eye_ratio_tensor
                    )
                if inference_cfg.flag_lip_retargeting:
                    c_d_lip_i = calc_lip_close_ratio(driving_landmarks[i][None])
                    combined_lip_ratio_tensor = (
                        self.live_portrait_wrapper.calc_combined_lip_ratio(
                            c_d_lip_i, source_lmk
                        )
                    )
                    combined_lip_ratio_tensor = (
                        combined_lip_ratio_tensor
                        * inference_cfg.lip_retargeting_multiplier
                    )
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(
                        x_s, combined_lip_ratio_tensor
                    )

                if inference_cfg.flag_relative:  # use x_s
                    x_d_i_new = (
                        x_s
                        + (
                            eyes_delta.reshape(-1, x_s.shape[1], 3)
                            if eyes_delta is not None
                            else 0
                        )
                        + (
                            lip_delta.reshape(-1, x_s.shape[1], 3)
                            if lip_delta is not None
                            else 0
                        )
                    )
                else:  # use x_d,i
                    x_d_i_new = (
                        x_d_i_new
                        + (
                            eyes_delta.reshape(-1, x_s.shape[1], 3)
                            if eyes_delta is not None
                            else 0
                        )
                        + (
                            lip_delta.reshape(-1, x_s.shape[1], 3)
                            if lip_delta is not None
                            else 0
                        )
                    )

                if inference_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            if inference_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
    
            cropped_image = torch.clamp(out["out"], 0, 1).permute(0, 2, 3, 1)
            cropped_image_list.append(cropped_image)
        #return cropped_image_list
            # Transform and blend
            if inference_cfg.flag_pasteback:
                cropped_image_np = self.live_portrait_wrapper.parse_output(out["out"])[0]
                cropped_image_to_original = _transform_img(
                cropped_image_np,
                crop_info["crop_info_list"][safe_index]["M_c2o"],
                dsize=(source_frame_rgb.shape[1], source_frame_rgb.shape[0]),
                )

                mask_ori = _transform_img(
                    inference_cfg.mask_crop,
                    crop_info["crop_info_list"][safe_index]["M_c2o"],
                    dsize=(source_frame_rgb.shape[1], source_frame_rgb.shape[0]),
                    )
                
                mask_ori = mask_ori.astype(np.float32) / 255.0
                cropped_image_to_original_blend = np.clip(
                    mask_ori * cropped_image_to_original + (1 - mask_ori) * source_frame_rgb, 0, 255
                    ).astype(np.uint8)

            composited_image_list.append(cropped_image_to_original_blend)
            out_mask_list.append(mask_ori)
            pbar.update(1)

        return cropped_image_list, composited_image_list, out_mask_list
