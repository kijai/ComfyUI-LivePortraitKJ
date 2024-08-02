import os
import torch
import yaml
import folder_paths
import comfy.model_management as mm
import comfy.utils
import numpy as np
import cv2
from tqdm import tqdm
import gc

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

from .liveportrait.live_portrait_pipeline import LivePortraitPipeline
try:
    from .liveportrait.utils.cropper import CropperMediaPipe
except:
    log.warning("Can't load MediaPipe, MediaPipeCropper not available")
try:
    from .liveportrait.utils.cropper import CropperInsightFace
except:
    log.warning("Can't load MediaPipe, MediaPipeCropper not available")
try:
    from .liveportrait.utils.cropper import CropperFaceAlignment
except:
    log.warning("Can't load FaceAlignment, CropperFaceAlignment not available")

from .liveportrait.modules.spade_generator import SPADEDecoder
from .liveportrait.modules.warping_network import WarpingNetwork
from .liveportrait.modules.motion_extractor import MotionExtractor
from .liveportrait.modules.appearance_feature_extractor import (
    AppearanceFeatureExtractor,
)
from .liveportrait.modules.stitching_retargeting_network import (
    StitchingRetargetingNetwork,
)
from .liveportrait.utils.camera import get_rotation_matrix
from .liveportrait.utils.crop import _transform_img_kornia


class InferenceConfig:
    def __init__(
        self,
        flag_use_half_precision=True,
        flag_lip_zero=True,
        lip_zero_threshold=0.03,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        flag_stitching=True,
        input_shape=(256, 256),
        device_id=0,
        flag_do_rot=True,
    ):
        self.flag_use_half_precision = flag_use_half_precision
        self.flag_lip_zero = flag_lip_zero
        self.lip_zero_threshold = lip_zero_threshold
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.input_shape = input_shape
        self.device_id = device_id
        self.flag_do_rot = flag_do_rot
        
class DownloadAndLoadLivePortraitModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "precision": (
                    [
                        "fp16",
                        "fp32",
                        "auto",
                    ],
                    {"default": "auto"},
                ),
                "mode": (
                    [
                        "human",
                        "animal",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("LIVEPORTRAITPIPE",)
    RETURN_NAMES = ("live_portrait_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "LivePortrait"

    def loadmodel(self, precision="fp16", mode="human"):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        if precision == 'auto':
            try:
                if mm.is_device_mps(device):
                    log.info("LivePortrait using fp32 for MPS")
                    dtype = 'fp32'
                elif mm.should_use_fp16():
                    log.info("LivePortrait using fp16")
                    dtype = 'fp16'
                else:
                    log.info("LivePortrait using fp32")
                    dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
        else:
            dtype = precision
            log.info(f"LivePortrait using {dtype}")

        pbar = comfy.utils.ProgressBar(3)

        base_bath = os.path.join(folder_paths.models_dir, "liveportrait")
        if mode == "human":
            model_path = base_bath
        else:
            model_path = os.path.join(base_bath, "animal")
      
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/LivePortrait_safetensors",
                ignore_patterns=["*landmark_model.pth*","*animal*"] if mode == "human" else ["*landmark_model.pth*"],
                local_dir=base_bath,
                local_dir_use_symlinks=False,
            )

        model_config_path = os.path.join(
            script_directory, "liveportrait", "config", "models.yaml"
        )
        with open(model_config_path, "r") as file:
            model_config = yaml.safe_load(file)

        feature_extractor_path = os.path.join(
            model_path, "appearance_feature_extractor.safetensors"
        )
        motion_extractor_path = os.path.join(model_path, "motion_extractor.safetensors")
        warping_module_path = os.path.join(model_path, "warping_module.safetensors")
        spade_generator_path = os.path.join(model_path, "spade_generator.safetensors")
        stitching_retargeting_path = os.path.join(
            model_path, "stitching_retargeting_module.safetensors"
        )

        # init F
        model_params = model_config["model_params"][
            "appearance_feature_extractor_params"
        ]
        self.appearance_feature_extractor = AppearanceFeatureExtractor(
            **model_params
        ).to(device)
        self.appearance_feature_extractor.load_state_dict(
            comfy.utils.load_torch_file(feature_extractor_path)
        )
        self.appearance_feature_extractor.eval()
        log.info("Load appearance_feature_extractor done.")
        pbar.update(1)
        # init M
        model_params = model_config["model_params"]["motion_extractor_params"]
        self.motion_extractor = MotionExtractor(**model_params).to(device)
        self.motion_extractor.load_state_dict(
            comfy.utils.load_torch_file(motion_extractor_path)
        )
        self.motion_extractor.eval()
        log.info("Load motion_extractor done.")
        pbar.update(1)
        # init W
        model_params = model_config["model_params"]["warping_module_params"]
        self.warping_module = WarpingNetwork(**model_params).to(device)
        self.warping_module.load_state_dict(
            comfy.utils.load_torch_file(warping_module_path)
        )
        self.warping_module.eval()
        log.info("Load warping_module done.")
        pbar.update(1)
        # init G
        model_params = model_config["model_params"]["spade_generator_params"]
        self.spade_generator = SPADEDecoder(**model_params).to(device)
        self.spade_generator.load_state_dict(
            comfy.utils.load_torch_file(spade_generator_path)
        )
        self.spade_generator.eval()
        log.info("Load spade_generator done.")
        pbar.update(1)

        def filter_checkpoint_for_model(checkpoint, prefix):
            """Filter and adjust the checkpoint dictionary for a specific model based on the prefix."""
            # Create a new dictionary where keys are adjusted by removing the prefix and the model name
            filtered_checkpoint = {
                key.replace(prefix + "_module.", ""): value
                for key, value in checkpoint.items()
                if key.startswith(prefix)
            }
            return filtered_checkpoint

        config = model_config["model_params"]["stitching_retargeting_module_params"]
        checkpoint = comfy.utils.load_torch_file(stitching_retargeting_path)

        stitcher_prefix = "retarget_shoulder"
        stitcher_checkpoint = filter_checkpoint_for_model(checkpoint, stitcher_prefix)
        stitcher = StitchingRetargetingNetwork(**config.get("stitching"))
        stitcher.load_state_dict(stitcher_checkpoint)
        stitcher = stitcher.to(device).eval()

        lip_prefix = "retarget_mouth"
        lip_checkpoint = filter_checkpoint_for_model(checkpoint, lip_prefix)
        retargetor_lip = StitchingRetargetingNetwork(**config.get("lip"))
        retargetor_lip.load_state_dict(lip_checkpoint)
        retargetor_lip = retargetor_lip.to(device).eval()

        eye_prefix = "retarget_eye"
        eye_checkpoint = filter_checkpoint_for_model(checkpoint, eye_prefix)
        retargetor_eye = StitchingRetargetingNetwork(**config.get("eye"))
        retargetor_eye.load_state_dict(eye_checkpoint)
        retargetor_eye = retargetor_eye.to(device).eval()
        log.info("Load stitching_retargeting_module done.")

        self.stich_retargeting_module = {
            "stitching": stitcher,
            "lip": retargetor_lip,
            "eye": retargetor_eye,
        }

        pipeline = LivePortraitPipeline(
            self.appearance_feature_extractor,
            self.motion_extractor,
            self.warping_module,
            self.spade_generator,
            self.stich_retargeting_module,
            InferenceConfig(
                device_id=device,
                flag_use_half_precision=True if precision == "fp16" else False,
            ),
        )

        return (pipeline,)


class LivePortraitProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "pipeline": ("LIVEPORTRAITPIPE",),
            "crop_info": ("CROPINFO", {"default": {}}),
            "source_image": ("IMAGE",),
            "driving_images": ("IMAGE",),
            "lip_zero": ("BOOLEAN", {"default": False}),
            "lip_zero_threshold": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 4.0, "step": 0.001}),
            "stitching": ("BOOLEAN", {"default": True}),
            "delta_multiplier": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
            "mismatch_method": (
                    [
                        "constant",
                        "cycle",
                        "mirror",
                        "cut"
                    ],
                    {"default": "constant"},
                ),
            
            "relative_motion_mode": (
                    [
                        "relative",
                        "source_video_smoothed",
                        "relative_rotation_only",
                        "single_frame",
                        "off"
                    ],
                ),
            "driving_smooth_observation_variance": ("FLOAT", {"default": 3e-6, "min": 1e-11, "max": 1e-2, "step": 1e-11}),
            },
            
            "optional": {
                "opt_retargeting_info": ("RETARGETINGINFO", {"default": None}),
                "expression_friendly": ("BOOLEAN", {"default": False}),
                "expression_friendly_multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "LP_OUT",
    )
    RETURN_NAMES = (
        "cropped_image",
        "output",
    )
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(
        self,
        source_image: torch.Tensor,
        driving_images: torch.Tensor,
        crop_info: dict,
        pipeline: LivePortraitPipeline,
        lip_zero: bool,
        lip_zero_threshold: float,
        stitching: bool,
        relative_motion_mode: str,
        driving_smooth_observation_variance: float,
        delta_multiplier: float = 1.0,
        mismatch_method: str = "constant",
        opt_retargeting_info: dict = None,
        expression_friendly: bool = False,
        expression_friendly_multiplier: float = 1.0,
    ):
        if driving_images.shape[0] < source_image.shape[0]:
            raise ValueError("The number of driving images should be larger than the number of source images.")
        if expression_friendly and source_image.shape[0] > 1:
            raise ValueError("expression_friendly works only with single source image")
        
        if opt_retargeting_info is not None:
            pipeline.live_portrait_wrapper.cfg.flag_eye_retargeting = opt_retargeting_info["eye_retargeting"]
            pipeline.live_portrait_wrapper.cfg.eyes_retargeting_multiplier = (opt_retargeting_info["eyes_retargeting_multiplier"])
            pipeline.live_portrait_wrapper.cfg.flag_lip_retargeting = opt_retargeting_info["lip_retargeting"]
            pipeline.live_portrait_wrapper.cfg.lip_retargeting_multiplier = (opt_retargeting_info["lip_retargeting_multiplier"])
            driving_landmarks = opt_retargeting_info["driving_landmarks"]
        else:
            pipeline.live_portrait_wrapper.cfg.flag_eye_retargeting = False
            pipeline.live_portrait_wrapper.cfg.eyes_retargeting_multiplier = 1.0
            pipeline.live_portrait_wrapper.cfg.flag_lip_retargeting = False
            pipeline.live_portrait_wrapper.cfg.lip_retargeting_multiplier = 1.0
            driving_landmarks = None

        pipeline.live_portrait_wrapper.cfg.flag_stitching = stitching
        pipeline.live_portrait_wrapper.cfg.flag_lip_zero = lip_zero
        pipeline.live_portrait_wrapper.cfg.lip_zero_threshold = lip_zero_threshold

        if lip_zero and opt_retargeting_info is not None:
            log.warning("Warning: lip_zero only has an effect with lip or eye retargeting")
        
        if driving_images.shape[1] != 256 or driving_images.shape[2] != 256:
            driving_images_256 = comfy.utils.common_upscale(driving_images.permute(0, 3, 1, 2), 256, 256, "lanczos", "disabled")
        else:
            driving_images_256 = driving_images.permute(0, 3, 1, 2)

        if pipeline.live_portrait_wrapper.cfg.flag_use_half_precision:
            driving_images_256 = driving_images_256.to(torch.float16)

        out = pipeline.execute(
            driving_images_256, 
            crop_info, 
            driving_landmarks,
            delta_multiplier,
            relative_motion_mode,
            driving_smooth_observation_variance,
            mismatch_method,
            expression_friendly=expression_friendly,
            driving_multiplier=expression_friendly_multiplier,
        )

        total_frames = len(out["out_list"])
      
        if total_frames > 1:
            cropped_image_list = []
            for i in (range(total_frames)):
                if not out["out_list"][i]:
                    cropped_image_list.append(torch.zeros(1, 512, 512, 3, dtype=torch.float32, device = "cpu"))
                else:
                    cropped_image = torch.clamp(out["out_list"][i]["out"], 0, 1).permute(0, 2, 3, 1).cpu()
                    cropped_image_list.append(cropped_image)

            cropped_out_tensors = torch.cat(cropped_image_list, dim=0)
        else:
            cropped_out_tensors = torch.clamp(out["out_list"][0]["out"], 0, 1).permute(0, 2, 3, 1)
      
        return (cropped_out_tensors, out,)
    
class LivePortraitComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "source_image": ("IMAGE",),
            "cropped_image": ("IMAGE",),
            "liveportrait_out": ("LP_OUT", ),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "full_images",
        "mask",
    )
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(self, source_image, cropped_image, liveportrait_out, mask=None):
        mm.soft_empty_cache()
        gc.collect()
        device = mm.get_torch_device()
        if mm.is_device_mps(device): 
            device = torch.device('cpu') #this function returns NaNs on MPS, defaulting to CPU

        B, H, W, C = source_image.shape
        source_image = source_image.permute(0, 3, 1, 2) # B,H,W,C -> B,C,H,W
        cropped_image = cropped_image.permute(0, 3, 1, 2)

        if mask is not None:
            if len(mask.size())==2:
                crop_mask = mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)
            else:    
                crop_mask = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        else:
            log.info("Using default mask template")
            crop_mask = cv2.imread(os.path.join(script_directory, "liveportrait", "utils", "resources", "mask_template.png"), cv2.IMREAD_COLOR)
            crop_mask = torch.from_numpy(crop_mask)
            crop_mask = crop_mask.unsqueeze(0).float() / 255.0

        crop_info = liveportrait_out["crop_info"]
        composited_image_list = []
        out_mask_list = []

        total_frames = len(liveportrait_out["out_list"])
        log.info(f"Total frames: {total_frames}")

        pbar = comfy.utils.ProgressBar(total_frames)
        for i in tqdm(range(total_frames), desc='Compositing..', total=total_frames):
            safe_index = min(i, len(crop_info["crop_info_list"]) - 1)

            if liveportrait_out["mismatch_method"] == "cut":
                source_frame = source_image[safe_index].unsqueeze(0).to(device)
            else:
                source_frame = _get_source_frame(source_image, i, liveportrait_out["mismatch_method"]).unsqueeze(0).to(device)

            if not liveportrait_out["out_list"][i]:
                composited_image_list.append(source_frame.cpu())
                out_mask_list.append(torch.zeros((1, 3, H, W), device="cpu"))
            else:
                cropped_image = torch.clamp(liveportrait_out["out_list"][i]["out"], 0, 1).permute(0, 2, 3, 1)

                # Transform and blend             
                cropped_image_to_original = _transform_img_kornia(
                    cropped_image,
                    crop_info["crop_info_list"][safe_index]["M_c2o"],
                    dsize=(W, H),
                    device=device
                    )

                mask_ori = _transform_img_kornia(
                    crop_mask[min(i,len(crop_mask)-1)].unsqueeze(0),
                    crop_info["crop_info_list"][safe_index]["M_c2o"],
                    dsize=(W, H),
                    device=device
                    )
               
                cropped_image_to_original_blend = torch.clip(
                        mask_ori * cropped_image_to_original + (1 - mask_ori) * source_frame, 0, 1
                        )

                composited_image_list.append(cropped_image_to_original_blend.cpu())
                out_mask_list.append(mask_ori.cpu())
            pbar.update(1)

        full_tensors_out = torch.cat(composited_image_list, dim=0)
        full_tensors_out = full_tensors_out.permute(0, 2, 3, 1)

        mask_tensors_out = torch.cat(out_mask_list, dim=0)
        mask_tensors_out = mask_tensors_out[:, 0, :, :]
        
        return (
            full_tensors_out.float(), 
            mask_tensors_out.float()
            )
    
def _get_source_frame(source, idx, method):
        if source.shape[0] == 1:
            return source[0]

        if method == "constant":
            return source[min(idx, source.shape[0] - 1)]
        elif method == "cycle":
            return source[idx % source.shape[0]]
        elif method == "mirror":
            cycle_length = 2 * source.shape[0] - 2
            mirror_idx = idx % cycle_length
            if mirror_idx >= source.shape[0]:
                mirror_idx = cycle_length - mirror_idx
            return source[mirror_idx]

class LivePortraitLoadCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "onnx_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML'], {
                        "default": 'CPU'
                    }),
            "keep_model_loaded": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LPCROPPER",)
    RETURN_NAMES = ("cropper",)
    FUNCTION = "crop"
    CATEGORY = "LivePortrait"

    def crop(self, onnx_device, keep_model_loaded, detection_threshold=0.5):
        cropper_init_config = {
            'keep_model_loaded': keep_model_loaded,
            'onnx_device': onnx_device,
            'detection_threshold': detection_threshold
        }
        
        if not hasattr(self, 'cropper') or self.cropper is None or self.current_config != cropper_init_config:
            self.current_config = cropper_init_config
            self.cropper = CropperInsightFace(**cropper_init_config)

        return (self.cropper,)

class LivePortraitLoadMediaPipeCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "landmarkrunner_onnx_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML', 'torch_gpu'], {
                        "default": 'CPU'
                    }),
            "keep_model_loaded": ("BOOLEAN", {"default": True})
            },           
        }

    RETURN_TYPES = ("LPCROPPER",)
    RETURN_NAMES = ("cropper",)
    FUNCTION = "crop"
    CATEGORY = "LivePortrait"

    def crop(self, landmarkrunner_onnx_device, keep_model_loaded):
        cropper_init_config = {
            'keep_model_loaded': keep_model_loaded,
            'onnx_device': landmarkrunner_onnx_device
        }
        
        if not hasattr(self, 'cropper') or self.cropper is None or self.current_config != cropper_init_config:
            self.current_config = cropper_init_config
            self.cropper = CropperMediaPipe(**cropper_init_config)

        return (self.cropper,)
    
class LivePortraitLoadFaceAlignmentCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "face_detector": (
                    ['blazeface', 'blazeface_back_camera', 'sfd'], {
                        "default": 'blazeface_back_camera'
                    }),

            "landmarkrunner_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML', 'torch_gpu'], {
                        "default": 'torch_gpu'
                    }),
            "face_detector_device": (
                    ['cuda', 'cpu', 'mps'], {
                        "default": 'cuda'
                    }),

            "face_detector_dtype": (
                    [
                        "fp16",
                        "bf16",
                        "fp32",
                    ],
                    {"default": "fp16"},
                ),
            "keep_model_loaded": ("BOOLEAN", {"default": True})

            },           
        }

    RETURN_TYPES = ("LPCROPPER",)
    RETURN_NAMES = ("cropper",)
    FUNCTION = "crop"
    CATEGORY = "LivePortrait"

    def crop(self, landmarkrunner_device, keep_model_loaded, face_detector, face_detector_device, face_detector_dtype):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[face_detector_dtype]
        cropper_init_config = {
            'keep_model_loaded': keep_model_loaded,
            'onnx_device': landmarkrunner_device,
            'face_detector_device': face_detector_device,
            'face_detector': face_detector,
            'face_detector_dtype': dtype
        }
        
        if not hasattr(self, 'cropper') or self.cropper is None or self.current_config != cropper_init_config:
            self.current_config = cropper_init_config
            self.cropper = CropperFaceAlignment(**cropper_init_config)

        return (self.cropper,)
    
class LivePortraitCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("LIVEPORTRAITPIPE",),
            "cropper": ("LPCROPPER",),
            "source_image": ("IMAGE",),
            "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
            "scale": ("FLOAT", {"default": 2.3, "min": 1.0, "max": 4.0, "step": 0.01}),
            "vx_ratio": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            "vy_ratio": ("FLOAT", {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.001}),
            "face_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            "face_index_order": (
                    [
                        'large-small', 
                        'left-right', 
                        'right-left',
                        'top-bottom',
                        'bottom-top',
                        'small-large',
                        'distance-from-retarget-face'
                     ],
                    ),
            "rotate": ("BOOLEAN", {"default": True}),
            },           
        }

    RETURN_TYPES = ("IMAGE", "CROPINFO",)
    RETURN_NAMES = ("cropped_image", "crop_info",)
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(self, pipeline, cropper, source_image, dsize, scale, vx_ratio, vy_ratio, face_index, face_index_order, rotate):
        source_image_np = (source_image.contiguous() * 255).byte().numpy()

        # Initialize lists
        crop_info_list = []
        cropped_images_list = []
        source_info = []
        source_rot_list = []
        f_s_list = []
        x_s_list = []
        
        # Initialize a progress bar for the combined operation
        pbar = comfy.utils.ProgressBar(len(source_image_np))
        for i in tqdm(range(len(source_image_np)), desc='Detecting, cropping, and processing..', total=len(source_image_np)):
            # Cropping operation
            crop_info, cropped_image_256 = cropper.crop_single_image(source_image_np[i], dsize, scale, vy_ratio, vx_ratio, face_index, face_index_order, rotate)
            
            # Processing source images
            if crop_info:
                crop_info_list.append(crop_info)

                cropped_images_list.append(cropped_image_256)

                I_s = pipeline.live_portrait_wrapper.prepare_source(cropped_image_256)

                x_s_info = pipeline.live_portrait_wrapper.get_kp_info(I_s)
                source_info.append(x_s_info)

                x_s = pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)
                x_s_list.append(x_s)

                R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
                source_rot_list.append(R_s)

                f_s = pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
                f_s_list.append(f_s)

                del I_s
                
            else:
                log.warning(f"Warning: No face detected on frame {str(i)}, skipping") 
                cropped_images_list.append(np.zeros((256, 256, 3), dtype=np.uint8))
                crop_info_list.append(None)
                f_s_list.append(None)
                x_s_list.append(None)
                source_info.append(None)
                source_rot_list.append(None)
        
            # Update progress bar
            pbar.update(1)
        
        cropped_tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in cropped_images_list])
            / 255
        )
        
        crop_info_dict = {
            'crop_info_list': crop_info_list,
            'source_rot_list': source_rot_list,
            'f_s_list': f_s_list,
            'x_s_list': x_s_list,
            'source_info': source_info
        }

        return (cropped_tensors_out, crop_info_dict)

class LivePortraitRetargeting:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "driving_crop_info": ("CROPINFO", {"default": []}),
            "eye_retargeting": ("BOOLEAN", {"default": False}),
            "eyes_retargeting_multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
            "lip_retargeting": ("BOOLEAN", {"default": False}),
            "lip_retargeting_multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("RETARGETINGINFO",)
    RETURN_NAMES = ("retargeting_info",)
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(self, driving_crop_info, eye_retargeting, eyes_retargeting_multiplier, lip_retargeting, lip_retargeting_multiplier):

        driving_landmarks = []
        for crop in driving_crop_info["crop_info_list"]:
            driving_landmarks.append(crop['lmk_crop'])
                          
        retargeting_info = {
            'eye_retargeting': eye_retargeting,
            'eyes_retargeting_multiplier': eyes_retargeting_multiplier,
            'lip_retargeting': lip_retargeting,
            'lip_retargeting_multiplier': lip_retargeting_multiplier,
            'driving_landmarks': driving_landmarks
        }

        return (retargeting_info,)


class KeypointsToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "crop_info": ("CROPINFO", {"default": []}),
            "draw_lines": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("keypoints_image",)
    FUNCTION = "drawkeypoints"
    CATEGORY = "LivePortrait"

    def drawkeypoints(self, crop_info, draw_lines):
        #           left upper eye | left lower eye | right upper eye | right lower eye | upper lip top | lower lip bottom | upper lip bottom | lower lip top | jawline         | left eyebrow | right eyebrow | nose            | left pupil    | right pupil  |  nose center
        indices = [                  12,               24,              37,               48,             66,                85,                96,             108,              145,           165,            185,             197,             198,            199,          203]
        colorlut = [(0, 0, 255),     (0, 255, 0),     (0, 0, 255),      (0, 255, 0),      (255, 0, 0),    (255, 0, 255),     (255, 255, 0),     (0, 255, 255),  (128, 128, 128), (128, 128, 0), (128, 128, 0),   (0,128,128),    (255, 255,255),   (255, 255,255), (255,255,255)]
        colors = []
        c = 0
        for i in range(203):
            if i == indices[c]:
                c+=1
            colors.append(colorlut[c])
        try:
            height, width = crop_info["crop_info_list"][0]['input_image_size']
        except:
            height, width = 512, 512
        keypoints_img_list = []
        pbar = comfy.utils.ProgressBar(len(crop_info))
        for crop in crop_info["crop_info_list"]:
            if crop:
                keypoints = crop['lmk_crop'].copy()
                blank_image = np.zeros((height, width, 3), dtype=np.uint8) * 255
                
                if draw_lines:
                    start_idx = 0
                    for end_idx in indices:
                        color = colors[start_idx]
                        for i in range(start_idx, end_idx - 1):
                            pt1 = tuple(map(int, keypoints[i]))
                            pt2 = tuple(map(int, keypoints[i+1]))
                            if all(0 <= c < d for c, d in zip(pt1 + pt2, (width, height) * 2)):
                                cv2.line(blank_image, pt1, pt2, color, thickness=1)
                        if end_idx == start_idx +1:
                            x,y = keypoints[start_idx]
                            cv2.circle(blank_image, (int(x), int(y)), radius=1, thickness=-1, color=colors[start_idx])
                              
                        start_idx = end_idx
                else:
                    for index, (x, y) in enumerate(keypoints):
                        cv2.circle(blank_image, (int(x), int(y)), radius=1, thickness=-1, color=colors[index])
                
                keypoints_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
            else:
                keypoints_image = np.zeros((height, width, 3), dtype=np.uint8) * 255
            keypoints_img_list.append(keypoints_image)
            pbar.update(1)

        keypoints_img_tensor = (
            torch.stack([torch.from_numpy(np_array) for np_array in keypoints_img_list]) / 255).float()

        return (keypoints_img_tensor,)

class KeypointScaler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "crop_info": ("CROPINFO", {"default": {}}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001}),
                "offset_x": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),

            }
        }

    RETURN_TYPES = ("CROPINFO", "IMAGE",)
    RETURN_NAMES = ("crop_info", "keypoints_image",)
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(self, crop_info, offset_x, offset_y, scale):

        keypoints = crop_info['crop_info']['lmk_crop'].copy()

        # Create an offset array
        # Calculate the centroid of the keypoints
        centroid = keypoints.mean(axis=0)

        # Translate keypoints to origin by subtracting the centroid
        translated_keypoints = keypoints - centroid

        # Scale the translated keypoints
        scaled_keypoints = translated_keypoints * scale

        # Translate scaled keypoints back to original position and then apply the offset
        final_keypoints = scaled_keypoints + centroid + np.array([offset_x, offset_y])

        crop_info['crop_info']['lmk_crop'] = final_keypoints #fix this

        # Draw each landmark as a circle
        width, height = 512, 512
        blank_image = np.zeros((height, width, 3), dtype=np.uint8) * 255
        for (x, y) in final_keypoints:
            # Ensure the coordinates are within the dimensions of the blank image
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(blank_image, (int(x), int(y)), radius=2, color=(0, 0, 255))

        keypoints_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        keypoints_image_tensor = torch.from_numpy(keypoints_image) / 255
        keypoints_image_tensor = keypoints_image_tensor.unsqueeze(0).cpu().float()
        
        return (crop_info, keypoints_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadLivePortraitModels": DownloadAndLoadLivePortraitModels,
    "LivePortraitProcess": LivePortraitProcess,
    "LivePortraitCropper": LivePortraitCropper,
    "LivePortraitRetargeting": LivePortraitRetargeting,
    #"KeypointScaler": KeypointScaler,
    "KeypointsToImage": KeypointsToImage,
    "LivePortraitLoadCropper": LivePortraitLoadCropper,
    "LivePortraitLoadMediaPipeCropper": LivePortraitLoadMediaPipeCropper,
    "LivePortraitLoadFaceAlignmentCropper": LivePortraitLoadFaceAlignmentCropper,
    "LivePortraitComposite": LivePortraitComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLivePortraitModels": "(Down)Load LivePortraitModels",
    "LivePortraitProcess": "LivePortrait Process",
    "LivePortraitCropper": "LivePortrait Cropper",
    "LivePortraitRetargeting": "LivePortrait Retargeting",
    #"KeypointScaler": "KeypointScaler",
    "KeypointsToImage": "LivePortrait KeypointsToImage",
    "LivePortraitLoadCropper": "LivePortrait Load InsightFaceCropper",
    "LivePortraitLoadMediaPipeCropper": "LivePortrait Load MediaPipeCropper",
    "LivePortraitLoadFaceAlignmentCropper": "LivePortrait Load FaceAlignmentCropper",
    "LivePortraitComposite": "LivePortrait Composite",
    }
