import os
import torch
import yaml
import folder_paths
import comfy.model_management as mm
import comfy.utils

script_directory = os.path.dirname(os.path.abspath(__file__))

from .liveportrait.config.argument_config import ArgumentConfig
from .liveportrait.live_portrait_pipeline import LivePortraitPipeline
from .liveportrait.utils.cropper import Cropper
from .liveportrait.modules.spade_generator import SPADEDecoder
from .liveportrait.modules.warping_network import WarpingNetwork
from .liveportrait.modules.motion_extractor import MotionExtractor
from .liveportrait.modules.appearance_feature_extractor import (
    AppearanceFeatureExtractor,
)
from .liveportrait.modules.stitching_retargeting_network import (
    StitchingRetargetingNetwork,
)


class InferenceConfig:
    def __init__(
        self,
        mask_crop=None,
        flag_use_half_precision=True,
        flag_lip_zero=True,
        lip_zero_threshold=0.03,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        flag_stitching=True,
        flag_relative=True,
        anchor_frame=0,
        input_shape=(256, 256),
        flag_write_result=True,
        flag_pasteback=True,
        ref_max_shape=1280,
        ref_shape_n=2,
        device_id=0,
        flag_do_crop=True,
        flag_do_rot=True,
    ):
        self.flag_use_half_precision = flag_use_half_precision
        self.flag_lip_zero = flag_lip_zero
        self.lip_zero_threshold = lip_zero_threshold
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.flag_relative = flag_relative
        self.anchor_frame = anchor_frame
        self.input_shape = input_shape
        self.flag_write_result = flag_write_result
        self.flag_pasteback = flag_pasteback
        self.ref_max_shape = ref_max_shape
        self.ref_shape_n = ref_shape_n
        self.device_id = device_id
        self.flag_do_crop = flag_do_crop
        self.flag_do_rot = flag_do_rot
        self.mask_crop = mask_crop


class CropConfig:
    def __init__(self, dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125):
        self.dsize = dsize
        self.scale = scale
        self.vx_ratio = vx_ratio
        self.vy_ratio = vy_ratio


class ArgumentConfig:
    def __init__(
        self,
        device_id=0,
        flag_lip_zero=True,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        flag_stitching=True,
        flag_relative=True,
        flag_pasteback=True,
        flag_do_crop=True,
        flag_do_rot=True,
        dsize=512,
        scale=2.3,
        vx_ratio=0,
        vy_ratio=-0.125,
    ):
        self.device_id = device_id
        self.flag_lip_zero = flag_lip_zero
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.flag_relative = flag_relative
        self.flag_pasteback = flag_pasteback
        self.flag_do_crop = flag_do_crop
        self.flag_do_rot = flag_do_rot
        self.dsize = dsize
        self.scale = scale
        self.vx_ratio = vx_ratio
        self.vy_ratio = vy_ratio


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
                    ],
                    {"default": "fp16"},
                ),
            },
        }

    RETURN_TYPES = ("LIVEPORTRAITPIPE",)
    RETURN_NAMES = ("live_portrait_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "LivePortrait"

    def loadmodel(self, precision="fp16"):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        pbar = comfy.utils.ProgressBar(3)

        download_path = os.path.join(folder_paths.models_dir, "liveportrait")
        model_path = os.path.join(download_path)

        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/LivePortrait_safetensors",
                local_dir=download_path,
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
        print("Load appearance_feature_extractor done.")
        pbar.update(1)
        # init M
        model_params = model_config["model_params"]["motion_extractor_params"]
        self.motion_extractor = MotionExtractor(**model_params).to(device)
        self.motion_extractor.load_state_dict(
            comfy.utils.load_torch_file(motion_extractor_path)
        )
        self.motion_extractor.eval()
        print("Load motion_extractor done.")
        pbar.update(1)
        # init W
        model_params = model_config["model_params"]["warping_module_params"]
        self.warping_module = WarpingNetwork(**model_params).to(device)
        self.warping_module.load_state_dict(
            comfy.utils.load_torch_file(warping_module_path)
        )
        self.warping_module.eval()
        print("Load warping_module done.")
        pbar.update(1)
        # init G
        model_params = model_config["model_params"]["spade_generator_params"]
        self.spade_generator = SPADEDecoder(**model_params).to(device)
        self.spade_generator.load_state_dict(
            comfy.utils.load_torch_file(spade_generator_path)
        )
        self.spade_generator.eval()
        print("Load spade_generator done.")
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
        stitcher = stitcher.to(device)
        stitcher.eval()

        lip_prefix = "retarget_mouth"
        lip_checkpoint = filter_checkpoint_for_model(checkpoint, lip_prefix)
        retargetor_lip = StitchingRetargetingNetwork(**config.get("lip"))
        retargetor_lip.load_state_dict(lip_checkpoint)
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        eye_prefix = "retarget_eye"
        eye_checkpoint = filter_checkpoint_for_model(checkpoint, eye_prefix)
        retargetor_eye = StitchingRetargetingNetwork(**config.get("eye"))
        retargetor_eye.load_state_dict(eye_checkpoint)
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()
        print("Load stitching_retargeting_module done.")

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


# OUR CURRENT NODE
class LivePortraitProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("LIVEPORTRAITPIPE",),
                "source_image": ("IMAGE",),
                "driving_images": ("IMAGE",),
                "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "scale": (
                    "FLOAT",
                    {"default": 2.3, "min": 1.0, "max": 4.0, "step": 0.01},
                ),
                "vx_ratio": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "vy_ratio": (
                    "FLOAT",
                    {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "lip_zero": ("BOOLEAN", {"default": True}),
                "eye_retargeting": ("BOOLEAN", {"default": False}),
                "eyes_retargeting_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001},
                ),
                "lip_retargeting": ("BOOLEAN", {"default": False}),
                "lip_retargeting_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.001},
                ),
                "stitching": ("BOOLEAN", {"default": True}),
                "relative": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mismatch_method": (
                    ["repeat", "cycle", "mirror", "nearest"],
                    {"default": "repeat"},
                ),
                "onnx_device": (
                    [
                        "CPU",
                        "CUDA",
                    ],
                    {"default": "CPU"},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "cropped_images",
        "full_images",
    )
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(
        self,
        source_image: torch.Tensor,
        driving_images: torch.Tensor,
        dsize: int,
        scale: float,
        vx_ratio: float,
        vy_ratio: float,
        pipeline: LivePortraitPipeline,
        lip_zero: bool,
        eye_retargeting: bool,
        lip_retargeting: bool,
        stitching: bool,
        relative: bool,
        eyes_retargeting_multiplier: float,
        lip_retargeting_multiplier: float,
        mismatch_method: str = "repeat",
        onnx_device="CUDA",
    ):
        source_np = (source_image * 255).byte().numpy()
        driving_images_np = (driving_images * 255).byte().numpy()

        crop_cfg = CropConfig(
            dsize=dsize,
            scale=scale,
            vx_ratio=vx_ratio,
            vy_ratio=vy_ratio,
        )

        cropper = Cropper(crop_cfg=crop_cfg, provider=onnx_device)
        pipeline.cropper = cropper
        pipeline.live_portrait_wrapper.cfg.flag_eye_retargeting = eye_retargeting
        pipeline.live_portrait_wrapper.cfg.eyes_retargeting_multiplier = (
            eyes_retargeting_multiplier
        )
        pipeline.live_portrait_wrapper.cfg.flag_lip_retargeting = lip_retargeting
        pipeline.live_portrait_wrapper.cfg.lip_retargeting_multiplier = (
            lip_retargeting_multiplier
        )
        pipeline.live_portrait_wrapper.cfg.flag_stitching = stitching
        pipeline.live_portrait_wrapper.cfg.flag_relative = relative
        pipeline.live_portrait_wrapper.cfg.flag_lip_zero = lip_zero

        cropped_out_list = []
        full_out_list = []

        cropped_out_list, full_out_list = pipeline.execute(
            source_np, driving_images_np, mismatch_method
        )
        cropped_tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in cropped_out_list])
            / 255
        )
        full_tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in full_out_list])
            / 255
        )

        return (cropped_tensors_out.cpu().float(), full_tensors_out.cpu().float())


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadLivePortraitModels": DownloadAndLoadLivePortraitModels,
    "LivePortraitProcess": LivePortraitProcess,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadLivePortraitModels": "(Down)Load LivePortraitModels",
}
