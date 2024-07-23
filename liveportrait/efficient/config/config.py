import os
import requests
from dataclasses import dataclass, asdict
from typing import Literal, Tuple
from tqdm import tqdm
import torch.cuda
import yaml

# Define the URLs for the model files
MODEL_URLS = {
    'live_portrait': {
        'grid_sample_3d': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/libgrid_sample_3d_plugin.so?download=true',
        'F_onnx': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/appearance_feature_extractor.onnx?download=true',
        'M_onnx': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/motion_extractor.onnx?download=true',
        'GW_onnx': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/generator_fix_grid.onnx?download=true',
        'S_onnx': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/stitching.onnx?download=true',
        'SE_onnx': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/stitching_eye.onnx?download=true',
        'SL_onnx': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/stitching_lip.onnx?download=true',
        # TensorRT FP32
        'F_rt': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP32/resolve/main/appearance_feature_extractor_fp32.engine?download=true',
        'M_rt': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP32/resolve/main/motion_extractor_fp32.engine?download=true',
        'GW_rt': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP32/resolve/main/generator_fp32.engine?download=true',
        'S_rt': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP32/resolve/main/stitching_fp32.engine?download=true',
        'SE_rt': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP32/resolve/main/stitching_eye_fp32.engine?download=true',
        'SL_rt': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP32/resolve/main/stitching_lip_fp32.engine?download=true',
        # TensorRT FP16
        'F_rt_half': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP16/resolve/main/appearance_feature_extractor_fp16.engine?download=true',
        'M_rt_half': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP16/resolve/main/motion_extractor_fp16.engine?download=true',
        'GW_rt_half': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP16/resolve/main/generator_fp16.engine?download=true',
        'S_rt_half': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP16/resolve/main/stitching_fp16.engine?download=true',
        'SE_rt_half': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP16/resolve/main/stitching_eye_fp16.engine?download=true',
        'SL_rt_half': 'https://huggingface.co/myn0908/Live-Portrait-TensorRT-FP16/resolve/main/stitching_lip_fp16.engine?download=true'
    },
    'insightface': {
        'arc_face': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/w600k_r50.onnx?download=true',
        '2d106det': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/2d106det.onnx?download=true',
        'det_10g': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/det_10g.onnx?download=true',
        'landmark': 'https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/landmark.onnx?download=true'
    }
}


# Function to download a file from a URL and save it locally
def downloading(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        return outf


def get_efficient_live_portrait():
    # Download the models and save them in the current working directory
    current_dir = os.getcwd()
    face_dir = os.path.join(current_dir, 'live_portrait_weights')
    model_paths = {}
    for main_key, sub_dict in MODEL_URLS.items():
        dir_path = os.path.join(current_dir, 'live_portrait_weights', main_key)
        os.makedirs(dir_path, exist_ok=True)
        model_paths[main_key] = {}
        for sub_key, url in sub_dict.items():
            filename = url.split('/')[-1].split('?')[0]
            save_path = os.path.join(dir_path, filename)
            downloading(url, save_path)
            model_paths[main_key][sub_key] = save_path
        print('Downloaded successfully and already saved')
    return model_paths, face_dir


@dataclass(repr=False)  # use repr from PrintableConfig
class Config:
    model_paths, face_dir = get_efficient_live_portrait()
    grid_sample_3d: str = model_paths['live_portrait']['grid_sample_3d']
    # ONNX
    checkpoint_F: str = model_paths['live_portrait']['F_onnx']  # path to checkpoint
    checkpoint_M: str = model_paths['live_portrait']['M_onnx']  # path to checkpoint
    checkpoint_GW: str = model_paths['live_portrait']['GW_onnx']
    checkpoint_S: str = model_paths['live_portrait']['S_onnx']  # path to checkpoint
    checkpoint_SE: str = model_paths['live_portrait']['SE_onnx']
    checkpoint_SL: str = model_paths['live_portrait']['SL_onnx']

    # TensorRT FP32
    F_rt: str = model_paths['live_portrait']['F_rt']  # path to checkpoint
    M_rt: str = model_paths['live_portrait']['M_rt']  # path to checkpoint
    GW_rt: str = model_paths['live_portrait']['GW_rt']  # path to checkpoint
    S_rt: str = model_paths['live_portrait']['S_rt']  # path to checkpoint
    SE_rt: str = model_paths['live_portrait']['SE_rt']
    SL_rt: str = model_paths['live_portrait']['SL_rt']

    # TensorRT FP16
    F_rt_half: str = model_paths['live_portrait']['F_rt_half']  # path to checkpoint
    M_rt_half: str = model_paths['live_portrait']['M_rt_half']  # path to checkpoint
    GW_rt_half: str = model_paths['live_portrait']['GW_rt_half']  # path to checkpoint
    S_rt_half: str = model_paths['live_portrait']['S_rt_half']  # path to checkpoint
    SE_rt_half: str = model_paths['live_portrait']['SE_rt_half']
    SL_rt_half: str = model_paths['live_portrait']['SL_rt_half']

    flag_use_half_precision: bool = True  # whether to use half precision
    flag_lip_zero: bool = True  # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    lip_zero_threshold: float = 0.03
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True  # we recommend setting it to True!
    flag_relative: bool = True  # whether to use relative motion
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    flag_do_crop: bool = True  # whether to crop the source portrait to the face-cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
    flag_write_result: bool = True  # whether to write output video
    flag_write_gif: bool = False

    anchor_frame: int = 0  # set this value if find_best_frame is True

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    output_fps: int = 30  # fps for output video
    crf: int = 15  # crf for output video
    mask_crop: str = 'None'
    size_gif: int = 256
    ref_max_shape: int = 1280
    ref_shape_n: int = 2

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # crop config
    ckpt_landmark: str = model_paths['insightface']['landmark']
    ckpt_arc_face: str = model_paths['insightface']['arc_face']
    ckpt_landmark_106: str = model_paths['insightface']['2d106det']
    ckpt_det: str = model_paths['insightface']['det_10g']
    ckpt_face: str = face_dir
    dsize: int = 512  # crop size
    scale: float = 2.3  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down


# Function to save the configuration to a YAML file
def save_config_to_yaml(filename="efficient-live-portrait.yaml"):
    # Define the path where the YAML file will be saved
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        # Save the configuration to the YAML file
        with open(file_path, 'w') as file:
            yaml.safe_dump(asdict(Config()), file)
    return file_path
