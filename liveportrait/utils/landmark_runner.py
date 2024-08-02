# coding: utf-8

import cv2#; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np

from .timer import Timer
from .crop import crop_image, _transform_pts
import folder_paths
import os
def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)

class LandmarkRunner(object):
    """landmark runner"""
    def __init__(self, **kwargs):
        ckpt_path = kwargs.get('ckpt_path')
        onnx_provider = kwargs.get('onnx_provider', 'cuda')
        device_id = kwargs.get('device_id', 0)
        self.dsize = kwargs.get('dsize', 224)
        self.timer = Timer()

        import onnxruntime

        if onnx_provider.lower() == 'cuda':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    ('CUDAExecutionProvider', {'device_id': device_id})
                ]
            )
        else:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=['CPUExecutionProvider'],
                sess_options=opts
            )

    def _run(self, inp):
        out = self.session.run(None, {'input': inp})
        return out

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct, img_crop_rgb = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        pts = to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize  # scale to 0-224
        pts = _transform_pts(pts, M=crop_dct['M_c2o'])
        del crop_dct, img_crop_rgb
        return {
            'pts': pts,  # 2d landmarks 203 points
        }

    def warmup(self):
        self.timer.tic()

        dummy_image = np.zeros((1, 3, self.dsize, self.dsize), dtype=np.float32)

        _ = self._run(dummy_image)

        elapse = self.timer.toc()
        print(f'LandmarkRunner warmup time: {elapse:.3f}s')

class LandmarkRunnerTorch(object):
    """landmark runner torch version"""
    def __init__(self, **kwargs):
        self.device = kwargs.get('device_id', 0)
        self.dsize = kwargs.get('dsize', 224)
        ckpt_path = kwargs.get('ckpt_path')

        if not os.path.exists(ckpt_path):
            download_path = os.path.join(folder_paths.models_dir, "liveportrait")
            print(f"Downloading model to: {ckpt_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/LivePortrait_safetensors",
                allow_patterns="*landmark_model.pth*",
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )
        self.model = torch.load(ckpt_path).to(self.device)

    def _run(self, inp):
        input = torch.from_numpy(inp).to(self.device)
        out = self.model(input)
        return out

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct, img_crop_rgb = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        pts = to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize  # scale to 0-224
        pts = _transform_pts(pts, M=crop_dct['M_c2o'])
        del crop_dct, img_crop_rgb
        return {
            'pts': pts,  # 2d landmarks 203 points
        }