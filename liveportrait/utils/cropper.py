# coding: utf-8

import numpy as np
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2#; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from .landmark_runner import LandmarkRunner
from .face_analysis_diy import FaceAnalysisDIY
from .crop import crop_image

import folder_paths
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Trajectory:
    start: int = -1
    end: int = -1
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        provider = kwargs.get('onnx_device', 'CPU')
        self.landmark_runner = LandmarkRunner(
            ckpt_path=os.path.join(folder_paths.models_dir, 'liveportrait', 'landmark.onnx'),
            onnx_provider=provider,
            device_id=device_id
        )
        self.landmark_runner.warmup()

        self.face_analysis_wrapper = FaceAnalysisDIY(
            name='buffalo_l',
            root=os.path.join(folder_paths.models_dir, 'insightface'),
            providers=[provider + 'ExecutionProvider',]
        )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512))
        self.face_analysis_wrapper.warmup()

    def crop_single_image(self, img_rgb, dsize, scale, vy_ratio, vx_ratio, face_index, rotate):
        direction = 'large-small'

        src_face = self.face_analysis_wrapper.get(
            img_rgb,
            flag_do_landmark_2d_106=True,
            direction=direction
        )

        if len(src_face) == 0:
            raise Exception("No face detected in the source image!")
        #elif len(src_face) > 1:
        #    print(f'More than one face detected in the image, only pick one face by rule {direction}.')

        src_face = src_face[face_index] # choose the index if multiple faces detected
        pts = src_face.landmark_2d_106
       
        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=dsize,
            scale=scale,
            vy_ratio=vy_ratio,
            vx_ratio=vx_ratio,
            rotate=rotate
        )
        # update a 256x256 version for network input or else
        ret_dct['img_crop_256x256'] = cv2.resize(ret_dct['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / dsize

        input_image_size = img_rgb.shape[:2]
        ret_dct['input_image_size'] = input_image_size
    
        recon_ret = self.landmark_runner.run(img_rgb, pts)
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct