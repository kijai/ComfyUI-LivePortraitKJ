# coding: utf-8

import numpy as np
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2#; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from .landmark_runner import LandmarkRunner, LandmarkRunnerTorch

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

class CropperInsightFace(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        provider = kwargs.get('onnx_device', 'CPU')
        detection_threshold = kwargs.get('detection_threshold', 0.5)
        self.landmark_runner = LandmarkRunner(
            ckpt_path=os.path.join(folder_paths.models_dir, 'liveportrait', 'landmark.onnx'),
            onnx_provider=provider,
            device_id=device_id
        )
        self.landmark_runner.warmup()

        from .face_analysis_diy import FaceAnalysisDIY
        self.face_analysis_wrapper = FaceAnalysisDIY(
            name='buffalo_l',
            root=os.path.join(folder_paths.models_dir, 'insightface'),
            providers=[provider + 'ExecutionProvider',]
        )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512), det_thresh=detection_threshold)
        self.face_analysis_wrapper.warmup()

    def crop_single_image(self, img_rgb, dsize, scale, vy_ratio, vx_ratio, face_index, face_index_order, rotate):
        direction = face_index_order

        src_face = self.face_analysis_wrapper.get(
            img_rgb,
            flag_do_landmark_2d_106=True,
            direction=direction
        )

        if len(src_face) == 0:
            ret_dct = {}
            cropped_image_256 = None
            return ret_dct, cropped_image_256

        src_face = src_face[face_index] # choose the index if multiple faces detected
        pts = src_face.landmark_2d_106
       
        # crop the face
        ret_dct, image_crop = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=dsize,
            scale=scale,
            vy_ratio=vy_ratio,
            vx_ratio=vx_ratio,
            rotate=rotate
        )
        # update a 256x256 version for network input or else
        cropped_image_256 = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_AREA)
        del image_crop
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / dsize

        input_image_size = img_rgb.shape[:2]
        ret_dct['input_image_size'] = input_image_size
    
        recon_ret = self.landmark_runner.run(img_rgb, pts)
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct, cropped_image_256
    
class CropperMediaPipe(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        provider = kwargs.get('onnx_device', 'CPU')

        if provider != "torch_gpu":
            self.landmark_runner = LandmarkRunner(
                ckpt_path=os.path.join(folder_paths.models_dir, 'liveportrait', 'landmark.onnx'),
                onnx_provider=provider,
                device_id=device_id
                )
            self.landmark_runner.warmup()
        else:
            self.landmark_runner = LandmarkRunnerTorch(
                    ckpt_path=os.path.join(folder_paths.models_dir, 'liveportrait', 'landmark_model.pth'),
                    onnx_provider=provider,
                    device_id=device_id
                )
        
        from ...media_pipe.mp_utils  import LMKExtractor
        self.lmk_extractor = LMKExtractor()

    def crop_single_image(self, img_rgb, dsize, scale, vy_ratio, vx_ratio, face_index, face_index_order, rotate):
       
        face_result = self.lmk_extractor(img_rgb)

        if face_result is None:
            ret_dct = {}
            cropped_image_256 = None
            return ret_dct, cropped_image_256

        face_landmarks = face_result[face_index]

        lmks = []
        for index in range(len(face_landmarks)):
            x = face_landmarks[index].x * img_rgb.shape[1]
            y = face_landmarks[index].y * img_rgb.shape[0]
            lmks.append([x, y])
        pts = np.array(lmks)
       
        # crop the face
        ret_dct, image_crop = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=dsize,
            scale=scale,
            vy_ratio=vy_ratio,
            vx_ratio=vx_ratio,
            rotate=rotate
        )
        # update a 256x256 version for network input or else
        cropped_image_256 = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_AREA)
        del image_crop
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / dsize

        input_image_size = img_rgb.shape[:2]
        ret_dct['input_image_size'] = input_image_size
    
        recon_ret = self.landmark_runner.run(img_rgb, pts)
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct, cropped_image_256
    
class CropperFaceAlignment(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        provider = kwargs.get('onnx_device', 'CPU')
        face_detector_device = kwargs.get('face_detector_device', 'cuda')
        face_detector = kwargs.get('face_detector', 'blazeface')
        face_detector_dtype = kwargs.get('face_detector_dtype', 'fp16')

        if provider != "torch_gpu":
            self.landmark_runner = LandmarkRunner(
                ckpt_path=os.path.join(folder_paths.models_dir, 'liveportrait', 'landmark.onnx'),
                onnx_provider=provider,
                device_id=device_id
                )
            self.landmark_runner.warmup()
        else:
            self.landmark_runner = LandmarkRunnerTorch(
                    ckpt_path=os.path.join(folder_paths.models_dir, 'liveportrait', 'landmark_model.pth'),
                    onnx_provider=provider,
                    device_id=device_id
                )
            
        from ...face_alignment import FaceAlignment, LandmarksType
        if 'blazeface' in face_detector:
            face_detector_kwargs = {'back_model': face_detector == 'blazeface_back_camera'}
            self.fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=face_detector_device, dtype=face_detector_dtype, face_detector='blazeface', face_detector_kwargs=face_detector_kwargs)
        else:
            self.fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=face_detector_device, dtype=face_detector_dtype, face_detector=face_detector)

    def crop_single_image(self, img_rgb, dsize, scale, vy_ratio, vx_ratio, face_index, face_index_order, rotate):
       
        face_result = self.fa.get_landmarks_from_image(img_rgb)

        if face_result is None:
            ret_dct = {}
            cropped_image_256 = None
            return ret_dct, cropped_image_256

        face_landmarks = face_result[face_index]

        pts = np.array(face_landmarks)
       
        # crop the face
        ret_dct, image_crop = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=dsize,
            scale=scale,
            vy_ratio=vy_ratio,
            vx_ratio=vx_ratio,
            rotate=rotate
        )
        # update a 256x256 version for network input or else
        cropped_image_256 = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_AREA)
        del image_crop
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / dsize

        input_image_size = img_rgb.shape[:2]
        ret_dct['input_image_size'] = input_image_size
    
        recon_ret = self.landmark_runner.run(img_rgb, pts)
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct, cropped_image_256
    
