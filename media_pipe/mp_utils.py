import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from . import face_landmark

CUR_DIR = os.path.dirname(__file__)

class LMKExtractor():
    def __init__(self):
        # Create an FaceLandmarker object.
        self.mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE
        base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models','face_landmarker_v2_with_blendshapes.task'))
        base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=self.mode,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1,
                                            min_face_detection_confidence=0.5,
                                            min_face_presence_confidence=0.5,
                                            min_tracking_confidence=0.5)
        self.detector = face_landmark.FaceLandmarker.create_from_options(options)

        det_base_options = python.BaseOptions(model_asset_path=os.path.join(CUR_DIR, 'mp_models','blaze_face_short_range.tflite'))
        det_options = vision.FaceDetectorOptions(base_options=det_base_options)
        self.det_detector = vision.FaceDetector.create_from_options(det_options)
                
    def __call__(self, img):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        try:
            detection_result, _ = self.detector.detect(image)
        except:
            return None
            
        return detection_result.face_landmarks
        