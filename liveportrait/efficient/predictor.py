from .utils.onnx_driver import ONNXEngine
import numpy as np


class EfficientLivePortraitPredictor:
    def __init__(self, use_tensorrt=False, half=False, **kwargs):
        super().__init__()
        self.use_tensorrt = use_tensorrt
        self.half = half
        self.cfg = kwargs
        if self.use_tensorrt:
            from .utils.tensorrt_driver import TensorRTEngine
            self.trt_engine = TensorRTEngine(self.half, **kwargs)
        else:
            self.onnx_engine = ONNXEngine().initialize_sessions(self.cfg)

    def run_time(self, engine_name, task, inputs_onnx=None, inputs_tensorrt=None):
        """
        Run inference using either TensorRT or ONNX Runtime based on the configuration.

        Args:
        - engine_name (str): Name of the engine/model.
        - task (str): The task or model session name.
        - inputs_onnx (dict): Input dict for inference.
        - inputs_tensorrt(np.array or tensor): Input for inference TensorRT
        Returns:
        - The outputs from the inference.
        """
        if self.use_tensorrt:
            return self.trt_engine.inference_tensorrt(engine_name, inputs_tensorrt)
        else:
            return self.inference_onnx(task, inputs_onnx)

    def inference_onnx(self, task, inputs):
        """
        Perform inference using ONNX Runtime.

        Args:
        - task (str): The name of the task/model to use for inference.
        - inputs (list or array): A list or array of input tensors.

        Returns:
        - List: The outputs of the inference.
        """
        session = self.onnx_engine[task]
        outputs = session.run(None, inputs)
        return outputs
