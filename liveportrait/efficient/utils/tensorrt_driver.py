import tensorrt as trt
import pycuda.driver as cuda
import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
import ctypes
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Binding:
    def __init__(self, engine, idx_or_name):
        self.name = idx_or_name if isinstance(idx_or_name, str) else engine.get_tensor_name(idx_or_name)
        if not self.name:
            raise IndexError(f"Binding index out of range: {idx_or_name}")
        self.is_input = engine.get_tensor_mode(self.name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(self.name)
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.BOOL: np.bool_,
        }
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32
        if hasattr(trt.DataType, 'INT64'):
            dtype_map[trt.DataType.INT64] = np.int64
        self.dtype = dtype_map[dtype]
        self.shape = tuple(engine.get_tensor_shape(self.name))
        self._host_buf = None
        self._device_buf = None

    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = cuda.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf

    def get_async(self, stream):
        self.device_buffer.get_async(stream, self.host_buffer)
        return self.host_buffer

    def cleanup(self):
        if self._host_buf is not None:
            del self._host_buf
        if self._device_buf is not None:
            del self._device_buf


class TensorRTEngine:
    def __init__(self, half, **kwargs):
        self.cfg = kwargs
        self.cfx = None
        if kwargs.get("cuda_ctx", None) is None:
            cuda.init()
            self.cfx = cuda.Device(0).make_context()
        else:
            self.cfx = kwargs.get("cuda_ctx")

        if half:
            self.model_paths = {
                #'feature_extractor': self.cfg['F_rt_half'],
                #'motion_extractor': self.cfg['M_rt_half'],
                'generator': "live_portrait_weights\\live_portrait\\warping_spade-fix.engine",
                #'stitching_retargeting': self.cfg['S_rt_half'],
                #'stitching_retargeting_eye': self.cfg['SE_rt_half'],
                #'stitching_retargeting_lip': self.cfg['SL_rt_half']
            }
        else:
            self.model_paths = {
                'feature_extractor': self.cfg['F_rt'],
                'motion_extractor': self.cfg['M_rt'],
                'generator': self.cfg['GW_rt'],
                'stitching_retargeting': self.cfg['S_rt'],
                'stitching_retargeting_eye': self.cfg['SE_rt'],
                'stitching_retargeting_lip': self.cfg['SL_rt']
            }
        self.plugin_path = Path("N:\\AI\\ComfyUI\\live_portrait_weights\\live_portrait\\grid_sample_3d_plugin.dll")
        self.load_plugins(TRT_LOGGER)
        self.engines = {}
        self.contexts = {}
        self.bindings = {}
        self.binding_addresses = {}
        self.inputs = {}
        self.outputs = {}
        self.stream = cuda.Stream()
        self.initialize_engines()

    def load_plugins(self, logger: trt.Logger):
        ctypes.CDLL(self.plugin_path, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(logger, "")

    def initialize_engines(self):
        for model_name, model_path in self.model_paths.items():
            engine = self.load_engine(model_path)
            if engine is None:
                raise RuntimeError(f"Failed to load engine for {model_name}")
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(f"Failed to create execution context for {model_name}")
            bindings = [Binding(engine, i) for i in range(engine.num_io_tensors)]
            self.engines[model_name] = engine
            self.contexts[model_name] = context
            self.bindings[model_name] = bindings
            self.binding_addresses[model_name] = [b.device_buffer.ptr for b in bindings]
            self.inputs[model_name] = [b for b in bindings if b.is_input]
            self.outputs[model_name] = [b for b in bindings if not b.is_input]
            self.prepare_buffers(model_name)

    @staticmethod
    def load_engine(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def prepare_buffers(self, model_name):
        for binding in self.inputs[model_name] + self.outputs[model_name]:
            _ = binding.device_buffer  # Force buffer allocation

    @staticmethod
    def check_input_validity(input_idx, input_array, input_binding):
        if input_array.shape != input_binding.shape:
            if not (input_binding.shape == (1,) and input_array.shape == ()):
                raise ValueError(
                    f"Wrong shape for input {input_idx}. Expected {input_binding.shape}, got {input_array.shape}.")
        if input_array.dtype != input_binding.dtype:
            if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
                input_array = input_array.astype(np.int32)
                if not np.array_equal(input_array, input_array.astype(np.int64)):
                    raise TypeError(
                        f"Wrong dtype for input {input_idx}. Expected {input_binding.dtype}, got {input_array.dtype}. Cannot safely cast.")
            else:
                raise TypeError(
                    f"Wrong dtype for input {input_idx}. Expected {input_binding.dtype}, got {input_array.dtype}.")
        return input_array

    def run_sequential_tasks(self, model_name, inputs):
        if model_name not in self.engines:
            raise ValueError(f"Model name {model_name} not found in engines.")
        engine = self.engines[model_name]
        context = self.contexts[model_name]
        binding_addresses = self.binding_addresses[model_name]
        inputs_bindings = self.inputs[model_name]
        outputs_bindings = self.outputs[model_name]

        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in inputs_bindings]
        if len(inputs) != len(inputs_bindings):
            raise ValueError(f"Number of input arrays does not match number of input bindings for model {model_name}.")

        self.cfx.push()  # Push CUDA context

        try:
            for i, (input_array, input_binding) in enumerate(zip(inputs, inputs_bindings)):
                input_array = self.check_input_validity(i, input_array, input_binding)
                input_array = np.ascontiguousarray(input_array)  # Ensure the input array is contiguous
                cuda.memcpy_htod(input_binding.device_buffer.ptr, input_array)

            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                if i < len(inputs) and engine.is_shape_inference_io(tensor_name):
                    context.set_tensor_address(tensor_name, inputs[i].ctypes.data)
                else:
                    context.set_tensor_address(tensor_name, binding_addresses[i])

            context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()

            outputs = []
            for output in outputs_bindings:
                host_output = np.empty(output.shape, dtype=output.dtype)
                cuda.memcpy_dtoh(host_output, output.device_buffer.ptr)
                outputs.append(host_output)

        except Exception as e:
            print(f"Error during inference for model {model_name}: {e}")
            outputs = None

        self.cfx.pop()  # Pop CUDA context

        return outputs

    def inference_tensorrt(self, task, inputs):
        if not isinstance(inputs, list):
            raise TypeError("Inputs should be a list of numpy arrays or tensors.")

        if task not in self.inputs:
            raise ValueError(f"Task {task} not found in the model inputs.")

        # Ensure all inputs are on the same memory type
        if isinstance(inputs[0], pycuda.gpuarray.GPUArray):
            # Ensure all inputs are on GPU
            inputs = [cuda.to_gpu(input_array) if not isinstance(input_array, pycuda.gpuarray.GPUArray) else input_array
                      for input_array in inputs]
        else:
            # Ensure all inputs are on CPU
            inputs = [input_array.get() if isinstance(input_array, pycuda.gpuarray.GPUArray) else input_array
                      for input_array in inputs]

        inputs = [self.check_input_validity(i, np.array(input_array), self.inputs[task][i])
                  for i, input_array in enumerate(inputs)]

        result = self.run_sequential_tasks(task, inputs)
        return result

    def __del__(self):
        del self.engines
        del self.contexts
        del self.bindings
        del self.binding_addresses
        del self.inputs
        del self.outputs
        del self.stream
        try:
            if self.cfx is not None:
                self.cfx.pop()
                del self.cfx
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Example usage
# engine = TensorRTEngine(half=True, F_rt_half="path/to/F_rt_half", M_rt_half="path/to/M_rt_half",
#                         GW_rt_half="path/to/GW_rt_half", S_rt_half="path/to/S_rt_half",
#                         SE_rt_half="path/to/SE_rt_half", SL_rt_half="path/to/SL_rt_half",
#                         grid_sample_3d="path/to/grid_sample_3d.so")
