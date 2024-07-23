import onnxruntime as ort
import torch
import numpy as np
from typing import Dict


class ONNXEngine:
    def __init__(self):
        pass

    @staticmethod
    def get_providers() -> list:
        """Returns the list of providers based on the current device."""
        if ort.get_device() == 'GPU':
            return ['CUDAExecutionProvider']
        elif ort.get_device() == 'CPU':
            return ['CPUExecutionProvider', 'CoreMLExecutionProvider']
        else:
            return []

    def initialize_sessions(self, cfg) -> Dict[str, ort.InferenceSession]:
        """
        Initialize ONNX InferenceSession instances for each model checkpoint.

        Args:
        - cfg (dict): Configuration dictionary containing checkpoint paths.

        Returns:
        - Dict[str, ort.InferenceSession]: Dictionary mapping session names to InferenceSession objects.
        """
        #providers = self.get_providers()
        providers = ['CUDAExecutionProvider']

        # Initialize each session manually
        gw_session = ort.InferenceSession("live_portrait_weights\\live_portrait\\generator_fix_grid.onnx", providers=providers)
        # m_session = ort.InferenceSession(cfg.get("checkpoint_M"), providers=providers)
        # f_session = ort.InferenceSession(cfg.get("checkpoint_F"), providers=providers)
        # s_session = ort.InferenceSession(cfg.get("checkpoint_S"), providers=providers)
        # se_session = ort.InferenceSession(cfg.get("checkpoint_SE"), providers=providers)
        # sl_session = ort.InferenceSession(cfg.get("checkpoint_SL"), providers=providers)

        # Return the sessions in a dictionary
        return {
            "gw_session": gw_session,
            # "m_session": m_session,
            # "f_session": f_session,
            # "s_session": s_session,
            # "se_session": se_session,
            # "sl_session": sl_session
        }

