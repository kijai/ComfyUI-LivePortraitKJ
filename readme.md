# ComfyUI nodes to use [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
## Update 2

Added another alternative face detector: https://github.com/1adrianb/face-alignment

![image](https://github.com/user-attachments/assets/1a77752a-9688-4b6f-9363-736367ad711a)

As this can use blazeface back camera model (or SFD), it's far better for smaller faces than MediaPipe, that only can use the blazeface short -model.
The warmup on the first run when using this can take a long time, but subsequent runs are quick.

Example detection using the blazeface_back_camera:

https://github.com/user-attachments/assets/40b1fdb4-0b1f-4ea8-8322-aa9151055db0

## Update

Rework of almost the whole thing that's been in develop is now merged into main, this means old workflows will not work, but everything should be faster and there's lots of new features.
For legacy purposes the old main branch is moved to the legacy -branch

Changes
- Added MediaPipe as alternative to Insightface, everything should now be covered under MIT and Apache-2.0 licenses when using it.
- Proper Vid2vid including smoothing algorhitm (thanks @melMass)
- Improved speed and efficiency, allows for near realtime view even in Comfy (~80-100ms delay)
- Restructured nodes for more options
- Auto skipping frames with no face detected
- Numerous other things I have forgotten about at this point, it's been a lot
- Better Mac support on MPS (thanks @Grant-CP

update to this update:
- converted the landmark runner onnx model to torch model, not something I have done before and I didn't manage to do anything but make it .pth file, so you'll just have to trust me on it.
  This allows running all this without even having onnxruntime, it runs on GPU and is about just as fast. It's available on the MediaPipe cropper node as option:
When selected it's automatically downloaded from here: https://huggingface.co/Kijai/LivePortrait_safetensors/blob/main/landmark_model.pth

![image](https://github.com/user-attachments/assets/c547f55a-9ef7-4bc7-85df-cdbab69a3ca8)


# Examples:

Realtime with webcam feed:

https://github.com/user-attachments/assets/31f77c10-b757-44ae-bb26-39e45ec0b2d9

Image2vid:

https://github.com/user-attachments/assets/cfec0419-d1eb-4e67-8913-890eeb155eef

Vid2Vid: 

https://github.com/user-attachments/assets/28438fcb-fbb0-4e4e-baf4-00fe06c455de


I have converted all the pickle files to safetensors: https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main

They go here (and are automatically downloaded if the folder is not present) `ComfyUI/models/liveportrait`

# Face detectors

You can either use the original default Insightface, or Google's MediaPipe. 

Biggest difference is the license: Insightface is strictly for NON-COMMERCIAL use.
MediaPipe is a bit worse at detection, and can't run on GPU in Windows, though it's much faster on CPU compared to Insightface

Insightface is not automatically installed, if you wish to use it follow these instructions:
If you have a working compile environment, installing it can be as easy as:

`pip install insightface`

or for the portable version, in the ComfyUI_windows_portable -folder:

`python_embeded/python.exe -m pip install insightface`

If this fails (and it's likely), you can check the Troubleshooting part of the reactor node for alternative:

https://github.com/Gourieff/comfyui-reactor-node

For insightface model, extract this to `ComfyUI/models/insightface/buffalo_l`:

https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

*Please note that insightface license is non-commercial in nature.*
