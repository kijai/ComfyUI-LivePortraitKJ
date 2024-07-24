# ComfyUI nodes to use [LivePortrait](https://github.com/KwaiVGI/LivePortrait)

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
