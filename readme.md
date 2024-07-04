# ComfyUI nodes to use LivePortrait


https://github.com/kijai/ComfyUI-LivePortrait/assets/40791699/e55e10f6-af61-4d73-b162-af29eb847516


I have converted all the pickle files to safetensors: https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main

They go here (and are automatically downloaded if the folder is not present) `ComfyUI/models/liveportrait`


Insightface is also required.
If you have a working compile encironment, installing it can be as easy as:

`pip install insightface`

or for the portable version, in the ComfyUI_windows_portable -folder:

`python_embeded/python.exe -m pip install insightface`

For insightface model, extract this to `ComfyUI/models/insightface/buffalo_l`:

https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

*Please note that insightface license is non-commercial in nature.*
