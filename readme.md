# ComfyUI nodes to use LivePortrait

I have converted all the pickle files to safetensors, and they

are automatically downloaded from here to `ComfyUI/models/liveportrait`:

https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main

Insightface is required.
If you have a working compile encironment, installing it can be as easy as:

`pip install insightface`

or for the portable version, in the ComfyUI_windows_portable -folder:

`python_embeded/python.exe -m pip install insightface`

For insightface model, extract this to `ComfyUI/models/insightface/buffalo_l`:

https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

*Please note that insightface license is non-commercial in nature.*