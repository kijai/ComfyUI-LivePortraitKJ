# coding: utf-8

"""
utility functions and classes to handle feature extraction and model loading
"""
import torch
import os
from glob import glob
import os.path as osp
import imageio
import numpy as np
import cv2
from rich.progress import track

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False


def is_template(file_path):
    if file_path.endswith(".pkl"):
        return True
    return False


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2cuda(dct: dict, device: str):
    for key in dct:
        dct[key] = torch.tensor(dct[key]).to(device)
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def load_image_rgb(image_path: str):
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_driving_info(driving_info):
    driving_video_ori = []

    def load_images_from_directory(directory):
        image_paths = sorted(glob(osp.join(directory, '*.png')) + glob(osp.join(directory, '*.jpg')))
        return [load_image_rgb(im_path) for im_path in image_paths]

    def load_images_from_video(file_path):
        reader = imageio.get_reader(file_path)
        return [image for idx, image in enumerate(reader)]

    if osp.isdir(driving_info):
        driving_video_ori = load_images_from_directory(driving_info)
    elif osp.isfile(driving_info):
        driving_video_ori = load_images_from_video(driving_info)

    return driving_video_ori


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj


def resize_to_limit(img: np.ndarray, max_dim=1920, n=2):
    """
    ajust the size of the image so that the maximum dimension does not exceed max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    n = max(n, 1)
    new_h = img.shape[0] - (img.shape[0] % n)
    new_w = img.shape[1] - (img.shape[1] % n)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def load_img_online(obj, mode="bgr", **kwargs):
    max_dim = kwargs.get("max_dim", 1920)
    n = kwargs.get("n", 2)
    if isinstance(obj, str):
        if mode.lower() == "gray":
            img = cv2.imread(obj, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(obj, cv2.IMREAD_COLOR)
    else:
        img = obj

    # Resize image to satisfy constraints
    img = resize_to_limit(img, max_dim=max_dim, n=n)

    if mode.lower() == "bgr":
        return contiguous(img)
    elif mode.lower() == "rgb":
        return contiguous(img[..., ::-1])
    else:
        raise Exception(f"Unknown mode {mode}")


def images2video(images, wfp, **kwargs):
    fps = kwargs.get('fps', 30)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    image_mode = kwargs.get('image_mode', 'rgb')
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

    writer = imageio.get_writer(
        wfp, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat,
        macro_block_size=macro_block_size
    )

    n = len(images)
    for i in track(range(n), description='writing', transient=True):
        if image_mode.lower() == 'bgr':
            writer.append_data(images[i][..., ::-1])
        else:
            writer.append_data(images[i])

    writer.close()

    # print(f':smiley: Dump to {wfp}\n', style="bold green")
    print(f'Dump to {wfp}\n')
    return wfp
