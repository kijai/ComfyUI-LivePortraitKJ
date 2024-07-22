# coding: utf-8

"""
cropping function and the related preprocess functions for cropping
"""

import cv2#; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False) # NOTE: enforce single thread
import numpy as np
from math import sin, cos, acos, degrees
DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR
import comfy.model_management as mm

def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    """ conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode, borderValue=(0, 0, 0))
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)

import torch
import kornia.geometry.transform as KGT

def _transform_img_kornia(img, M, dsize, flags='bilinear', borderMode='zeros'):
    """Conduct similarity or affine transformation to the image using Kornia.
    
    img: Input image as a PyTorch tensor of shape (C, H, W).
    M: 2x3 transformation matrix as a PyTorch tensor.
    dsize: Target shape (width, height).
    """
    device = mm.get_torch_device()
    if mm.is_device_mps(device): 
        device = torch.device('cpu') #this function returns NaNs on MPS, defaulting to CPU
   
    # Convert dsize to tensor shape (H, W)
    _dsize = torch.tensor([dsize[1], dsize[0]])  # Kornia expects (H, W)

    # Convert M from numpy.ndarray to PyTorch tensor
    M = torch.from_numpy(M).float().to(device)
    if M.shape == (3, 3):
        M = M[:2, :].unsqueeze(0)  # Adjust M to the expected shape Bx2x3
    elif M.shape == (2, 3):
        M = M.unsqueeze(0)  # Add batch dimension if not present

    # Reshape M for Kornia (1, 2, 3) and upscale to 3D affine matrix if not already
    if M.shape == (2, 3):
        M = M.unsqueeze(0)  # Add batch dimension

    # Convert image to floating point tensor if not already
    if img.dtype != torch.float32:
        img = img.float()
    img = img.to(device)

    # Reshape img for Kornia (B, C, H, W)
    img = img.permute(0, 3, 1, 2)

    # Apply the affine transformation
    img_warped = KGT.warp_affine(img, M, _dsize, mode=flags, padding_mode=borderMode)

    return img_warped

def _transform_pts(pts, M):
    """ conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]


def parse_pt2_from_pt101(pt101, use_lip=True):
    """
    parsing the 2 points according to the 101 points, which cancels the roll
    """
    # the former version use the eye center, but it is not robust, now use interpolation
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt106(pt106, use_lip=True):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2


def parse_pt2_from_pt203(pt203, use_lip=True):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)  # right eye center
    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
    return pt2

def parse_pt2_from_pt9(pt9, use_lip=True):
    '''
    animal_face = {"keypoints": ['right eye right', 'right eye left', 'left eye right', 'left eye left', 'nose tip', 'lip right', 'lip left', 'upper lip', 'lower lip'], "skeleton": []}


    '''
    if use_lip:
        pt9 = np.stack([
            (pt9[2]+pt9[3])/2, # left eye
            (pt9[0]+pt9[1])/2, # right eye
            pt9[4],
            # (pt9[5]+pt9[6]+pt9[7]+pt9[8])/4 # lip
            (pt9[5] + pt9[6] ) / 2 # lip
        ], axis=0)
        pt2 = np.stack([
            (pt9[0] + pt9[1]) / 2, # eye
            pt9[3] # lip
        ], axis=0)
    else:
        pt2 = np.stack([
            (pt9[2] + pt9[3]) / 2,
            (pt9[0] + pt9[1]) / 2,
        ], axis=0)

    return pt2

def parse_pt2_from_pt68(pt68, use_lip=True):
    '''
face = {"keypoints": ['right cheekbone 1', 'right cheekbone 2', 'right cheek 1', 'right cheek 2', 'right cheek 3', 'right cheek 4', 'right cheek 5', 'right chin', 'chin center', 
'left chin', 'left cheek 5', 'left cheek 4', 'left cheek 3', 'left cheek 2', 'left cheek 1', 'left cheekbone 2', 'left cheekbone 1', 'right eyebrow 1', 'right eyebrow 2', 'right eyebrow 3', 
'right eyebrow 4', 'right eyebrow 5', 'left eyebrow 1', 'left eyebrow 2', 'left eyebrow 3', 'left eyebrow 4', 'left eyebrow 5', 'nasal bridge 1', 'nasal bridge 2', 'nasal bridge 3', 'nasal bridge 4', 
'right nasal wing 1', 'right nasal wing 2', 'nasal wing center', 'left nasal wing 1', 'left nasal wing 2', 'right eye eye corner 1', 'right eye upper eyelid 1', 'right eye upper eyelid 2', 
'right eye eye corner 2', 'right eye lower eyelid 2', 'right eye lower eyelid 1', 'left eye eye corner 1', 'left eye upper eyelid 1', 'left eye upper eyelid 2', 'left eye eye corner 2', 'left eye lower eyelid 2', 
'left eye lower eyelid 1', 'right mouth corner', 'upper lip outer edge 1', 'upper lip outer edge 2', 'upper lip outer edge 3', 'upper lip outer edge 4', 'upper lip outer edge 5', 'left mouth corner', 
'lower lip outer edge 5', 'lower lip outer edge 4', 'lower lip outer edge 3', 'lower lip outer edge 2', 'lower lip outer edge 1', 'upper lip inter edge 1', 'upper lip inter edge 2', 'upper lip inter edge 3', 
'upper lip inter edge 4', 'upper lip inter edge 5', 'lower lip inter edge 3', 'lower lip inter edge 2', 'lower lip inter edge 1'], "skeleton": []}


    '''
    if use_lip:
        pt68 = np.stack([
            (pt68[42] + pt68[43] + pt68[44] + pt68[45] + pt68[46]+ pt68[47])/6, # left eye
            (pt68[36] + pt68[37] + pt68[38] + pt68[39] + pt68[40] + pt68[41]) / 6,  # right eye
            (pt68[48] + pt68[54])/2

        ], axis=0)
        pt2 = np.stack([
            (pt68[0] + pt68[1]) / 2,
            pt68[2]
        ], axis=0)
    else:
        pt2 = np.stack([
            (pt68[42] + pt68[43] + pt68[44] + pt68[45] + pt68[46] + pt68[47]) / 6,  # left eye
            (pt68[36] + pt68[37] + pt68[38] + pt68[39] + pt68[40] + pt68[41]) / 6,  # right eye
        ], axis=0)

    return pt2


def parse_pt2_from_pt5(pt5, use_lip=True):
    """
    parsing the 2 points according to the 5 points, which cancels the roll
    """
    if use_lip:
        pt2 = np.stack([
            (pt5[0] + pt5[1]) / 2,
            (pt5[3] + pt5[4]) / 2
        ], axis=0)
    else:
        pt2 = np.stack([
            pt5[0],
            pt5[1]
        ], axis=0)
    return pt2


def parse_pt2_from_pt_x(pts, use_lip=True):
    if pts.shape[0] == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    elif pts.shape[0] == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)
    elif pts.shape[0] == 68:
        pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
    elif pts.shape[0] == 5:
        pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
    elif pts.shape[0] == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip)
    elif pts.shape[0] > 101:
        # take the first 101 points
        pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
    elif pts.shape[0] == 9:
        pt2 = parse_pt2_from_pt9(pts, use_lip=use_lip)
    else:
        raise Exception(f'Unknow shape: {pts.shape}')

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2


def parse_rect_from_landmark(
    pts,
    scale=1.5,
    need_square=True,
    vx_ratio=0,
    vy_ratio=0,
    use_deg_flag=False,
    **kwargs
):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get('use_lip', True))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=DTYPE)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    # print(uy)
    # print(ux)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = center0 + ux * center1[0] + uy * center1[1]  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = center + ux * (vx_ratio * size) + uy * \
        (vy_ratio * size)  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle


def parse_bbox_from_landmark(pts, **kwargs):
    center, size, angle = parse_rect_from_landmark(pts, **kwargs)
    cx, cy = center
    w, h = size

    # calculate the vertex positions before rotation
    bbox = np.array([
        [cx-w/2, cy-h/2],  # left, top
        [cx+w/2, cy-h/2],
        [cx+w/2, cy+h/2],  # right, bottom
        [cx-w/2, cy+h/2]
    ], dtype=DTYPE)

    # construct rotation matrix
    bbox_rot = bbox.copy()
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ], dtype=DTYPE)

    # calculate the relative position of each vertex from the rotation center, then rotate these positions, and finally add the coordinates of the rotation center
    bbox_rot = (bbox_rot - center) @ R.T + center

    return {
        'center': center,  # 2x1
        'size': size,  # scalar
        'angle': angle,  # rad, counterclockwise
        'bbox': bbox,  # 4x2
        'bbox_rot': bbox_rot,  # 4x2
    }


def crop_image_by_bbox(img, bbox, lmk=None, dsize=512, angle=None, flag_rot=False, **kwargs):
    left, top, right, bot = bbox
    if int(right - left) != int(bot - top):
        print(f'right-left {right-left} != bot-top {bot-top}')
    size = right - left

    src_center = np.array([(left + right) / 2, (top + bot) / 2], dtype=DTYPE)
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)

    s = dsize / size  # scale
    if flag_rot and angle is not None:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = src_center[0], src_center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_o2c = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=DTYPE
        )
    else:
        M_o2c = np.array(
            [[s, 0, tgt_center[0] - s * src_center[0]],
             [0, s, tgt_center[1] - s * src_center[1]]],
            dtype=DTYPE
        )

    if flag_rot and angle is None:
        print('angle is None, but flag_rotate is True', style="bold yellow")

    img_crop = _transform_img(img, M_o2c, dsize=dsize, borderMode=kwargs.get('borderMode', None))

    lmk_crop = _transform_pts(lmk, M_o2c) if lmk is not None else None

    M_o2c = np.vstack([M_o2c, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    # cv2.imwrite('crop.jpg', img_crop)

    return {
        'img_crop': img_crop,
        'lmk_crop': lmk_crop,
        'M_o2c': M_o2c,
        'M_c2o': M_c2o,
    }


def _estimate_similar_transform_from_pts(
    pts,
    dsize,
    scale=1.5,
    vx_ratio=0,
    vy_ratio=-0.1,
    flag_do_rot=True,
    **kwargs
):
    """ calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
        use_lip=kwargs.get('use_lip', True)
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=DTYPE
        )
    else:
        M_INV = np.array(
            [[s, 0, tgt_center[0] - s * center[0]],
             [0, s, tgt_center[1] - s * center[1]]],
            dtype=DTYPE
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]


def crop_image(img, pts: np.ndarray, **kwargs):
    dsize = kwargs.get('dsize', 224)
    scale = kwargs.get('scale', 1.5)  # 1.5 | 1.6
    vy_ratio = kwargs.get('vy_ratio', -0.1)  # -0.0625 | -0.1
    vx_ratio = kwargs.get('vx_ratio', 0)

    M_INV, _ = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        vx_ratio=vx_ratio,
        flag_do_rot=kwargs.get('rotate', True),
    )

    if img is None:
        M_INV_H = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
        M = np.linalg.inv(M_INV_H)
        ret_dct = {
            'M': M[:2, ...],  # from the original image to the cropped image
            'M_o2c': M[:2, ...],  # from the cropped image to the original image
            'img_crop': None,
            'pt_crop': None,
        }
        return ret_dct

    img_crop = _transform_img(img, M_INV, dsize)  # origin to crop
    pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    ret_dct = {
        'M_o2c': M_o2c,  # from the original image to the cropped image 3x3
        'M_c2o': M_c2o,  # from the cropped image to the original image 3x3
        'img_crop': img_crop,  # the cropped image
        'pt_crop': pt_crop,  # the landmarks of the cropped image
    }

    return ret_dct

def average_bbox_lst(bbox_lst):
    if len(bbox_lst) == 0:
        return None
    bbox_arr = np.array(bbox_lst)
    return np.mean(bbox_arr, axis=0).tolist()
