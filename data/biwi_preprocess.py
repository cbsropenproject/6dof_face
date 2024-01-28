import os
import pdb
import cv2
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation
from Sim3DR import RenderPipeline       # ref: https://github.com/cleardusk/3DDFA_V2/tree/master/Sim3DR

np.set_printoptions(suppress=True)
os.makedirs('debug_dir', exist_ok=True)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

cfg = {
    "intensity_ambient": 0.3,
    "color_ambient": (1, 1, 1),
    "intensity_directional": 0.6,
    "color_directional": (1, 1, 1),
    "intensity_specular": 0.1,
    "specular_exp": 5,
    "light_pos": (0, 0, 5),
    "view_pos": (0, 0, 5),
}
render_app = RenderPipeline(**cfg)


def load_obj(path_to_file):
    """ Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices).astype(np.float32)
    faces = np.asarray(faces).astype(np.int32)
    return vertices, faces




if __name__ == '__main__':

    data_root = Path('/PATH/TO/BIWI/hpdb')

    subject_id = np.random.randint(1, 25)
    print('subject_id =', subject_id)

    obj_path = data_root / ('%02d.obj' % subject_id)
    verts_origin, tris = load_obj(obj_path)

    tris_copy = tris.copy()
    tris[:, [0, 1]] = tris_copy[:, [1, 0]]
    tris = np.ascontiguousarray(tris.astype(np.int32))


    T = np.array([
        [1.0, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]) 
    verts_origin = verts_origin @ T


    img_dir = data_root / ('%02d' % subject_id)

    cal_path = img_dir / 'rgb.cal'
    with open(cal_path, 'r') as f:
        line_list = f.readlines()


    K_img = []
    for i in range(3):
        oneline = line_list[i].strip()
        oneline = oneline.split(' ')
        oneline = [float(x) for x in oneline]
        K_img.append(oneline)

    K_img = np.array(K_img, dtype=np.float32)

 
    R_calibration = []
    for i in range(6, 9):
        oneline = line_list[i].strip()
        oneline = oneline.split(' ')
        oneline = [float(x) for x in oneline]
        R_calibration.append(oneline)
    R_calibration = np.array(R_calibration, dtype=np.float32)


    oneline = line_list[10].strip()
    oneline = oneline.split(' ')
    oneline = [float(x) for x in oneline]
    t_calibration = np.array(oneline, dtype=np.float32)


    R_t_calibration = np.identity(4).astype(np.float32)
    R_t_calibration[:3, :3] = R_calibration.T
    R_t_calibration[3, :3] = t_calibration


    img_list = sorted(list(img_dir.glob('*.png')))
    index = np.random.randint(len(img_list))
    print('index =', index)


    img_path = str(img_list[index])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape


    fig, axes = plt.subplots(4, 2, figsize=(26, 32))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('original image')


    txt_path = img_path.replace('_rgb.png', '_pose.txt')    
    with open(txt_path, 'r') as f:
        line_list = f.readlines()


    R_ = []
    for i in range(3):
        oneline = line_list[i].strip()
        oneline = oneline.split(' ')
        oneline = [float(x) for x in oneline]
        R_.append(oneline)
    R_ = np.array(R_, dtype=np.float32)

    oneline = line_list[4].strip()
    oneline = oneline.split(' ')
    oneline = [float(x) for x in oneline]
    t_ = np.array(oneline, dtype=np.float32)
    

    R_t_raw = np.identity(4).astype(np.float32)
    R_t_raw[:3, :3] = R_.T
    R_t_raw[3, :3] = t_
    R_t_cv = R_t_raw @ R_t_calibration


    # mm -> m
    R_t_cv[3, :3] /= 1000
    verts_origin /= 1000
    print('\nR_t_cv =\n', R_t_cv)   # row-major
    assert R_t_cv[3, 2] > 0


    R_t_gl = R_t_cv.copy()
    euler_angle = Rotation.from_matrix(R_t_gl[:3, :3].T).as_euler('yxz', degrees=True)
    euler_angle[0] *= -1
    euler_angle[2] *= -1
    R_t_gl[:3, :3] = Rotation.from_euler('yxz', euler_angle, degrees=True).as_matrix().T

    T_vec_new = R_t_gl[3, :3].copy()
    T_vec_new[1] *= -1
    T_vec_new[2] *= -1
    R_t_gl[3, :3] = T_vec_new
    print('\nR_t_gl =\n', R_t_gl)
    assert R_t_gl[3, 2] < 0


    ones = np.ones([verts_origin.shape[0], 1], dtype=np.float32)
    verts_homo = np.concatenate([verts_origin, ones], axis=1)


    Cx, Cy = K_img[0, 2], K_img[1, 2]
    M2 = np.array([
        [K_img[0, 0], 0, 0, 0],
        [0, K_img[1, 1], 0, 0],
        [-Cx, -(img_h - Cy), -0.99999976, -1],
        [0, 0, -0.001, 0]
    ])
    verts = verts_homo @ R_t_gl @ M2
    w_ = verts[:, [3]]
    verts = verts / w_


    temp = img.copy()
    for p in verts:
        cv2.circle(temp, ( int(p[0]), img_h - int(p[1]) ), radius=1, color=(0, 255, 0), thickness=-1)
    axes[0, 1].imshow(temp)
    axes[0, 1].set_title('plot points')



    kpt_ind = [938, 954, 935, 946, 6374, 6370, 6769, 6759, 6749] + \
        [2634, 2647, 2657, 2234, 2243, 3364, 3377, 3357, 3375] + \
        [4805, 5007, 5001, 5000, 179] + [3800, 864, 861, 870, 3486] + \
        [3728, 3718, 2836, 2816] + [4773, 6612, 2772, 2489, 3518] + \
        [4449, 4627, 4721, 4410, 4425, 4472] + [562, 572, 478, 456, 325, 242] + \
        [6798, 6638, 6631, 2508, 2512, 2517, 2685, 2736, 2885, 2893, 5278, 6847] + \
        [6817, 6565, 2438, 2866, 2719, 2856, 5296]

    temp = img.copy()
    for p in verts[kpt_ind]:
        cv2.circle(temp, ( int(p[0]), img_h - int(p[1]) ), radius=1, color=(0, 255, 0), thickness=-1)
    axes[1, 0].imshow(temp)
    axes[1, 0].set_title('plot 68 points')



    points2d = verts[:, :2].copy()
    points2d[:, 1] = img_h - points2d[:, 1]
    verts_temp = np.concatenate([points2d, w_], axis=1)

    f = M2[0, 0]
    tz = R_t_gl[3, 2]
    scale = f / tz
    verts_temp[:, 2] *= scale
    verts_temp = _to_ctype(verts_temp.astype(np.float32))
    overlap = img.copy()
    overlap = render_app(verts_temp, tris, overlap)
    alpha = 0.75

    img_render = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    axes[1, 1].imshow(img_render)
    axes[1, 1].set_title('render with GL_format')



    # Translate the head mesh of BIWI to align it with the standard head of ARKit
    center = np.array([0, 0.01, -0.021], dtype=np.float32)
    center_mat = np.identity(4, dtype=np.float32)
    center_mat[3, :3] = center

    verts_origin -= center
    R_t_gl = center_mat @ R_t_gl

    ones = np.ones([verts_origin.shape[0], 1], dtype=np.float32)
    verts_homo = np.concatenate([verts_origin, ones], axis=1)


    f = K_img[0, 0]
    M_proj = np.array([
        [f/(img_w/2), 0, 0, 0],
        [0, f/(img_h/2), 0, 0],
        [M2[2, 0]/(img_w/2)+1, M2[2, 1]/(img_h/2)+1, -0.99999976, -1],
        [0, 0, -0.001, 0]
    ])
    M1 = np.array([
        [img_w/2,       0, 0, 0],
        [      0, img_h/2, 0, 0],
        [      0,       0, 1, 0],
        [img_w/2, img_h/2, 0, 1]
    ])

    verts = verts_homo @ R_t_gl @ M_proj @ M1
    w_ = verts[:, [3]]
    verts = verts / w_

    temp = img.copy()
    for p in verts:
        cv2.circle(temp, ( int(p[0]), int(img_h - p[1]) ), radius=1, color=(0, 255, 0), thickness=-1)
    axes[2, 0].imshow(temp)
    axes[2, 0].set_title('plot points2d')




    points2d = verts[:, :2].copy()
    points2d[:, 1] = img_h - points2d[:, 1]
    verts_temp = np.concatenate([points2d, w_], axis=1)

    f = M2[0, 0]
    tz = R_t_gl[3, 2]
    scale = f / tz
    verts_temp[:, 2] *= scale
    verts_temp = _to_ctype(verts_temp.astype(np.float32))


    overlap = img.copy()
    overlap = render_app(verts_temp, tris, overlap)
    alpha = 0.75

    img_render = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    axes[2, 1].imshow(img_render)
    axes[2, 1].set_title('align to ARKit')



    
    # The standard FOV for the ARKitFace dataset is 64.84315322103066
    tan_ = 608 / 957.2577
    FOV = np.rad2deg(np.arctan(tan_)) * 2
    print('std FOV =', FOV)

    focal_length = K_img[0, 0]
    pixel_expand = int(round(tan_ * focal_length))

    A = M_proj[0, 0]
    B = M_proj[1, 1]
    C = M_proj[2, 0]
    D = M_proj[2, 1]
    E = M_proj[2, 2]
    F = M_proj[3, 2]

    n = 0.01
    r = n * (C + 1) / A
    l = n * (C - 1) / A
    t = n * (D + 1) / B
    b = n * (D - 1) / B
    print('\n')
    print('l=%f, r=%f' % (l, r))
    print('b=%f, t=%f' % (b, t))

    factor = focal_length / n
    
    r_pixel = int(round(r * factor))
    t_pixel = int(round(t * factor))


    # pad or cut
    delta_r_pixel = pixel_expand - r_pixel
    delta_l_pixel = pixel_expand * 2 - delta_r_pixel - img_w

    delta_t_pixel = pixel_expand - t_pixel
    delta_b_pixel = pixel_expand * 2 - delta_t_pixel - img_h


    temp = img.copy()

    if delta_r_pixel < 0:
        temp = temp[:, -delta_l_pixel:delta_r_pixel]
    else:
        temp = np.pad(temp, ((0, 0), (delta_l_pixel, delta_r_pixel), (0, 0)))
    
    
    if delta_t_pixel < 0:
        temp = temp[-delta_t_pixel:delta_b_pixel]
    else:
        temp = np.pad(temp, ((delta_t_pixel, delta_b_pixel), (0, 0), (0, 0)))
        

    full_img = temp
    img_h, img_w, _ = full_img.shape
    assert img_h == pixel_expand * 2 and img_w == pixel_expand * 2    



    M_proj[0, 0] = 1 / tan_
    M_proj[1, 1] = 1 / tan_
    M_proj[2, 0] = 0
    M_proj[2, 1] = 0

    M3 = np.array([
        [img_w/2,       0, 0, 0],
        [      0, img_h/2, 0, 0],
        [      0,       0, 1, 0],
        [img_w/2, img_h/2, 0, 1]
    ])
    verts = verts_homo @ R_t_gl @ M_proj @ M3
    w_ = verts[:, [3]]
    verts = verts / w_

    temp = full_img.copy()
    for p in verts:
        cv2.circle(temp, ( int(p[0]), int(img_h - p[1]) ), radius=1, color=(0, 255, 0), thickness=-1)
    axes[3, 0].imshow(temp)
    axes[3, 0].set_title(f'size={img_w}')


    final_size = 800

    full_img = cv2.resize(full_img, (final_size, final_size))
    img_h, img_w, _ = full_img.shape

    M4 = np.array([
        [img_w/2,       0, 0, 0],
        [      0, img_h/2, 0, 0],
        [      0,       0, 1, 0],
        [img_w/2, img_h/2, 0, 1]
    ])

    verts = verts_homo @ R_t_gl @ M_proj @ M4
    w_ = verts[:, [3]]
    verts = verts / w_

    temp = full_img.copy()
    for p in verts:
        cv2.circle(temp, ( int(p[0]), int(img_h - p[1]) ), radius=1, color=(0, 255, 255), thickness=-1)
    axes[3, 1].imshow(temp)
    axes[3, 1].set_title(f'size={img_w}')

    plt.savefig('debug_dir/display.jpg')