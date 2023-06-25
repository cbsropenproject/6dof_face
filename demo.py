import matplotlib
matplotlib.use('Agg')
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models import create_model
from util.renderer import Renderer
import torch.nn.functional as F

renderer = Renderer(img_size=800)

def resize_img(img, bbox, target_size=None):
    img_h, img_w, _ = img.shape
    if img_w > img_h:
        max_size = img_w
        diff = img_w - img_h
        if diff % 2 == 1:
            pad1 = (diff + 1) // 2
            pad2 = (diff - 1) // 2
        else:
            pad1 = diff // 2
            pad2 = pad1
        img_pad = np.pad(img, ((pad1, pad2), (0, 0), (0, 0)),mode='constant')
        bbox[:,1] +=pad1
        bbox[:,3] +=pad1
    elif img_w == img_h:
        img_pad = img
    elif img_h > img_w:
        max_size = img_h
        diff = img_h - img_w
        if diff % 2 == 1:
            pad1 = (diff + 1) // 2
            pad2 = (diff - 1) // 2
        else:
            pad1 = diff // 2
            pad2 = pad1
        img_pad = np.pad(img, ((0, 0), (pad1, pad2), (0, 0)),mode='constant')
        bbox[:,0] +=pad1
        bbox[:,2] +=pad1

    img_resize = cv2.resize(img_pad, (target_size,)*2)
    scale = target_size / img_pad.shape[0]
    # lms *= scale
    bbox*=scale

    return img_resize, bbox


npz_path = './npy/kpt_ind.npy'
kpt_ind = np.load(npz_path)

npy_path = './npy/uv_coords_std_202109.npy'
uv_coords_std = np.load(npy_path)[:1220]
uv_coords = uv_coords_std * 2 - 1
grid = torch.tensor(uv_coords).unsqueeze(0).unsqueeze(1).to('cuda')


if __name__ == '__main__':
    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')

    opt = TestOptions().parse()  # get test options

    model = create_model(opt)
    model.setup(opt)
    model.netR.eval()

    img_path = 'example/test.jpg'
    save_img_r = 'example/test_r.jpg'
    save_img_d = 'example/test_d.jpg'

    img_raw = cv2.imread(str(img_path))
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    bboxes = [[]]
    bboxes = np.array(bboxes)
    num_faces = len(bboxes)
    print('\nnum_faces =', num_faces)

    img, bboxes = resize_img(img_raw, bboxes, target_size=800)

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    dst_pts = np.float32([
        [0, 0],
        [0, opt.img_size - 1],
        [opt.img_size - 1, 0]
    ])

    img_crop_list = []
    img_raw_list = []
    input_tensor_list = []
    input_tensor_raw_list = []
    tform_inv_list = []
    left_list = []
    right_list = []
    top_list = []
    bottom_list = []

    for index in range(num_faces):

        x_min = bboxes[index][0]
        y_min = bboxes[index][1]
        x_max = bboxes[index][2]
        y_max = bboxes[index][3]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        size = max(w, h)
        ss = np.array([0.75, 0.75, 0.75, 0.75])

        left = x_center - ss[0] * size
        right = x_center + ss[1] * size
        top = y_center - ss[2] * size
        bottom = y_center + ss[3] * size

        left_list.append(left)
        right_list.append(right)
        top_list.append(top)
        bottom_list.append(bottom)

        src_pts = np.float32([
            [left, top],
            [left, bottom],
            [right, top]
        ])

        tform = cv2.getAffineTransform(src_pts, dst_pts)
        tform_inv = cv2.getAffineTransform(dst_pts, src_pts)
        img_crop = cv2.warpAffine(img, tform, (opt.img_size,) * 2)
        img_raw = cv2.resize(img, (opt.img_size * 2, opt.img_size * 2))

        tform_inv_list.append(tform_inv)
        img_crop_list.append(img_crop)
        img_raw_list.append(img_raw)

        input_tensor = tfm_test(Image.fromarray(img_crop)).unsqueeze(0)
        input_raw_tensor = tfm_test(Image.fromarray(img_raw)).unsqueeze(0)
        input_tensor_list.append(input_tensor)
        input_tensor_raw_list.append(input_raw_tensor)

    left = left_list
    right = right_list
    top = top_list
    bottom = bottom_list

    img_crop_display = torchvision.utils.make_grid(
        torch.tensor(img_crop_list).permute(0, 3, 1, 2),
        nrow=4
    ).permute(1, 2, 0).numpy()

    if num_faces > 0:
        model.img = torch.cat(input_tensor_list, dim=0).to(model.device)
        with torch.no_grad():
            _, seg_pred, _, _ = model.netR.cnn(model.img)
            bs = seg_pred.shape[0]
            mask_pr = torch.argmax(seg_pred, 1).cpu().detach().numpy()
            chooses = np.zeros((bs, opt.n_pts))

            for i in range(bs):
                choose = mask_pr[i].flatten().nonzero()[0]
                if len(choose) > opt.n_pts:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:opt.n_pts] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, opt.n_pts - len(choose)), 'wrap')
                chooses[i, :] = choose

            chooses = chooses.astype(np.int64)
            chooses = torch.LongTensor(chooses).to(device='cuda')

            assign_mat, seg_pred, uv_pred = model.netR(model.img, chooses)
            verts3d_pred = F.grid_sample(uv_pred, grid.expand(bs, -1, -1, -1), align_corners=False)
            verts3d_pred = verts3d_pred.squeeze(2).permute(0, 2, 1).contiguous()

    temp2 = img.copy()
    temp3 = img.copy()

    img_size = 800
    f = 1.574437 * img_size / 2
    K_img = np.array([
        [f, 0, img_size / 2.0],
        [0, f, img_size / 2.0],
        [0, 0, 1]
    ])
    T = np.array([
        [1.0, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    inst_shape = verts3d_pred / 9.0
    verts3d_pred = inst_shape.cpu().numpy().reshape(-1, 1220, 3)
    assign_mat = F.softmax(assign_mat, dim=2)
    nocs_coords = torch.bmm(assign_mat, inst_shape)
    nocs_coords = nocs_coords.detach().cpu().numpy().reshape(-1, opt.n_pts, 3)

    for i in range(num_faces):
        choose = chooses.cpu().numpy()[i]
        choose, choose_idx = np.unique(choose, return_index=True)
        nocs_coord = nocs_coords[i, choose_idx, :]
        col_idx = choose % opt.img_size
        row_idx = choose // opt.img_size
        local_pts2d = np.concatenate((col_idx.reshape(-1, 1), row_idx.reshape(-1, 1)), axis=1)

        tform_inv = tform_inv_list[i]

        W, b = tform_inv.T[:2], tform_inv.T[2]
        global_pts68_pred = local_pts2d @ W + b

        for p in global_pts68_pred:
            cv2.circle(temp2, (int(p[0]), int(p[1])), radius=2, color=(0, 255, 0), thickness=-1)

        _, rvecs, tvecs, _ = cv2.solvePnPRansac(
            nocs_coord,
            global_pts68_pred,
            K_img,
            None
        )

        rotM = cv2.Rodrigues(rvecs)[0].T
        tvecs = tvecs.squeeze(axis=1)

        # to GL style
        R_temp = np.identity(4)
        R_temp[:3, :3] = rotM
        R_temp[3, :3] = tvecs
        R_t_pred = R_temp @ T

        temp3 = renderer(verts3d_pred[i], R_t_pred, temp3)

    temp2 = temp2[:, :, :: -1]
    temp3 = temp3[:, :, :: -1]

    cv2.imwrite(save_img_r, temp3)
    cv2.imwrite(save_img_d, temp2)



