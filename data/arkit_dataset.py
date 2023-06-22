import pdb
import cv2
import socket
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.augmentation import EulerAugmentor, HorizontalFlipAugmentor
from util.util import landmarks106to68
from lib import mesh
from lib import mesh_ori
from lib.eyemouth_index import vert_index, face_em
import time

use_jpeg4py = False

data_root = Path('./dataset')

npz_path = 'npy/kpt_ind.npy'
kpt_ind = np.load(npz_path)

npy_path = 'npy/uv_coords_std_202109.npy'
uv_coords_std = np.load(npy_path)   # (1279, 2)，[0, 1]

npy_path = 'npy/tris_2500x3_202110.npy'
tris_full = np.load(npy_path)
npy_path = 'npy/tris_2388x3.npy'
tris_mask = np.load(npy_path)
txt_path = 'npy/projection_matrix.txt'
M_proj = np.loadtxt(txt_path, dtype=np.float32)


class ARKitDataset(BaseDataset):
    def __init__(self, opt):
        self.data_root = data_root
        self.opt = opt
        self.is_train = opt.isTrain
        self.img_size = opt.img_size
        self.n_pts = opt.n_pts
        if self.is_train:
            self.df = pd.read_csv(opt.csv_path_train, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                nrows=2721 if opt.debug else None)
            self.rnd=np.random.permutation(len(self.df))
        else:
            self.df = pd.read_csv(opt.csv_path_test, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                nrows=1394 if opt.debug else None)

        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.tfm_train = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        self.tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        self.kpt_ind = kpt_ind
        self.dst_pts = np.float32([
            [0, 0],
            [0, opt.img_size - 1],
            [opt.img_size - 1, 0]
        ])

        self.faces=np.load('./data/triangles.npy')              
        ############ uv map ############
        self.uv_size = self.img_size
        uv_coords = uv_coords_std * (self.uv_size - 1)
        zeros = np.zeros([uv_coords.shape[0], 1])
        self.uv_coords_extend = np.concatenate([uv_coords, zeros], axis=1)
        self.tris_full = tris_full
        self.contour_ind = [
                20, 853, 783, 580, 659, 660, 765, 661, 616, 579,
                489, 888, 966, 807, 730, 1213, 1214, 1215, 1216, 822,
                906, 907, 908, 909, 910, 911, 912, 913, 1047, 914,
                915, 916, 917, 918, 919, 920, 921, 392, 462, 905,
                904, 208, 295, 376, 57, 467, 39, 130, 167, 213,
                330, 212, 211, 131, 352, 425
                ]
        self.mouth_ind = [24, 691, 690, 689, 688, 687, 686, 685, 823, 684, 834, 740, 683, 682, 710, 725, 709, 700,
                25, 265, 274, 290, 275, 247, 248, 305, 404, 249, 393, 250, 251, 252, 253, 254, 255, 256]
        self.eye1_ind = [1101, 1100, 1099, 1098, 1097, 1096, 1095, 1094, 1093, 1092, 1091, 1090,
                1089, 1088, 1087, 1086, 1085, 1108, 1107, 1106, 1105, 1104, 1103, 1102]
        self.eye2_ind = [1081, 1080, 1079, 1078, 1077, 1076, 1075, 1074, 1073, 1072, 1071, 1070,
                1069, 1068, 1067, 1066, 1065, 1064, 1063, 1062, 1061, 1084, 1083, 1082]
        self.tris_mask = tris_mask

    def generate_uv_position_map(self, verts):
        temp1 = verts[self.contour_ind] * 1.1
        temp2 = verts[self.mouth_ind].mean(axis=0, keepdims=True)
        temp3 = verts[self.eye1_ind].mean(axis=0, keepdims=True)
        temp4 = verts[self.eye2_ind].mean(axis=0, keepdims=True)
        verts_ = np.vstack([verts, temp1, temp2, temp3, temp4])  # (1279, 3)
        uv_map = mesh_ori.render.render_colors(self.uv_coords_extend, self.tris_full, verts_, h=self.uv_size, w=self.uv_size, c=3)   # 范围[0, 1]
        uv_map = np.clip(uv_map, 0, 1)
        return uv_map

    def get_item(self, index):
        try:
            if self.is_train:
                index =self.rnd[index]
            subject_id = str(self.df['subject_id'][index])
            facial_action = str(self.df['facial_action'][index])
            img_id = str(self.df['img_id'][index])            
            img_path = self.data_root / 'image' / subject_id / facial_action / f'{img_id}_ar.jpg'
            npz_path = self.data_root / 'info' / subject_id / facial_action / f'{img_id}_info.npz'
            img_path = str(img_path)
            npz_path = str(npz_path)
            M = np.load(npz_path)            
            ###3d model and mean shape
            model = M['verts']
            R_t = M['R_t']

            img_raw = cv2.imread(img_path)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = img_raw.shape
            
            ###render for coord and mask
            ones = np.ones([model.shape[0], 1])
            verts_homo = np.concatenate([model, ones], axis=1)

            assert R_t[3, 2] < 0  # tz is always negative
            M1 = np.array([
                [img_w / 2, 0, 0, 0],
                [0, img_h / 2, 0, 0],
                [0, 0, 1, 0],
                [img_w / 2, img_h / 2, 0, 1]
            ])
            # world space -> camera space -> NDC space -> image space
            verts = verts_homo @ R_t @ M_proj @ M1
            z = verts[:, [3]]
            image_vertices = verts / z
            image_vertices[:, 2] = -z[:, 0]
            image_vertices = image_vertices[:,:3]
            tx, ty, tz = R_t[3, :3]
            tz = -tz           
            attribute = (model*4.5+0.5)*255
            coord, corr_weight = mesh.render.render_colors(image_vertices, self.faces, attribute, img_h, img_w, c=3)
            coord = coord/255.0
            
            att = np.ones((1220,1))
            att[vert_index] = 0
            tris_organ = vert_index[face_em]
            triangles = np.concatenate((self.faces, tris_organ), axis=0)
            mask = mesh_ori.render.render_colors(image_vertices, triangles, att, img_h, img_w, c=1)
            mask = (np.squeeze(mask)>0).astype(np.uint8)
                 
            roi_cnt = 0.5*(np.max(image_vertices[self.kpt_ind,:],0)[0:2] + np.min(image_vertices[self.kpt_ind,:],0)[0:2])
            size = np.max(np.max(image_vertices[self.kpt_ind,:],0)[0:2] - np.min(image_vertices[self.kpt_ind,:],0)[0:2])
            x_center = roi_cnt[0]
            y_center = roi_cnt[1]
            ss = np.array([0.75, 0.75, 0.75, 0.75])

            # center
            if self.is_train:
                rnd_size = 0.15 * size
                dx = np.random.uniform(-rnd_size, rnd_size)
                dy = np.random.uniform(-rnd_size, rnd_size)
                x_center, y_center = x_center + dx, y_center + dy 
                ss *= np.random.uniform(0.95, 1.05)
 
            left = int(x_center - ss[0] * size)
            right = int(x_center + ss[1] * size)
            top = int(y_center - ss[2] * size)
            bottom = int(y_center + ss[3] * size)

            src_pts = np.float32([
                [left, top],
                [left, bottom],
                [right, top]
            ])

            tform = cv2.getAffineTransform(src_pts, self.dst_pts)
            img = cv2.warpAffine(img_raw, tform, (self.img_size,)*2)
            img_raw = cv2.resize(img_raw, (self.img_size*2, self.img_size*2))

            if self.is_train:
                img = self.tfm_train(Image.fromarray(img))
                img_raw = self.tfm_train(Image.fromarray(img_raw))
            else:
                img = self.tfm_test(Image.fromarray(img))
                img_raw = self.tfm_test(Image.fromarray(img_raw))
                
                
            ### sample points
            choose = mask[top:bottom+1, left:right+1].flatten().nonzero()[0]
            #print(len(choose))
            if len(choose) > self.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.n_pts-len(choose)), 'wrap')

            ##### choose to full pic
            choose_raw = choose * self.img_size*2 / 800
            choose_raw = choose_raw.astype(np.int64)
            if choose_raw.max() > self.img_size*2 * self.img_size*2 - 1:
                choose_raw[choose_raw > self.img_size*2 * self.img_size*2 - 1] = np.random.randint(0, self.img_size*2 * self.img_size*2)

            if choose_raw.min() < 0:
                choose_raw[choose_raw < 0] = np.random.randint(0, self.img_size*2 * self.img_size*2)

            nocs = coord[top:bottom+1, left:right+1, :].reshape((-1, 3))[choose, :] - 0.5
            corr2d_3d = corr_weight[top:bottom+1, left:right+1, :].reshape((-1,6))[choose,:]####[id0,id1,id2,w0,w1,w2]
            corr_mat = np.zeros((self.n_pts,1220))
            corr_mat[np.arange(self.n_pts),corr2d_3d[:,0].astype(np.int16)] = corr2d_3d[:,3]
            corr_mat[np.arange(self.n_pts),corr2d_3d[:,1].astype(np.int16)] = corr2d_3d[:,4]
            corr_mat[np.arange(self.n_pts),corr2d_3d[:,2].astype(np.int16)] = corr2d_3d[:,5]

            ##### choose to local bbox
            crop_w = right-left + 1
            col_idx = choose % crop_w + left
            row_idx = choose // crop_w + top
            W, b = tform.T[:2], tform.T[2]
            point2d = np.concatenate((col_idx.reshape(-1,1), row_idx.reshape(-1,1)), axis=1)
            #print(W,b)
            point2d = point2d @ W + b
            #col_idx = col_idx *W[0] + b
            #row_idy = row_idy *W[1] + b
            choose = (np.floor(point2d[:,1])  * self.img_size + np.floor(point2d[:,0])).astype(np.int64)
            if choose.max()>self.img_size*self.img_size-1:
                choose[choose>self.img_size*self.img_size-1] = np.random.randint(0,self.img_size*self.img_size)

            if choose.min()<0:
                choose[choose<0] = np.random.randint(0,self.img_size*self.img_size)

            mask = cv2.warpAffine(mask, tform, (self.img_size,self.img_size))
            mask = torch.tensor(np.array(mask>0).astype(np.int64))
            nocs = torch.tensor(nocs.astype(np.float32))
            model = torch.tensor(model.astype(np.float32))
            corr_mat = torch.tensor(corr_mat.astype(np.float32))
            

            # tform_inv
            dst_pts2 = np.float32([
            [0, 0],
            [0, self.img_size*2 - 1],
            [self.img_size*2 - 1, 0]
        ])
            tform_inv = cv2.getAffineTransform(dst_pts2, src_pts)
            W_inv = tform_inv.T[:2, :2].astype(np.float32)
            b_inv = tform_inv.T[2].astype(np.float32)

            ############ uv map ############
            # UV position map
            verts = M['verts'] * 4.5 + 0.5     # [-0.111, 0.111] -> [0, 1]
            uvmap = self.generate_uv_position_map(verts)
            uvmap = uvmap * 2 - 1     # [0, 1] -> [-1, 1]
            uvmap = torch.tensor(uvmap, dtype=torch.float32).permute(2, 0, 1)

            d = {
                'img': img,
                'img_raw': img_raw,
                'tz': tz,
                'choose': choose,
                'choose_raw': choose_raw,
                'model': model,
                'uvmap': uvmap,
                'nocs': nocs,
                'mask': mask,
                'corr_mat': corr_mat,
                'W_inv': W_inv,
                'b_inv': b_inv,
                'left': left,
                'right': right,
                'bottom': bottom,
                'top': top

            }

            if hasattr(self.opt, 'eval'):
                tform_inv = cv2.getAffineTransform(self.dst_pts, src_pts)
                #R_t = M['faceAnchortransform'] @ np.linalg.inv(M['cameratransform'])
                R_t = M['R_t']
                d['tform_inv'] = tform_inv
                d['R_t'] = R_t
                d['img_path'] = str(img_path)
                #d['data_batch'] = str(data_batch)
                d['subjectid'] = str(subject_id)
                d['imgid'] = str(img_id)
                d['facial_action'] = str(facial_action)

            return d
        except:
            print(self.df['img_id'][index],'error!')
            return None

    def __getitem__(self, idx):
        data = self.get_item(idx)
        while data is None:
            idx = np.random.randint(0, len(self.df))
            data = self.get_item(idx)
        return data

    def __len__(self):
        return len(self.df)

    def compute_metrics(self, inference_data):
        bs_list = np.array(inference_data['batch_size'])
        loss1 = np.array(inference_data['loss_total'])
        loss2 = np.array(inference_data['loss_corr'])
        loss3 = np.array(inference_data['loss_recon3d'])
        loss4 = np.array(inference_data['loss_uv'])
        loss5 = np.array(inference_data['loss_mat'])
        loss6 = np.array(inference_data['loss_seg'])
        # loss7 = np.array(inference_data['loss_tz'])
        loss_total = (loss1 * bs_list).sum() / bs_list.sum() 
        loss_corr = (loss2 * bs_list).sum() / bs_list.sum() 
        loss_recon3d = (loss3 * bs_list).sum() / bs_list.sum()
        loss_uv = (loss4 * bs_list).sum() / bs_list.sum()
        loss_mat = (loss5 * bs_list).sum() / bs_list.sum()
        loss_seg = (loss6 * bs_list).sum() / bs_list.sum()
        # loss_tz = (loss7 * bs_list).sum() / bs_list.sum()
        d = {
            'loss_total': loss_total,
            'loss_corr': loss_corr,
            'loss_recon3d': loss_recon3d,
            'loss_uv': loss_uv,
            'loss_mat': loss_mat,
            'loss_seg': loss_seg,
        }
        if hasattr(self.opt, 'eval'):
            d['pnp_fail'] = inference_data['pnp_fail']
            d['3DRecon'] = np.mean(inference_data['3DRecon'])
            # print(np.std(inference_data['3DRecon']))
            # print(np.median(inference_data['3DRecon']))
            d['ADD'] = np.mean(inference_data['ADD'])
            d['pitch_mae'] = np.mean(inference_data['pitch_mae'])
            d['yaw_mae'] = np.mean(inference_data['yaw_mae'])
            d['roll_mae'] = np.mean(inference_data['roll_mae'])
            d['tx_mae'] = np.mean(inference_data['tx_mae'])
            d['ty_mae'] = np.mean(inference_data['ty_mae'])
            d['tz_mae'] = np.mean(inference_data['tz_mae'])
            d['5°5cm'] = inference_data['strict_success'] / inference_data['total_count']
            d['5°10cm'] = inference_data['easy_success'] / inference_data['total_count']
            d['mean_IoU'] = np.mean(inference_data['IoU'])

            d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
            d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
            d['pitch_mae'] = '%.2f °' % d['pitch_mae']
            d['yaw_mae'] = '%.2f °' % d['yaw_mae']
            d['roll_mae'] = '%.2f °' % d['roll_mae']
            d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
            d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
            d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)
            d['tz_duli_mae'] = '%.2f mm' % (d['tz_duli_mae'] * 1000)
            d['5°5cm'] = '%.2f ' % (d['5°5cm'] * 100)
            d['5°10cm'] = '%.2f' % (d['5°10cm'] * 100)
            d['mean_IoU'] = '%.4f' % d['mean_IoU']

        return d
