import pdb
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset
from data.augmentation import EulerAugmentor, HorizontalFlipAugmentor

use_jpeg4py = False
data_root = Path('./dataset/ARKitFace')


npz_path = 'npy/kpt_ind.npy'
kpt_ind = np.load(npz_path)



class BIWIDataset(BaseDataset):

    def __init__(self, opt):

        self.data_root = data_root
        self.opt = opt
        self.is_train = False
        self.img_size = opt.img_size

        csv_path = data_root / 'csv/metadata_biwi.csv'

        self.df = pd.read_csv(csv_path, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
            nrows=2721 if opt.debug else None)
        self.df['data_batch'] += '_v1'


        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.tfm_train = transforms.Compose([
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





    def __getitem__(self, index):
        data_batch = str(self.df['data_batch'][index])
        subject_id = str(self.df['subject_id'][index])
        facial_action = str(self.df['facial_action'][index])
        img_id = str(self.df['img_id'][index])

        img_path = self.data_root / 'image' / data_batch / subject_id / facial_action / f'{img_id}_ar.jpg'
        npz_path = self.data_root / 'info' / data_batch / subject_id / facial_action / f'{img_id}_info.npz'
        M = np.load(npz_path)

        if use_jpeg4py:
            img_raw = jpeg.JPEG(img_path).decode()
        else:
            img_raw = cv2.imread(str(img_path))
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = img_raw.shape




        if self.opt.use_gt_bbox:
            points2d_68 = M['points2d'][self.kpt_ind]   # (68, 2)
        else:
            raise ValueError('please add --use_gt_bbox')


        
        x_min, y_min = points2d_68.min(axis=0)
        x_max, y_max = points2d_68.max(axis=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        size = max(w, h)
        ss = np.array([0.75, 0.75, 0.75, 0.75])

        if self.is_train:
            rnd_size = 0.15 * size
            dx = np.random.uniform(-rnd_size, rnd_size)
            dy = np.random.uniform(-rnd_size, rnd_size)
            x_center, y_center = x_center + dx, y_center + dy 
            ss *= np.random.uniform(0.95, 1.05)
 
        left = x_center - ss[0] * size
        right = x_center + ss[1] * size
        top = y_center - ss[2] * size
        bottom = y_center + ss[3] * size


        src_pts = np.float32([
            [left, top],
            [left, bottom],
            [right, top]
        ])

        tform = cv2.getAffineTransform(src_pts, self.dst_pts)
        img = cv2.warpAffine(img_raw, tform, (self.img_size,)*2)

        if self.is_train:
            img = self.tfm_train(Image.fromarray(img))
        else:
            img = self.tfm_test(Image.fromarray(img))



        points2d_68 = M['points2d'][self.kpt_ind]   # 
        W, b = tform.T[:2], tform.T[2]
        local_points2d_68 = points2d_68 @ W + b
        pts68 = torch.tensor(local_points2d_68, dtype=torch.float32) / self.img_size * 2 - 1  # 范围[-1, 1]


        verts3d = torch.tensor(M['verts_gt'], dtype=torch.float32) * 4.5 * 2   # [-1, 1]
        choose = torch.tensor(np.random.rand(1024).astype(np.int64))
        mask = torch.tensor(np.zeros((192,192)).astype(np.int64))
        
        nocs = torch.tensor(np.random.rand(1024,3).astype(np.float32))
        model = torch.tensor(np.random.rand(1220,3).astype(np.float32)) ### for biwi, please change it to GT model!
        corr_mat = torch.tensor(np.random.rand(1024,1220).astype(np.float32))
        uvmap = torch.tensor(np.random.rand(192,192,3), dtype=torch.float32).permute(2, 0, 1)
        d = {
            'img': img,
            'choose': choose,
            'model': model,
            'uvmap': uvmap,
            'nocs': nocs,
            'mask': mask,
            'corr_mat': corr_mat
        }

        if hasattr(self.opt, 'eval'):
            tform_inv = cv2.getAffineTransform(self.dst_pts, src_pts) 
            d['tform_inv'] = tform_inv
            d['R_t'] = M['R_t']
            d['img_path'] = str(img_path)

        return d



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
        loss_total = (loss1 * bs_list).sum() / bs_list.sum() 
        loss_corr = (loss2 * bs_list).sum() / bs_list.sum() 
        loss_recon3d = (loss3 * bs_list).sum() / bs_list.sum()
        loss_entropy = (loss4 * bs_list).sum() / bs_list.sum()
        loss_mat = (loss5 * bs_list).sum() / bs_list.sum()
        loss_seg = (loss6 * bs_list).sum() / bs_list.sum()
        d = {
                'loss_total': loss_total,
                'loss_corr': loss_corr,
                'loss_recon3d': loss_recon3d,
                'loss_entropy': loss_entropy,
                'loss_mat': loss_mat,
                'loss_seg': loss_seg
                }
        if hasattr(self.opt, 'eval'):
            d['pnp_fail'] = inference_data['pnp_fail']
            d['3DRecon'] = np.mean(inference_data['3DRecon'])
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

            # 单位转换
            d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
            d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
            d['pitch_mae'] = '%.2f °' % d['pitch_mae']
            d['yaw_mae'] = '%.2f °' % d['yaw_mae']
            d['roll_mae'] = '%.2f °' % d['roll_mae']
            d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
            d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
            d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)
            d['5°5cm'] = '%.2f' % (d['5°5cm'] * 100)
            d['5°10cm'] = '%.2f' % (d['5°10cm'] * 100)
            d['mean_IoU'] = '%.4f' % d['mean_IoU']

        return d
