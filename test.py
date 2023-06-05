import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import os
import sys
import cv2
import time
import pprint
import numpy as np
from datetime import datetime
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import pdb

if __name__ == '__main__':
    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')

    opt = TestOptions().parse()  # get test options

    # 这些选项是后面改的，所以打印出来的参数是修改之前的
    #opt.batch_size = 32    # 测试时的batch_size可以设置大一些
    opt.serial_batches = True
    opt.num_thread = 4
    if opt.test_visualization:
        opt.serial_batches = False




    dataset = create_dataset(opt)
    num_batch = len(dataset.dataloader)
    print('The number of testing images = %d' % len(dataset))


    model = create_model(opt)      # create a model given opt.model and other options




    t_start = time.time()
    for i, data in enumerate(dataset):
        print('[%s]【progress】%d/%d' % (datetime.now().isoformat(sep=' '), i, num_batch))

        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
            model.init_evaluation()

        model.set_input(data)  # unpack data from data loader
        model.inference_curr_batch()

        if i == -1:
            break
    
        if opt.test_visualization:
            break





    if not opt.test_visualization:
        #pdb.set_trace()
        metric = dataset.dataset.compute_metrics(model.inference_data)
        print('\n【metric】')
        pprint.pprint(metric)
        print('Done in %.2f sec' % (time.time() - t_start))

    else:
        os.makedirs('debug_dir', exist_ok=True)

        img = model.img.permute(0, 2, 3, 1).cpu().numpy()
        img = ((img + 1) * 0.5 * 255).astype(np.uint8)

        pts68_pred = model.out1.cpu().numpy().reshape(-1, 68, 2)
        pts68_pred = (pts68_pred + 1) * 0.5 * opt.img_size


        fig, axes = plt.subplots(7, 7, figsize=(32, 32))

        for k, ax in enumerate(axes.flat):
            index = np.random.randint(opt.batch_size)

            cur_img = img[index]
            cur_pts68 = pts68_pred[index]

            temp = cur_img.copy()
            for p in cur_pts68:
                cv2.circle(temp, ( int(p[0]), int(p[1]) ), radius=3, color=(0, 255, 0), thickness=-1)

            ax.imshow(temp)
            ax.axis('off')

        plt.savefig('debug_dir/display.jpg')
