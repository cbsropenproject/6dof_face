import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import pdb
import cv2
import time
import copy
import numpy as np
import torch
from datetime import datetime
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')

    train_opt = TrainOptions().parse()   # get training options
    train_dataset = create_dataset(train_opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.

    
    val_opt = copy.deepcopy(train_opt)
    val_opt.isTrain = False
    val_opt.batch_size = 64
    val_opt.num_thread = 6
    val_opt.serial_batches = True
    val_dataset = create_dataset(val_opt)

    print('The number of training images = %d' % train_dataset_size)
    print('The number of val images = %d\n' % len(val_dataset))


    model = create_model(train_opt)      # create a model given opt.model and other options
    visualizer = Visualizer(train_opt)   # create a visualizer that display/save images and plots
    train_opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(train_opt.epoch_count, train_opt.n_epochs + train_opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch


        #train_dataset.set_epoch(epoch)
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data['img'].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(train_opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            optimize_start_time = time.time()
            if epoch ==2 and i==0:
                train_opt.epoch_count=2
                train_opt.lr=0.0001
                train_opt.continue_train=True
                model = create_model(train_opt)      # create a model given opt.model and other options

            if epoch == train_opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(train_opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()


            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights


            if len(train_opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % train_opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % train_opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % train_opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if train_opt.display_id is None or train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % train_opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(train_opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % train_opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.n_epochs + train_opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.





        # 在验证集上做evaluation
        if epoch % 1 == 0:
            print('===================== evaluation epoch %d begin =====================' % epoch)
            t_start = time.time()
            model.init_evaluation()
            num_batch = len(val_dataset.dataloader)

            for j, data in enumerate(val_dataset):
                if (j + 1) % 50 == 0:
                    print('[%s]【progress】%d/%d' % (datetime.now().isoformat(sep=' '), j + 1, num_batch))
                model.set_input(data)
                model.inference_curr_batch()
 
            metrics = val_dataset.dataset.compute_metrics(model.inference_data)
            print(metrics)
            print('Done in %.2f sec' % (time.time() - t_start))
            print('===================== evaluation epoch %d end =====================\n' % epoch)





