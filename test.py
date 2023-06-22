import os
import sys
import cv2
import time
import pprint
import numpy as np
from datetime import datetime
from options.test_options import TestOptions
from models import create_model
from data import create_dataset


###usage: python -u test.py --image_size=192 --model perspnet --dataset_mode arkit --csv_path_test 'test.csv' 
if __name__ == '__main__':
    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')

    opt = TestOptions().parse()  # get test options
    opt.serial_batches = True
    opt.num_thread = 4
    dataset = create_dataset(opt)
    num_batch = len(dataset.dataloader)
    print("The number of testing images = %d" % len(dataset))
    
    model = create_model(opt)
    t_start = time.time()
    for i, data in enumerate(dataset):
        print('[%s][progresss]%d %d'%(datetime.now().isoformat(sep=' '), i, num_batch))
        if i==0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
            if model.eval:
                model.eval()
            model.init_evaluation()
        model.set_input(data)    
        model.inference_curr_batch()   
        if i==-1:
            break
    metric = dataset.dataset.compute_metrics(model.inference_data)
    print('\n [metric] ')
    pprint.pprint(metric)
    print('Done in %.3f sec' %(time.time()-t_start))





