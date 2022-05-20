"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import os

##z
def test():
    """ Training
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    ##
    # ARGUMENTS
    opt = Options().parse()

    # define test args by Jasper
    opt.dataset = 'nexperia_test'
    opt.load_weights = True
    opt.extralayers = 3
    opt.manualseed = 0
    opt.isize = 256
    opt.batchsize = 16
    opt.niter = 200
    opt.nz = 256
    opt.display = True
    opt.isTrain = True
    opt.ouf = './output'
    opt.w_adv = 1
    opt.w_con = 50
    opt.w_enc = 10
    opt.save_test_images = False
    opt.save_image_freq = 100



    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL
    # model.train()
    model.test()

if __name__ == '__main__':
    test()
