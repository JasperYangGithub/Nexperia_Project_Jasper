# -*- coding: utf-8 -*-
"""
# @file name  : train_resnet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : resnet training on cifar10
"""
import sys
# import sys
# from prettytable import PrettyTable
# reload(sys)
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
# from tools.Nexperia_downsample_dataset import NexperiaFolder
# from tools.resnet import resnet56, resnet20
from tools.common_tools import ModelTrainer, show_confMat, plot_line, cal_auc, \
    cal_focal_loss_alpha, cal_loss_eachClass, dataset_info, correct_label, Logger
# from tools.common_tools import ModelTrainer, show_confMat, plot_line, cal_auc_train, cal_focal_loss_alpha, cal_loss_eachClass, dataset_info


# ============================ Jasper added import============================
from torchvision import models
import torchvision.datasets as datasets
import torch.nn.functional as F
from tools.focal import FocalLoss
from os.path import *
import os
from tools.Nexperia_txt_dataset import textReadDataset
import pickle
from glob import glob
from prettytable import PrettyTable
import pandas as pd
import sys
import shutil
# ============================ Jasper added import-END============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





############ Jasper added to process the empty images ###################
def collate_fn(batch):
    """
    Jasper added to process the empty images
    referece: https://github.com/pytorch/pytorch/issues/1137
    :param batch:
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":

    #++++++++ hyper parameters ++++++++#

    ###++++++++ for FL
    model_dir = '05-03_11-42_FL_res50_NexTrainSet'
    data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021",
                            "Jan2021_corrected","FP_Jan_II", "FP_rename")
    backbone = "Res50"
    method = "FL"
    test_data_name = "JanCorrect70_correct"
    best_model = '_best2'
    best_thresh = 0.01685745
    target_dir_root = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "noisy_data")
    target_dir = os.path.join(target_dir_root, model_dir + '_' + test_data_name)
    if not exists(target_dir):
        os.mkdir(target_dir)
    #++++++++ hyper parameters end ++++++++#

    name_test1, labels_test1 = dataset_info(join(data_dir, '..', 'FP_rename.txt'))


    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str = time_str + "_test" + "_" + "_".join([backbone, method, test_data_name])
    log_dir = os.path.join(BASE_DIR, "../..", "results", "model_test_results", time_str)




    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)

    class_names = (
    'Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass', 'Foreign_material', 'Empty_pocket', 'Device_flip',
    'Chipping')

    table = PrettyTable(['Model', 'Epoch', 'ACC', 'LOSS', 'FPR_980', 'FPR_995', 'FPR_100','AUC'])

    table_loss_name = list(class_names)
    table_loss_name.insert(0, 'Model')
    table_loss = PrettyTable(table_loss_name)

    table_badScore_name = list(class_names)
    table_badScore_name.insert(0, 'Model')
    table_badScore_name.insert(1, 'Thr998')
    table_badScore = PrettyTable(table_badScore_name)
    # table_loss = PrettyTable(['Model', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    num_classes = len(class_names)

    BATCH_SIZE = 16


    # ============================ step 1/5 数据 ============================

    ############ Nexperial_compare_traingSet Train set ############
    norm_mean = [0.2391, 0.2391, 0.2391]
    norm_std = [0.1365, 0.1365, 0.1365]

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_data1 = textReadDataset(data_dir, name_test1, labels_test1, valid_transform)

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data1, collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # ============================ step 2/5 模型 ============================
    ###### ResNet50 #######
    resnet_model = models.resnet50(num_classes=num_classes)

    ###### Load Model #######
    saved_model_dir = os.path.join(BASE_DIR, "../..", "savedModel", model_dir)
    os.chdir(saved_model_dir)
    print("Current working directory: {0}".format(os.getcwd()))

    for model_name in sorted(glob("*.pth")):
        checkpoint_file_loc = os.path.join(saved_model_dir, model_name)
        print("")
        print("________________________________________________"+model_name)
        print("")

        checkpoint = torch.load(checkpoint_file_loc)
        state_dict = checkpoint['model']
        train_epoch = checkpoint['epoch']


        resnet_model.load_state_dict(state_dict)
        resnet_model.to(device)

        # ============================ step 3/5 损失函数 ============================

        # criterion = FocalLoss()
        criterion = nn.CrossEntropyLoss()
        # ============================ step 4/5 优化器 ============================



        # ============================ step 5/5 test ============================
        # acc_valid, mat_valid, y_true_valid, y_outputs_valid = model_test(test_loader, resnet_model, device)
        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_train, logits_val = ModelTrainer.valid(test_loader,
                                                                                                     resnet_model,
                                                                                                     criterion, device)


        y_score = torch.cat(y_outputs_train).cpu()
        bad_score = 1 - y_score[:,4]


        # ============================ save bad score ============================
        all_image_path = name_test1
        all_image_label = labels_test1
        bad_score_list = list(bad_score.numpy())
        output_dict = {'Path': all_image_path, 'Label':all_image_label, 'Bad_score': bad_score}
        output_df = pd.DataFrame(data = output_dict)
        output_df['Bad_score_mean'] = output_df.groupby('Label')['Bad_score'].transform('mean')
        fileName = model_name[:-4] + "_badScore.xlsx"
        output_df.to_excel(log_dir + "/" + fileName)

    # ============================ extract noisy data ============================
    cls2int = {
        'Others': 1,
        'Marking_defect': 2,
        'Lead_glue': 3,
        'Lead_defect': 4,
        'Pass': 5,
        'Foreign_material': 6,
        'Empty_pocket': 7,
        'Device_flip': 8,
        'Chipping': 9,
        # 'Pocket_damage':3,
        # 'scratch':2,
        # 'tin_particles':6
    }
    class_l = list(cls2int.keys())

    ######## acquire possible noisy data
    bad_score_file_name = best_model +'_badScore.xlsx'
    best1_file = os.path.join(log_dir, bad_score_file_name)
    df = pd.read_excel(best1_file)
    label_groups = df.groupby('Label')
    noisy_data = df[df.Bad_score>=best_thresh]

    ######## created folders to save the possible noisy data
    noisy_list = noisy_data.Label.unique()
    for CLASS in noisy_list:
        if not exists(join(target_dir, class_l[CLASS-1])):
            os.mkdir(join(target_dir, class_l[CLASS-1]))


    ######## copy the possible noisy data to created folders
    os.chdir(data_dir)
    for i in range(len(noisy_data)):
        CLASS_name = class_l[noisy_data.Label.iloc[i]-1]
        shutil.copy(noisy_data.Path.iloc[i], join(target_dir,CLASS_name))

    print("the possible noisy images amount is: {}".format(len(noisy_data)))
    print("the best score file is: {}".format(bad_score_file_name))
    print('Use the method: {} with threshold: {}'.format(method,best_thresh))


    # ============================ plot noisy data ============================
    plt_y = df.Bad_score
    plt_x =  np.arange(0,len(df))
    # plt.plot(plt_x, plt_y, label=class_l[i] + "_" + str(len(plt_y)), linewidth=lw)
    plt.scatter(plt_x, plt_y, s=30, label='Jan_corrected_data' + "_" + str(len(noisy_data)))


    plt.plot(np.arange(0, len(df)), np.ones(len(df)) * best_thresh, label='Thr_980_' + str(best_thresh), linewidth=2, color = 'red')

    plt.ylabel("bad_score")
    plt.xlabel('data_points')
    plt.ylim(0,1)
    location = 'upper center'
    # location = 'lower right'
    plt.title(method + '_' +best_model+ '_'+test_data_name)
    plt.legend(loc=location)

    plt.savefig(os.path.join(target_dir, 'bad_score.svg'), format='svg', dpi=600)
    plt.show()
    plt.close()



    print(" done ~~~~ {} ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M')))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f








