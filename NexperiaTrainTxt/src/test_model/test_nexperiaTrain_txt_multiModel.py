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

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from tools.common_tools import ModelTrainer, show_confMat, plot_line, cal_auc, \
    cal_focal_loss_alpha, cal_loss_eachClass, dataset_info, correct_label, Logger
# from tools.common_tools import ModelTrainer, show_confMat, plot_line, cal_auc_train, cal_focal_loss_alpha, cal_loss_eachClass, dataset_info


# ============================ Jasper added import============================
from torchvision import models
import torchvision.datasets as datasets
import torch.nn.functional as F
from tools.focal import FocalLoss
from os.path import join, dirname
import os
from tools.Nexperia_txt_dataset import textReadDataset
import pickle
from glob import glob
from prettytable import PrettyTable
import pandas as pd
# ============================ Jasper added import-END============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    model_dir = '05-03_11-42_FL_res50_NexTrainSet'
    data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021", "Feb2021")
    backbone = "Res50"
    method = "FL"
    test_data_name = "Feb2021_down_img256"

    name_test1, labels_test1 = dataset_info(join(data_dir, 'Feb2021_train_down.txt'))
    name_test2, labels_test2 = dataset_info(join(data_dir, 'Feb2021_val_down.txt'))
    name_test3, labels_test3 = dataset_info(join(data_dir, 'Feb2021_test_down.txt'))
    # ++++++++ hyper parameters end ++++++++#


    #++++++++ correct labels by using Jan_correct.txt ++++++++#
    # name_correct, labels_correct = dataset_info(join(Jan_correct_dir, 'Jan_correct.txt'))
    # labels_test1_correct = correct_label(name_test1, labels_test1, name_correct)
    # labels_test2_correct = correct_label(name_test2, labels_test2, name_correct)
    # labels_test3_correct = correct_label(name_test3, labels_test3, name_correct)


    # ++++++++ log_dir ++++++++#
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


    names = name_test1 + name_test2 + name_test3
    labels = labels_test1 + labels_test2 + labels_test3
    np_names_labels = np.array([names, labels]).transpose()
    np_names_labels_unique = np.unique(np_names_labels, axis=0)
    #### remove the corrupted image just for Feb
    corrupted_img = np.where(np_names_labels_unique[:,0] == 'Pass/20210226_WEPA06311D3A_15_1_377_1.bmp')
    np_names_labels_unique = np.delete(np_names_labels_unique, corrupted_img[0].item(),0)

    names_list = np_names_labels_unique[:,0].tolist()
    labels_list = np_names_labels_unique[:,1].tolist()
    labels_list =  list(map(int, labels_list))
    combined_data = textReadDataset(data_dir, names_list, labels_list, valid_transform)


    # 构建DataLoder
    test_loader = DataLoader(dataset=combined_data, collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # ============================ step 2/5 模型 ============================
    ###### ResNet50 #######
    resnet_model = models.resnet50(num_classes=num_classes)
    # num_ftrs = resnet_model.fc.in_features
    # resnet_model.fc = nn.Linear(num_ftrs, num_classes)

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


        # ============================ step 5/5 test ============================
        # acc_valid, mat_valid, y_true_valid, y_outputs_valid = model_test(test_loader, resnet_model, device)
        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid, logits_val = ModelTrainer.valid(test_loader,
                                                                                                     resnet_model,
                                                                                                     criterion, device)

        # roc_auc_valid, pr_auc_valid, fpr_valid = cal_auc(y_true_valid, y_outputs_valid)
        roc_auc_valid, pr_auc_valid, fpr_dict_val, threshhold980, badScore_class_acc, bad_score = cal_auc(y_true_valid, y_outputs_valid, log_dir)

        # print("test Acc:{:.2%} test fpr:{:.2%} test AUC:{:.2%} ".format(acc_valid, fpr_980_rec, roc_auc_valid))

        fpr_980_val = fpr_dict_val['fpr_980']
        fpr_990_val = fpr_dict_val['fpr_995']
        fpr_100_val = fpr_dict_val['fpr_1']
        loss_class_val = cal_loss_eachClass(logits_val, y_true_valid, num_classes)
        loss_class_val = np.around(loss_class_val,3)


        print("test Acc:{:.2%} test LOSS:{:.3f} test fpr980:{:.2%} test AUC:{:.2%} Epoch:{:.0f}".format(acc_valid, loss_valid, fpr_980_val, roc_auc_valid, train_epoch))
        print("loss of each class is:")
        print(["{0:0.3f} ".format(k) for k in list(loss_class_val)])



        # ============================ Put data into table ===========================
        table.add_row([model_name[:-4], train_epoch, round(acc_valid, 2), round(loss_valid, 3),
                       round(fpr_980_val, 3), round(fpr_990_val, 3), round(fpr_100_val, 3), round(roc_auc_valid, 3)])

        loss_class_list = list(loss_class_val)
        loss_class_list.insert(0, model_name[:-4])
        table_loss.add_row(loss_class_list)

        table_badScore_list = badScore_class_acc
        table_badScore_list.insert(0, model_name[:-4])
        table_badScore_list.insert(1, threshhold980)
        table_badScore.add_row(table_badScore_list)

        # ============================ 保存测试结果 ============================
        test_results_dict = {"acc": acc_valid, "loss": loss_valid, "roc_auc_rec": roc_auc_valid, "fpr":fpr_dict_val, "confMat":mat_valid, "loss_class":loss_class_val}
        fileName = model_name[:-4] + "_results.pkl"
        test_results_file = open(join(log_dir + "/" + fileName), "wb")
        pickle.dump(test_results_dict, test_results_file)
        test_results_file.close()

        # ============================ save bad score ============================
        all_image_path = names_list
        all_image_label = labels_list
        bad_score_list = list(bad_score.numpy())
        output_dict = {'Path': all_image_path, 'Label':all_image_label, 'Bad_score': bad_score}
        output_df = pd.DataFrame(data = output_dict)
        output_df['Bad_score_mean'] = output_df.groupby('Label')['Bad_score'].transform('mean')
        fileName = model_name[:-4] + "_badScore.xlsx"
        output_df.to_excel(log_dir + "/" + fileName)

        show_confMat(mat_valid, class_names, model_name, log_dir, verbose=True)

    print(table)
    print(table_loss)
    print(table_badScore)
    print(" done ~~~~ {} ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M')))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f








