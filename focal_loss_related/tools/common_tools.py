# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : 通用函数
"""
import numpy as np
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
from sklearn import metrics



class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch):
        # print("data_loader length is {}".format(len(data_loader)))
        model.train()

        conf_mat = np.zeros((8, 8))
        loss_sigma = []
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs)

            optimizer.zero_grad()
            # optimizer.module.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 300 == 300 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append

    @staticmethod
    def valid(data_loader, model, loss_f, gpu):
        model.eval()

        conf_mat = np.zeros((8, 8))
        loss_sigma = []
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())

            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append




def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(classes)):
            confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' or mode == 'roc_auc' or mode == 'pr_auc' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()



def cal_auc(y_true_train, y_outputs_train):
    """
    计算PR_AUC 和 ROC_AUC
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    y_true = torch.cat(y_true_train).cpu()
    y_score = torch.cat(y_outputs_train).cpu()
    roc_auc = metrics.roc_auc_score(y_true != 4, 1. - y_score[:, 4])
    precision, recall, _ = metrics.precision_recall_curve(y_true != 4, 1. - y_score[:, 4])
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(y_true != 4, 1. - y_score[:, 4])
    fpr_98 = fpr[np.where(tpr >= 0.98)[0][0]]
    # fpr_991 = fpr[np.where(tpr >= 0.991)[0][0]]
    # fpr_993 = fpr[np.where(tpr >= 0.993)[0][0]]
    # fpr_995 = fpr[np.where(tpr >= 0.995)[0][0]]
    # fpr_997 = fpr[np.where(tpr >= 0.997)[0][0]]
    # fpr_999 = fpr[np.where(tpr >= 0.999)[0][0]]
    # fpr_1 = fpr[np.where(tpr == 1.)[0][0]]

    return roc_auc, pr_auc, fpr_98

