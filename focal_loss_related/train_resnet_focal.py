# -*- coding: utf-8 -*-
"""
# @file name  : train_resnet_focal.py
# @author     : Jasper
# @date       : 2021-03-01
# @brief      : resnet training on Nexperia dataset
"""
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "9"

from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tools.common_tools import ModelTrainer, show_confMat, plot_line, cal_auc

# ============================ Jasper added import============================
from torchvision import models
import torchvision.datasets as datasets
from tools.focal import FocalLoss
from bisect import bisect_right
# ============================ Jasper added import-END============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", '3')
# device = torch.device(f'cuda:{3}')
class Logger(object):
    """
    save the output of the console into a log file
    """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]





if __name__ == "__main__":


    # train_dir = os.path.join(BASE_DIR, "..", "data_split_small", "train")
    # test_dir = os.path.join(BASE_DIR, "..", "data_split_small", "val")

    # local PC
    # train_dir = 'E:/batch_2/train'
    # test_dir = 'E:/batch_2/test'

    # remote server
    train_dir = os.path.join(BASE_DIR, "..", "..", "..", "Data", "batch_2", "train")
    test_dir = os.path.join(BASE_DIR, "..", "..", "..", "Data", "batch_2", "val")



    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str = time_str + "_S_FL" + "_N-Pre" + "_res34" + "alpha.25"
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    log_dir_train = os.path.join(log_dir, "train")
    log_dir_val = os.path.join(log_dir, "val")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    if not os.path.exists(log_dir_val):
        os.makedirs(log_dir_val)

    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)

    class_names = ('chipping', 'device_flip', 'empty_pocket', 'foreign_material', 'good', 'lead_defect', 'lead_glue', 'marking_defect')
    num_classes = len(class_names)

    MAX_EPOCH = 182
    BATCH_SIZE = 128
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [92, 136]  # divide it by 10 at 32k and 48k iterations

    # ============================ step 1/5 数据 ============================
    # norm_mean = [0.485, 0.456, 0.406]
    # norm_std = [0.229, 0.224, 0.225]


    norm_mean = [0.2203, 0.2203, 0.2203]
    norm_std = [0.1407, 0.1407, 0.1407]


    train_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.1), scale=(0.75, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # Construct Dataset
    train_data = datasets.ImageFolder(train_dir, train_transform)
    valid_data = datasets.ImageFolder(test_dir, valid_transform)

    # Construct Dataloader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=4)

    # ============================ step 2/5 模型 ============================

    ######DenseNet#######
    # resnet_model = models.densenet121(pretrained=True)
    # num_ftrs = resnet_model.classifier.in_features
    # resnet_model.classifier = nn.Linear(num_ftrs, 10)
    # resnet_model.to(device)

    ######ResNet34#######
    resnet_model = models.resnet34()
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, num_classes)
    resnet_model.to(device)

    # ============================ step 3/5 loss function ============================
    # criterion = nn.CrossEntropyLoss()
    # alpha = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25])
    criterion = FocalLoss(class_num=num_classes, alpha=None, gamma=2)
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    optimizer = optim.SGD(resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # Optimizer

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

    # warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
    #             math.cos((epoch - warm_up_epochs) / (MAX_EPOCH - warm_up_epochs) * math.pi) + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=milestones,
                                  gamma=0.1,
                                  warmup_factor=0.1,
                                  warmup_iters=5,
                                  warmup_method="linear",
                                  last_epoch=-1)


# ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    roc_auc_rec = {"train": [], "valid": []}
    pr_auc_rec = {"train": [], "valid": []}
    best_acc, best_epoch, best_auc = 0, 0, 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train, y_true_train, y_outputs_train = ModelTrainer.train(train_loader, resnet_model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid = ModelTrainer.valid(valid_loader, resnet_model, criterion, device)

        roc_auc_train, pr_auc_train, fpr_train = cal_auc(y_true_train, y_outputs_train)
        roc_auc_valid, pr_auc_valid, fpr_valid = cal_auc(y_true_valid, y_outputs_valid)

        # print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train fpr:{:.2%} Valid fpr:{:.2%} LR:{}".format(
        #     epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, fpr_train, fpr_valid, optimizer.param_groups[0]["lr"]))
        print(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train fpr:{:.2%} Valid fpr:{:.2%} Train AUC:{:.2%} Valid AUC:{:.2%} LR:{}".format(
                epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, fpr_train, fpr_valid, roc_auc_train, roc_auc_valid, optimizer.param_groups[0]["lr"]))

        scheduler.step()  # Update learning rate

        # Plot
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        roc_auc_rec["train"].append(roc_auc_train), roc_auc_rec["valid"].append(roc_auc_valid)
        pr_auc_rec["train"].append(pr_auc_train), pr_auc_rec["valid"].append(pr_auc_valid)

        np.save(os.path.join(log_dir_train, 'loss_rec.npy'), loss_rec["train"])
        np.save(os.path.join(log_dir_train, 'acc_rec.npy'), acc_rec["train"])
        np.save(os.path.join(log_dir_train, 'roc_auc_rec.npy'), roc_auc_rec["train"])
        np.save(os.path.join(log_dir_train, 'pr_auc_rec.npy'), pr_auc_rec["train"])

        np.save(os.path.join(log_dir_val, 'loss_rec.npy'), loss_rec["valid"])
        np.save(os.path.join(log_dir_val, 'acc_rec.npy'), acc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'roc_auc_rec.npy'), roc_auc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'pr_auc_rec.npy'), pr_auc_rec["valid"])

        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        plot_line(plt_x, roc_auc_rec["train"], plt_x, roc_auc_rec["valid"], mode="roc_auc", out_dir=log_dir)
        plot_line(plt_x, pr_auc_rec["train"], plt_x, pr_auc_rec["valid"], mode="pr_auc", out_dir=log_dir)


        if epoch > (MAX_EPOCH / 2) and best_auc < roc_auc_valid:
            best_auc = roc_auc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": resnet_model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_auc": best_auc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best auc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_auc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f








