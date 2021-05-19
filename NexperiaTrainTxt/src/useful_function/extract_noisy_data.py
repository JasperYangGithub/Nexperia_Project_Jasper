import os
from os.path import *
import shutil
from tqdm import tqdm
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from prettytable import PrettyTable
from tools.common_tools import Logger
import sys


def acquire_fpr_threshold(result_file):

    tpr_point_list = [0.980, 0.991, 0.993, 0.995, 0.997, 0.999, 1]
    fpr_list = []
    threshold_list = []
    fpr_list2 = []

    a_file = open(result_file, "rb")
    output = pickle.load(a_file)
    fpr = output['fpr']['fpr_all']
    tpr = output['fpr']['tpr_all']
    thresholds = output['fpr']['thresholds']

    for tpr_point in tpr_point_list:
        threshold_list.append(thresholds[[np.where(tpr >= tpr_point)[0][0]]].item())

    # for name in fpr_name_list:
    #     fpr_list1.append(output['fpr'][name])

    for tpr_point in tpr_point_list:
        fpr_list.append(fpr[[np.where(tpr >= tpr_point)[0][0]]].item())

    return threshold_list, fpr_list

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
fpr_name_dict = {'fpr_980':0, 'fpr_991':1, 'fpr_993':2, 'fpr_995':3, 'fpr_997':4, 'fpr_999':5, 'fpr_1':6}


pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    #++++++++ hyper parameters ++++++++#
    test_res_dir_name= '05-17_20-11_test_Res50_FL_Jan2021_down_img256'
    bad_score_file_name = '_best2_badScore.xlsx'
    result_file_name = '_best2_results.pkl'
    # data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021", "Nex_trainingset")
    data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021",
                            "Jan2021")

    table = PrettyTable(['TPR', 'tpr_980', 'tpr_991', 'tpr_993', 'tpr_995', 'tpr_997', 'tpr_999', 'tpr_1'])

    target_dir_root = os.path.join(BASE_DIR, "..", "..", "..", "..", "Data", "noisy_data")

    target_dir = os.path.join(target_dir_root, test_res_dir_name)
    if not exists(target_dir):
        os.mkdir(target_dir)
    test_results_dir = os.path.join(BASE_DIR, "../..", "results", "model_test_results", test_res_dir_name)
    result_file = os.path.join(test_results_dir, result_file_name)

    sys.stdout = Logger(os.path.join(target_dir, 'log.txt'), sys.stdout)


    threshold_list, fpr_list = acquire_fpr_threshold(result_file)
    threshold980 = round(threshold_list[fpr_name_dict['fpr_980']],6)
    threshold995 = round(threshold_list[fpr_name_dict['fpr_995']], 6)
    threshold1 = round(threshold_list[fpr_name_dict['fpr_1']], 6)



    best1_file = os.path.join(test_results_dir, bad_score_file_name)
    df = pd.read_excel(best1_file)
    label_groups = df.groupby('Label')


    # ============================ Plot bad_score distribution ============================
    plt.figure(figsize=(10, 5))
    # threshold980

    plt_x_start = 0
    for i in range(9):
        # if i!=4:
        single_group = label_groups.get_group(i+1)
        single_group_sort = single_group.sort_values(by='Bad_score')
        plt_y = single_group_sort.Bad_score
        plt_x_end = plt_x_start + len(plt_y)
        plt_x = np.arange(plt_x_start,plt_x_end)
        if len(plt_y)<=5:
            lw = 80
        elif len(plt_y)<=10:
            lw=30
        else:
            lw = 1
        # plt.plot(plt_x, plt_y, label=class_l[i] + "_" + str(len(plt_y)), linewidth=lw)
        plt.scatter(plt_x, plt_y, s=lw, label=class_l[i] + "_" + str(len(plt_y)))

        # plt.show()
        plt_x_start = plt_x_end

    plt.plot(np.arange(0,plt_x_end), np.ones(plt_x_end)*threshold980, label='Thr_980_'+str(threshold980), linewidth=1)
    plt.plot(np.arange(0, plt_x_end), np.ones(plt_x_end) * threshold1, label='Thr_1_' + str(threshold1), linewidth=1)
    plt.plot(np.arange(0, plt_x_end), np.ones(plt_x_end) * threshold995, label='Thr_995_' + str(threshold995), linewidth=1)

    plt.ylabel("bad_score")
    plt.xlabel('data_points')
    location = 'upper center'
    # location = 'lower right'
    plt.title(test_res_dir_name[23:]+bad_score_file_name[:-14])
    plt.legend(loc=location)

    plt.savefig(os.path.join(target_dir, 'bad_score.svg'), format='svg', dpi=600)
    plt.show()
    plt.close()


    # ============================ acquire the possible noisy pass and defect data ============================
    defect_noise_list = []
    for i in range(9):
        single_group = label_groups.get_group(i+1)
        # single_group_sort = single_group.sort_values(by='Bad_score')
        if i==4:
            pass_noise = single_group[single_group.Bad_score>=0.5]
        else:
            defect_noise_list.append(single_group[single_group.Bad_score<=threshold980])
    defect_noise = pd.concat(defect_noise_list)

    print('total pass_noise images are {}'.format(len(pass_noise)))
    print('total defect_noise images are {}'.format(len(defect_noise)))


    noisy_data = defect_noise.append(pass_noise)
    # ============================ created folders to save the possible noisy data ============================
    noisy_list = noisy_data.Label.unique()
    for CLASS in noisy_list:
        if not exists(join(target_dir, class_l[CLASS-1])):
            os.mkdir(join(target_dir,class_l[CLASS-1]))


    # ============================ copy the possible noisy data to created folders ============================
    os.chdir(data_dir)
    for i in range(len(noisy_data)):
        CLASS_name = class_l[noisy_data.Label.iloc[i]-1]
        shutil.copy(noisy_data.Path.iloc[i], join(target_dir,CLASS_name))

    threshold_list = [round(num, 4) for num in threshold_list]
    fpr_list = [round(num, 5) for num in fpr_list]
    threshold_list.insert(0, 'Thr')
    fpr_list.insert(0, 'FPR')
    table.add_row(fpr_list)
    table.add_row(threshold_list)

    print(table)

    f = open(os.path.join(target_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f