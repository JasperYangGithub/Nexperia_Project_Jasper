from img_similarity import runtwoImageSimilaryFun
import os
from PIL import Image
import shutil
import time
import numpy as np
import os
import imghdr
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def similar(path1, path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    w1 = img1.size[0]  # 图片的宽
    h1 = img2.size[1]  # 图片的高
    w2 = img2.size[0]  # 图片的宽
    h2 = img2.size[1]  # 图片的高
    w_err = abs(w1 - w2) / w1
    h_err = abs(h1 - h2) / h1
    if w_err > 0.1 or h_err > 0.1:
        return 0
    else:
        phash, color_hist = runtwoImageSimilaryFun(path1, path2)
        if phash >= 20 or color_hist <= 0.4:
            # phash_str = str(phash)
            # color_hist_str = str(color_hist)
            # plt.subplot(121)
            # plt.imshow(img1)
            # plt.subplot(122)
            # plt.imshow(img2)
            # plt.title(phash_str+" "+color_hist_str)
            # plt.show()
            return 1
        else:
            return 0



path = os.path.join(BASE_DIR, "../..", "..", "..", "Data", "batch_2", "val", "good")
empty_pocket_path = path + '/' + "WEL93407711A_03-JVW-ITISA13-1_358_2.bmp"
result_imgdirs_path = os.path.join(BASE_DIR, "../..", "..", "..", "Data", "batch_2_noneRepeat", "val", "good")


folder_path = path
new_folder_path = result_imgdirs_path
os.makedirs(new_folder_path)


imglist = os.listdir(folder_path)


imglist.sort()

time_start = time.time()

for i, item1 in enumerate(imglist):
    if item1 == '0':
        continue
    path1 = folder_path + '/' + item1
    check1 = imghdr.what(path1)
    if check1 == None:
        imglist[i] = '0'
        continue

    t = similar(path1, empty_pocket_path)
    if t:
        # 将判断为相似的图片在trans_list中的名字置‘0’，代表不需要复制
        imglist[i] = '0'

imglist = list(set(imglist))
if '0' in imglist:
    imglist.remove('0')

time_end = time.time()
time_c = time_end - time_start
print('similarity judgement list time cost {}s'.format(time_c))

time_start = time.time()
# 移动图片
for item3 in imglist:
    ori_img_path = folder_path + '/' + item3
    new_img_path = new_folder_path + '/' + item3
    shutil.copy(ori_img_path, new_img_path)

time_end = time.time()
time_c = time_end - time_start  # 运行所花时间
print('move image time cost {}s'.format(time_c))


