"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from os.path import join
from torch.utils.data import DataLoader

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


class textReadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, rootdir, names, labels, img_transformer=None):
        # self.data_path = join(dirname(__file__),'kfold')
        self.rootdir = rootdir
        self.names = names
        self.labels = labels
        # self.N = len(self.names)
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()

        img_name = self.rootdir + '/' + self.names[index]

        try:
            image = Image.open(img_name).convert('RGB')
            # image = Image.open(img_name)
        except:
            print(img_name)
            return None
        return self._image_transformer(image), int(self.labels[index])


def dataset_info(txt_labels):
    '''
    file_names:List, labels:List
    '''
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        # file_names.append(row[0])
        file_names.append(' '.join(row[:-1]))
        try:
            # labels.append(int(row[1].replace("\n", "")))
            labels.append(int(row[-1].replace("\n", "")))
        except ValueError as err:
            # print(row[0],row[1])
            print(' '.join(row[:-1]), row[-1])
    return file_names, labels

def reorganize_labels_for_Ganomaly(labels):
    for i in range(len(labels)):
        if labels[i] == 5:
            labels[i] = 0
        else:
            labels[i] = 1

def collate_fn(batch):
    """
    Jasper added to process the empty images
    referece: https://github.com/pytorch/pytorch/issues/1137
    :param batch:
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['cifar10']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=classes[opt.abnormal_class],
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

    elif opt.dataset in ['mnist']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            abn_cls_idx=opt.abnormal_class,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

    elif opt.dataset in ['mnist2']:
        opt.abnormal_class = int(opt.abnormal_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist2_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            nrm_cls_idx=opt.abnormal_class,
            proportion=opt.proportion,
            manualseed=opt.manualseed
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

    elif opt.dataset in ['nexperia_train']:

        # data_dir = os.path.join("F:/100_Work/120_UST_Nexperia/121_data/121.1_raw_data/121.12_Nexperia-2021_04_21/Nexperial_compare_trainingSet/Training_Set/Nex_trainingSet_byClass/All")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021",
                                "Nex_trainingset_shuffle")

        name_train, labels_train = dataset_info(join(data_dir, 'Nex_trainingset_shuffle_train.txt'))
        name_val, labels_val = dataset_info(join(data_dir, 'Nex_trainingset_shuffle_val.txt'))
        name_test, labels_test = dataset_info(join(data_dir, 'Nex_trainingset_shuffle_test.txt'))
        # name_train, labels_train = dataset_info(join(data_dir, 'Nex_trainingSet_byClass_train.txt'))
        # name_val, labels_val = dataset_info(join(data_dir, 'Nex_trainingSet_byClass_val.txt'))
        # name_test, labels_test = dataset_info(join(data_dir, 'Nex_trainingSet_byClass_test.txt'))
        reorganize_labels_for_Ganomaly(labels_train)
        reorganize_labels_for_Ganomaly(labels_val)
        reorganize_labels_for_Ganomaly(labels_test)

        np_name_train = np.array(name_train)
        np_labels_train = np.array(labels_train)
        name_train_0 = np.delete(np_name_train, np_labels_train.nonzero()).tolist()
        labels_train_0 = np.delete(np_labels_train, np_labels_train.nonzero()).tolist()


        transform_train = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=(0.8,1.5)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        transform_val = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])


        # transform = transforms.Compose([transforms.Resize(opt.isize),
        #                                 transforms.CenterCrop(opt.isize),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        train_data = textReadDataset(data_dir, name_train_0, labels_train_0, transform_train)
        valid_data = textReadDataset(data_dir, name_val+name_test, labels_val+labels_test, transform_val)

        train_loader = DataLoader(dataset=train_data, batch_size=opt.batchsize, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=opt.batchsize, num_workers=4, pin_memory=True,
                                  drop_last=False)
        dataloader = {'train' : train_loader, 'test' : valid_loader}
        # return Data(train_loader, valid_loader)
        return dataloader


    elif opt.dataset in ['nexperia_test']:
        month_name = 'Feb2021'
        # data_dir = os.path.join("F:/100_Work/120_UST_Nexperia/121_data/121.1_raw_data/121.12_Nexperia-2021_04_21/Nexperial_compare_trainingSet/Training_Set/Nex_trainingSet_byClass/All")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021", month_name)

        name_test1, labels_test1 = dataset_info(join(data_dir, month_name + '_train_down.txt'))
        name_test2, labels_test2 = dataset_info(join(data_dir, month_name + '_val_down.txt'))
        name_test3, labels_test3 = dataset_info(join(data_dir, month_name + '_test_down.txt'))


        # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # data_dir = os.path.join(BASE_DIR, "..", "..", "..", "..", "..", "share", "from_Nexperia_April2021",
        #                         "Nex_trainingset_shuffle")
        #
        # # name_train, labels_train = dataset_info(join(data_dir, 'Nex_trainingset_shuffle_train.txt'))
        # name_test2, labels_test2 = dataset_info(join(data_dir, 'Nex_trainingset_shuffle_val.txt'))
        # name_test3, labels_test3 = dataset_info(join(data_dir, 'Nex_trainingset_shuffle_test.txt'))
        #
        reorganize_labels_for_Ganomaly(labels_test1)
        reorganize_labels_for_Ganomaly(labels_test2)
        reorganize_labels_for_Ganomaly(labels_test3)
        #
        names = name_test1 + name_test2 + name_test3
        labels = labels_test1 + labels_test2 + labels_test3
        # names = name_test2 + name_test3
        # labels = labels_test2 + labels_test3

        np_names_labels = np.array([names, labels]).transpose()
        np_names_labels_unique = np.unique(np_names_labels, axis=0)
        #### remove the corrupted image just for Feb
        if 'Feb' in month_name:
            corrupted_img = np.where(np_names_labels_unique[:, 0] == 'Pass/20210226_WEPA06311D3A_15_1_377_1.bmp')
            np_names_labels_unique = np.delete(np_names_labels_unique, corrupted_img[0].item(), 0)

        if 'Apr' in month_name:
            corrupted_img = np.where(np_names_labels_unique[:, 0] == 'Pass/20210404_WEPA1220512A_02_1_667_4.bmp')
            np_names_labels_unique = np.delete(np_names_labels_unique, corrupted_img[0].item(), 0)

            corrupted_img = np.where(np_names_labels_unique[:, 0] == 'Pass/20210404_WEPA1220512A_02_2_1229_1.bmp')
            np_names_labels_unique = np.delete(np_names_labels_unique, corrupted_img[0].item(), 0)

            corrupted_img = np.where(np_names_labels_unique[:, 0] == 'Pass/20210404_WEPA1220512A_02_2_209_4.bmp')
            np_names_labels_unique = np.delete(np_names_labels_unique, corrupted_img[0].item(), 0)

        names_list = np_names_labels_unique[:, 0].tolist()
        labels_list = np_names_labels_unique[:, 1].tolist()
        labels_list = list(map(int, labels_list))

        #### only test the abnormal samples.
        # np_name_train = np.array(names_list)
        # np_labels_train = np.array(labels_list)
        # name_train_1 = np_name_train[np_labels_train!=0].tolist()
        # labels_train_1 = np_labels_train[np_labels_train!=0].tolist()

        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])


        # test_data = textReadDataset(data_dir, name_train_1, labels_train_1, transform)
        test_data = textReadDataset(data_dir, names_list, labels_list, transform)

        # test_loader = DataLoader(dataset=test_data, collate_fn= collate_fn, batch_size=opt.batchsize, shuffle=True, num_workers=4,
        #                           pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_data, batch_size=opt.batchsize, num_workers=4, pin_memory=False, shuffle=False,
                                  drop_last=False)

        dataloader = {'train' : None, 'test': test_loader}
        return dataloader
    else:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader

##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist2_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, proportion=0.5,
                               manualseed=-1):
    """ Create mnist 2 anomaly dataset.

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [tensor] -- New training-test images and labels.
    """
    # Seed for deterministic behavior
    if manualseed != -1:
        torch.manual_seed(manualseed)

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

    # Get n percent of the abnormal samples.
    abn_tst_idx = abn_tst_idx[torch.randperm(len(abn_tst_idx))]
    abn_tst_idx = abn_tst_idx[:int(len(abn_tst_idx) * proportion)]


    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl