import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import Dataset
import os
import os.path
import logging
import sys
import torch
import io
import scipy.io as matio
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import ipdb
from sklearn.model_selection import KFold
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

random.seed(0)

def map_list(lst):
    unique_values = list(set(lst))  # 找出列表中的唯一值
    unique_values.sort()  # 对唯一值进行排序
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    mapped_list = [value_to_index[value] for value in lst]

    return mapped_list



def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def default_loader(image_path):
    return Image.open(image_path).convert('RGB')    

    

class CIFAR10_truncated(torch.utils.data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, split=None):
        """
        Args:
            root (str): 数据集存储路径
            dataidxs (list, optional): 索引列表，用于筛选数据
            train (bool): 是否加载训练集
            transform (callable, optional): 图像转换
            target_transform (callable, optional): 标签转换
            download (bool): 是否下载数据集
            split (str, optional): 选择 'test_base' 或 'test_new'，仅在 train=False 时有效
        """
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split = split

        self.data, self.targets, self.m_targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, train=self.train, download=self.download)

        # 获取数据和标签
        if hasattr(cifar_dataobj, 'data'):
            data, targets = cifar_dataobj.data, np.array(cifar_dataobj.targets)
        else:
            data, targets = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels) if self.train else cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)

        # 如果是训练集，仅保留类别 0-4
        if self.train:
            mask = np.isin(targets, [0, 1, 2, 3, 4])
        else:
            if self.split == "test_base":
                # test_base: 仅包含类别 0-4
                mask = np.isin(targets, [0, 1, 2, 3, 4])
            elif self.split == "test_new":
                # test_new: 仅包含类别 5-9
                mask = np.isin(targets, [5, 6, 7, 8, 9])
            else:
                raise ValueError("Invalid split value. Choose 'test_base' or 'test_new'.")

        # 过滤数据
        data, targets, m_targets = data[mask], targets[mask], targets[mask]

        # 如果提供了 dataidxs，则进一步筛选数据
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]
            m_targets = map_list(targets)

        return data, targets, m_targets

    def __getitem__(self, index):
        img, target, m_target = self.data[index], self.targets[index], self.m_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, m_target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, split=None):
        """
        Args:
            root (str): 数据集存储路径
            train (bool): 是否加载训练集
            transform (callable, optional): 图像转换
            target_transform (callable, optional): 标签转换
            download (bool): 是否下载数据集
            split (str, optional): 选择 'test_base' 或 'test_new'，仅在 train=False 时有效
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataidxs = dataidxs
        self.split = split

        self.data, self.targets, self.m_targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_data = CIFAR100(self.root, train=self.train, download=self.download)

        # 获取数据和标签
        data, targets = cifar_data.data, np.array(cifar_data.targets)

        if self.train:
            # 训练集：仅包含类别 0-49
            mask = np.isin(targets, np.arange(50))
        else:
            if self.split == "test_base":
                # test_base: 仅包含类别 0-49
                mask = np.isin(targets, np.arange(50))
            elif self.split == "test_new":
                # test_new: 仅包含类别 50-99
                mask = np.isin(targets, np.arange(50, 100))
            else:
                raise ValueError("Invalid split value. Choose 'test_base' or 'test_new'.")

        # 过滤数据
        data, targets, m_targets = data[mask], targets[mask], targets[mask]
        # 如果提供了 dataidxs，则进一步筛选数据
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]
            m_targets = map_list(targets)

        return data, targets, m_targets

    def __getitem__(self, index):
        img, target, m_target = self.data[index], self.targets[index], self.m_targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, m_target

    def __len__(self):
        return len(self.data)
    
    

class TinyImageNet_load(Dataset):
    def __init__(self, root, dataidxs=None, train=True, split=None, transform=None):
        self.train = train
        self.split = split
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.dataidxs = dataidxs

        if self.train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.train, self.split)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        if self.dataidxs is not None:
            self.samples = self.images[self.dataidxs]
        else:
            self.samples = self.images
       
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        print(self.samples.shape)
        self.data = self.samples[:, 0]
        self.labels = self.samples[:, 1]
        self.m_labels = map_list(self.labels)
    
    def _create_class_idx_dict_train(self):
        classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        classes = sorted(classes)

        num_images = sum(len(files) for _, _, files in os.walk(self.train_dir) if any(f.endswith(".JPEG") for f in files))
        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()

        with open(val_annotations_file, 'r') as fo:
            for data in fo.readlines():
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(self.val_img_to_class)
        classes = sorted(list(set_of_classes))

        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, train=True, split='train'):
        self.images = []
        if train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        target_class = self.class_to_tgt_idx[tgt] if train else self.class_to_tgt_idx[self.val_img_to_class[fname]]

                        if train:  # 训练集
                            if target_class < 100:  # Base 类
                                self.images.append((path, target_class))

                        else:  # 测试集
                            if split == 'test_base':  # 测试集，前100类作为 base 类
                                if target_class < 100:  # Base 类
                                    self.images.append((path, target_class))
                            elif split == 'test_new':  # 测试集，后100类作为 novel 类
                                if target_class >= 100 and target_class < 200:  # Novel 类
                                    self.images.append((path, target_class))

        self.images = np.array(self.images)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tgt, m_tgt = self.data[idx], self.labels[idx], self.m_labels[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, int(tgt), int(m_tgt)


class Food101_truncated(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, loader=default_loader, mode=None):
        image_path = '/Food101_Image/food-101/images/'
        data_path = '/Food101_Text/'
        
        # 读取图像路径和标签
        if mode == 'train':
            with io.open(data_path + 'train_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            with io.open(data_path + 'train_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1]
        elif mode == 'test':
            with io.open(data_path + 'test_images.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            with io.open(data_path + 'test_labels.txt', encoding='utf-8') as file:
                labels = file.read().split('\n')[:-1]
        else:
            raise ValueError("Please set mode to 'train' or 'test'")
        
        labels = np.array(labels, dtype=int)
        
        # 进行类别划分
        base_classes = set(range(50))  # 前50个类
        base_indices = [i for i, label in enumerate(labels) if label in base_classes]
        new_indices = [i for i, label in enumerate(labels) if label not in base_classes]
        
        if mode == 'train':
            selected_indices = base_indices  # 训练集中仅包含 base 类
            if dataidxs is not None:
                selected_indices = np.intersect1d(selected_indices, dataidxs)   
                
        elif mode == 'test':
            if split == 'test_base':
                selected_indices = base_indices  # 测试集中 base 类
            elif split == 'test_new':
                selected_indices = new_indices  # 测试集中 new 类
        
        self.image_path = image_path
        self.path_to_images = np.array(path_to_images)[selected_indices]
        self.labels = labels[selected_indices]
        self.m_labels = map_list(self.labels)
        self.transform = transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        path = self.path_to_images[index]
        img = self.loader(self.image_path + path + '.jpg')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        m_label = self.m_labels[index]
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)
    
    def get_test_split(self):
        """ 获取测试集中的 base 类和 new 类数据集 """
        return (self.test_base_images, self.test_base_labels), (self.test_new_images, self.test_new_labels)

        
 

class Vireo172_truncated(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, loader=default_loader, mode=None):
        image_path = '/Vireo172_Image/ready_chinese_food/'
        data_path = '/Vireo172_Text/SplitAndIngreLabel/'
        
        # 读取图像路径和标签
        if mode == 'train':
            with io.open(data_path + 'TR.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'train_label.mat')['train_label'][0] - 1
        elif mode == 'test':
            with io.open(data_path + 'TE.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'test_label.mat')['test_label'][0] - 1
        else:
            raise ValueError("Please set mode to 'train' or 'test'")
        
        labels = np.array(labels, dtype=int)
        
        # 进行类别划分
        base_classes = set(range(50))  # 前50个类
        base_indices = [i for i, label in enumerate(labels) if label in base_classes]
        new_indices = [i for i, label in enumerate(labels) if label not in base_classes]
        
        if mode == 'train':
            selected_indices = base_indices  # 训练集中仅包含 base 类
            if dataidxs is not None:
                selected_indices = np.intersect1d(selected_indices, dataidxs)   
                
        elif mode == 'test_base':
            selected_indices = base_indices  # 测试集中 base 类
        elif mode == 'test_new':
            selected_indices = new_indices  # 测试集中 new 类
        
        if dataidxs is not None:
            selected_indices = np.intersect1d(selected_indices, dataidxs)
        
        self.image_path = image_path
        self.path_to_images = np.array(path_to_images)[selected_indices]
        self.labels = labels[selected_indices]
        self.m_labels = map_list(self.labels)
        self.transform = transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        path = self.path_to_images[index]
        img = self.loader(self.image_path + path)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        m_label = self.m_labels[index]
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)
    
    def get_test_split(self):
        """ 获取测试集中的 base 类和 new 类数据集 """
        return (self.test_base_images, self.test_base_labels), (self.test_new_images, self.test_new_labels)



    
       

    
class Caltech101(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/Caltech101/train_base.csv'
        elif mode == 'test_base':
            csv_dir = '../datasets/Caltech101/test_base.csv'
        elif mode == 'test_new':
            csv_dir = '../datasets/Caltech101/test_new.csv'            
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]
        self.labels = self.csv_info[:, 1]
        self.transform = transform
        self.m_labels = self.csv_info[:, 1]
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    

        
class Flowers102(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
            
        if mode == 'train':
            csv_dir = '../datasets/Flowers102/train_base.csv'
        elif mode == 'test_base':
            csv_dir = '../datasets/Flowers102/test_base.csv'
        elif mode == 'test_new':
            csv_dir = '../datasets/Flowers102/test_new.csv'            
            
            
        self.csv_info = pd.read_csv(csv_dir).values
#         print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1] - 1
        self.m_labels = self.csv_info[:, 1] - 1
 
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    
        
    
class Air100(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        if mode == 'train':
            csv_dir = '../datasets/FGVC_Aircraft/train_base.csv'
        elif mode == 'test_base':
            csv_dir = '../datasets/FGVC_Aircraft/test_base.csv'
        elif mode == 'test_new':
            csv_dir = '../datasets/FGVC_Aircraft/test_new.csv'
            
            
        self.csv_info = pd.read_csv(csv_dir).values
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    
    
    
    
class Cars196(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/Stanford_Cars/train_base.csv'
        elif mode == 'test_base':
            csv_dir = '../datasets/Stanford_Cars/test_base.csv'
        elif mode == 'test_new':
            csv_dir = '../datasets/Stanford_Cars/test_new.csv'            
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
 
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    
          
        
class UCF101(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/UCF101/train_base.csv'
        elif mode == 'test_base':
            csv_dir = '../datasets/UCF101/test_base.csv'
        elif mode == 'test_new':
            csv_dir = '../datasets/UCF101/test_new.csv'            
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    
                


class DTD(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/DTD/train_base.csv'
        elif mode == 'test_base':
            csv_dir = '../datasets/DTD/test_base.csv'
        elif mode == 'test_new':
            csv_dir = '../datasets/DTD/test_new.csv'
            
        self.csv_info = pd.read_csv(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1]
        self.m_labels = self.csv_info[:, 1]
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)    
                
class Pets(torch.utils.data.Dataset):
    def __init__(self, dataidxs=None, transform=None, mode='train'):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        
        if mode == 'train':
            csv_dir = '../datasets/OxfordPets/train_base.xlsx'
        elif mode == 'test_base':
            csv_dir = '../datasets/OxfordPets/test_base.xlsx'
        elif mode == 'test_new':
            csv_dir = '../datasets/OxfordPets/test_new.xlsx'
            
        self.csv_info = pd.read_excel(csv_dir).values
        print(self.csv_info)
        self.path_to_images = self.csv_info[:, 0]

        self.labels = self.csv_info[:, 1] - 1
        self.m_labels = self.csv_info[:, 1] - 1
        self.transform = transform
        
        if mode == 'train' and dataidxs != None:
            self.path_to_images = self.path_to_images[dataidxs]
            self.labels = self.labels[dataidxs]
            self.m_labels = map_list(self.labels)
            print('mode:', mode, 'len(path_to_images):', len(self.path_to_images))
        

    def __getitem__(self, index):
        img_dir, label, m_label = self.path_to_images[index], self.labels[index], self.m_labels[index]
        img_dir = '../datasets/OxfordPets/' + img_dir + '.jpg'
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
#             print(img.shape)
        return img, label, m_label

    def __len__(self):
        return len(self.path_to_images)   


class Domain_G(torch.utils.data.Dataset):
    def __init__(self, dataset, domain='amazon', transform=None, dataidxs=None):
        """
        初始化Office数据集类，加载数据集路径和域、类别信息
        :param dataset_path: 数据集的根目录路径
        """
        self.dataset = dataset
        if dataset == "domain_office":
            self.dataset_path = '../datasets/Office_Home/'
        elif dataset == "domain_vlcs":
            self.dataset_path = '../datasets/VLCS/'
        elif dataset == "domain_pacs":
            self.dataset_path = '../datasets/PACS/'
        elif dataset == 'domain_net':
            self.dataset_path = '../datasets/DomainNet/'
        self.domain = domain  # 存储域名称
        self.dataidxs = dataidxs
        self.classes = []  # 存储类别名称
        self.class_to_idx = {}  # 类别名到标签的映射
        self.data = []  # 存储图像数据
        self.labels = []  # 存储对应标签
        self.transform = transform
        # 加载数据集
        self._load_data()

    def _load_data(self):
        """
        加载数据集的图像和标签，保存到数据列表中
        """
        # 获取所有域的文件夹（即子目录）,先排序
        
        domain_folder = os.path.join(self.dataset_path, self.domain)
        
        # 遍历域中的所有类别
        if self.dataset == "domain_office":
            class_names = sorted([x for x in os.listdir(domain_folder) if x != '.ipynb_checkpoints'])
        elif self.dataset == 'domain_net':
            class_names = sorted([x for x in os.listdir(domain_folder) if x != '.ipynb_checkpoints'])[:20]
        # print(class_names)
        for class_name in class_names:
            class_folder = os.path.join(domain_folder, class_name)
            
            if os.path.isdir(class_folder):
                # 为每个类别分配标签
                self.classes.append(class_name)
                self.class_to_idx[class_name] = len(self.classes) - 1
                
                # 遍历类别中的所有图像文件
                for image_name in os.listdir(class_folder):
                    image_path = os.path.join(class_folder, image_name)
                    
                    if image_path.endswith(('.jpg', '.png', '.jpeg')):  # 只考虑图像文件
                        self.data.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
                        
        indices = list(range(len(self.data)))
        random.seed(2023)
        random.shuffle(indices)
        
        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        # ipdb.set_trace()
        # 将数据划分为 k 份
        if self.dataset == 'domain_net':
            kk = 5
        else:
            kk = 5
        kf = KFold(n_splits=kk, shuffle=True, random_state=42)  # KFold 进行划分
        self.folds = []  # 存储 k 份数据
    
        for _, test_idx in kf.split(self.data):
            fold_data = [self.data[i] for i in test_idx]
            fold_labels = [self.labels[i] for i in test_idx]
            m_labels = map_list(fold_labels)
            self.folds.append((fold_data, fold_labels, m_labels))  # 保存每一份

    def __getitem__(self, index):
        """
        获取指定索引的图像、类别标签和域标签
        :param index: 索引
        :return: 图像、类别标签和域标签
        """
        image_path = self.data[index]
        label = self.labels[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]

        return img, label, label        

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        自定义数据集
        :param data: 图像路径列表
        :param labels: 对应的标签列表
        :param transform: 图像变换（可选）
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # 读取图像
        image = Image.open(img_path).convert("RGB")  # 确保是RGB格式
        
        if self.transform:
            image = self.transform(image)  # 进行变换
        
        return image, label, label