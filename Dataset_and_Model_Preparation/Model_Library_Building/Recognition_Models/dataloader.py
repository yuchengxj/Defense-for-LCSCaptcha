
from tkinter import X
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch
import os
from PIL import Image
from torchvision.io import read_image
# 使用config中的参数load数据
from config import config as conf

def load_data(conf, training=True):
    if training:
        data_src = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
        # 训练时的标签仍然是类别
        data = Necaptcha_char_dataset(data_src, transform=transform)


        # class_idx = data.class_to_idx
        # with open(conf.class_to_dix_path, 'a', encoding='utf-8') as f:
        #     for class_name in class_idx.keys():
        #         idx = class_idx[class_name]
        #         f.write(str(class_name)+' '+str(idx)+'\n')
        #         print(1)
        # f.close()
        

        train_len = int(conf.train_val_split*len(data))
        val_len = len(data) - int(conf.train_val_split*len(data))
        train_data, val_data = random_split(data, [train_len, val_len], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)
        # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=conf.pin_memory,
        #                     num_workers=conf.num_workers)

        val_loader = None
        class_num = data.num_class
        return train_loader, val_loader, class_num

    else:
        data_src = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size
        data = Necaptcha_char_dataset(data_src, transform=transform)


        class_num = data.num_class
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)

        return data_loader, class_num




class test_char_dataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        
        imgs = []
        
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                imgs.append((img_path, conf.class_to_idx[class_name]))


        self.imgs = imgs
        self.num_class = len(conf.class_to_idx.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.class_to_idx = conf.class_to_idx

    def __getitem__(self, index):
        x, y = self.imgs[index]
        img = Image.open(x).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.imgs)

class Necaptcha_char_dataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        
        imgs = []
        
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)

                imgs.append((img_path, int(class_name)))


        self.imgs = imgs
        self.num_class = len(os.listdir(path))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = self.imgs[index]
        img = Image.open(x).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.imgs)