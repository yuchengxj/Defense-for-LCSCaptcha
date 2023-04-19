from importlib.util import module_for_loader
import os
from pickletools import optimize
import shutil
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from config import config as conf
import matplotlib.pyplot as plt
import dataloader
import timm
from torchsummary import summary
from torch.utils.data import random_split
from PIL import Image


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f'Accuracy: {correct}, Avg loss: {test_loss:>8f}\n')
    return test_loss, correct

def predict_img(model, img_path):
    img = Image.open(img_path)
    img_tensor = conf.test_transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(conf.device)
    out = model(img_tensor)
    y = out.argmax(1).item() 
    return y

if __name__ == '__main__':
    data_loader_test, num_class = dataloader.load_data(conf=conf, training=False)
    device = conf.device
    model = torch.load(conf.test_model)
    summary(model, (3, 128, 128))
    print(model)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss, acc = test_loop(data_loader_test, model, loss_fn)
    # y = predict_img(model, '/root/autodl-tmp/dachuang/数据集/易盾/test/1946/1425_1.png')
    # char_name = conf.class_to_idx[y]
    # print(char_name)