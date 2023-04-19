
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
import dataloader as dataloader
import timm
from torchsummary import summary
from torch.utils.data import random_split

def train_loop(dataloader, model, loss_fn, optimizer, i):
    loss = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    correct = 0
    for batch, (X, y) in loop:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        acc = correct / (conf.train_batch_size * (batch+1))
        loop.set_description(f'Epoch {i}/{conf.epochs}')
        loop.set_postfix(loss=loss.item(), acc=acc)
    return loss.item(), acc

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
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')
    return test_loss, correct


if __name__ == '__main__':

    data_loader_train, data_loader_val, num_class = dataloader.load_data(conf=conf)
    device = conf.device
    model = nn.DataParallel(timm.create_model(conf.model_name, num_classes=num_class, pretrained=False))
    summary(model, (3, 128, 128))

    loss_fn = nn.CrossEntropyLoss()
    if conf.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=conf.learning_rate)
    
    if conf.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    best_acc = 0
    acc_train_history = []
    acc_val_history = []
    loss_train_history = []
    loss_val_history = []
    for t in range(conf.epochs):
        print(f'Epoch {t+1}\n------------------------------')
        loss_train, acc_train = train_loop(data_loader_train, model, loss_fn, optimizer, t+1)

        loss_val, acc_val = test_loop(data_loader_val, model, loss_fn)
        if acc_val > best_acc_val:
            print(f'Save model with best validation accuracy:{acc_val}')
            best_acc_val = acc_val
            torch.save(model, conf.model_save_path+'best_model.pth')
        acc_train_history.append(acc_train)
        acc_val_history.append(acc_val)
        loss_train_history.append(loss_train)
        loss_val_history.append(loss_val)
    
    torch.save(model, conf.model_save_path+'final_model.pth')    

    fig = plt.figure(1)
    plt.plot(acc_train_history)
    plt.plot(acc_val_history)
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(conf.model_save_path+'acc.jpg')
    plt.close(1)

    fig = plt.figure(2)
    plt.plot(loss_train_history)
    plt.plot(loss_val_history)
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(conf.model_save_path+'loss.jpg')
    plt.close(2)