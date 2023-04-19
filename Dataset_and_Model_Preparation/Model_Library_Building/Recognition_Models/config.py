import torch
import timm
import torchvision.transforms as T
import os


# 参数全放在这里
class config:

    # 数据集部分
    # train_root = "/root/autodl-tmp/dachuang/数据集/极验/字符/"
    train_root = "/root/autodl-tmp/dachuang/adv_training_reco/renmin_char"
    test_root = '/root/autodl-tmp/dachuang/数据集/test_recognition/renmin/test'
    # train_val_split = 0.8
    train_val_split = 1.0
    input_shape = [3, 128, 128]
    train_transform = T.Compose([
        # T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(), # totensor即为归一化
    ])
    test_transform = T.Compose([
        # T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(), # totensor即为归一化
    ])
    train_batch_size  = 64
    test_batch_size  = 64
    
    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 4  # data loader

    class_to_dix_path = '/root/autodl-tmp/dachuang/数据集/极验/class_to_idx.txt'

    # class_to_idx = dict()
    # with open(class_to_dix_path, 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         idx, char  = line.split(' ')
    #         char = char.strip('\n')
    #         class_to_idx[int(idx)] = char
    # f.close()

    # 模型部分
    model_name = 'resnet50' # 'inception_resnet_v2' 'inception-v3' 'vgg16' 'resnet50'
    model_pretrained = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_path = '/root/autodl-tmp/dachuang/models/renmin/Res50_adv/'
    test_model = '/root/autodl-tmp/dachuang/models/renmin/Res50/final_model.pth'
    finetune_model = '/root/autodl-tmp/dachuang/models/renmin/Res50/final_model.pth'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # 优化器部分
    epochs = 50
    optimizer = 'sgd'  # ['sgd', 'adam']
    learning_rate = 1e-3
    
    
    