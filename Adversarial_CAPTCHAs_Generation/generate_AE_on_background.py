import os
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config as conf
from dataloader import Dataset, dataset_collate
from SVRE_MI_FGSM import svre_mi_fgsm
import sys
sys.path.append(r"./Dataset_and_Model_Preparation/Model_Library_Building")
from frcnn.nets.frcnn import FasterRCNN
from frcnn.nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)

from yolov5.nets.yolo import YoloBody

from yolov5.nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from yolov5.utils.utils import get_anchors, get_classes

from ssd.utils.anchors import get_anchors as get_anchors_ssd
from ssd.nets.ssd import SSD300
from ssd.nets.ssd_training import MultiboxLoss

def generate_loop(gen, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd):
    loop = tqdm(enumerate(gen), total=min(len(gen),conf.detect_break_point))
    for iter, batch in loop:
        if iter >= conf.detect_break_point:
            break
        images, boxes, locations, paths = batch[0], batch[1], batch[2], batch[3]
        images = images.to(conf.device)


        X_adv = svre_mi_fgsm(images, boxes, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, conf.detect_max_eps, conf.detect_num_iter, conf.detect_momentum, conf.detect_m_svrg)
        n = images.size(0)
        for i in range(n):
            image, location, path = X_adv[i], locations[i], paths[i]
            new_path = path.replace('JPEGImages', 'JPEGImages_svre_adv_'+str(conf.detect_max_eps))
            new_dir = os.path.dirname(new_path)
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            save_image(image, new_path, location)

        

def save_image(x, path, location):
    image = x.cpu()
    img = image.detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255.0
    img = Image.fromarray(img.astype(np.uint8))
    img = img.crop(location)
    old_path = path.replace('JPEGImages_svre_adv_'+str(conf.detect_max_eps), 'JPEGImages')
    cap = Image.open(old_path).convert('RGB')
    w, h = cap.size
    img = img.resize((w, h))
    img.save(path)


if __name__ == "__main__":
    class_names, num_classes = get_classes(conf.detect_classes_path)

    ## 定义yolo模型
    anchors_yolo, num_anchors_yolo = get_anchors(conf.detect_anchors_path_yolo)
    model_yolo = YoloBody(conf.detect_anchors_mask_yolo, num_classes, conf.detect_phi, conf.detect_backbone_yolo, pretrained=conf.detect_pretrained, input_shape=conf.detect_input_shape)
    model_yolo.load_state_dict(torch.load(conf.detect_model_path_yolo, map_location=conf.device))
    model_yolo = model_yolo.to(conf.device)
    loss_yolo    = YOLOLoss(anchors_yolo, num_classes, conf.detect_input_shape, True, conf.detect_anchors_mask_yolo, 0)
    model_yolo = model_yolo.eval()
    
    ## 定义frcnn模型
    model_frcnn = FasterRCNN(num_classes, anchor_scales=conf.detect_anchors_size_frcnn, backbone=conf.detect_backbone_frcnn, pretrained=conf.detect_pretrained)
    model_frcnn.load_state_dict(torch.load(conf.detect_model_path_frcnn, map_location = conf.device))
    model_frcnn = model_frcnn.eval()
    model_frcnn = torch.nn.DataParallel(model_frcnn).cuda()
    optimizer = optim.Adam(model_frcnn.parameters())
    train_util_frcnn = FasterRCNNTrainer(model_frcnn, optimizer)

    ## 定义ssd模型
    anchors_ssd = get_anchors_ssd(conf.detect_input_shape, conf.detect_anchors_size_ssd, conf.detect_backbone_ssd)
    model_ssd = SSD300(num_classes+1, conf.detect_backbone_ssd, conf.detect_pretrained)
    model_ssd.load_state_dict(torch.load(conf.detect_model_path_ssd, map_location = conf.device))
    model_ssd = model_ssd.eval()
    model_ssd = torch.nn.DataParallel(model_ssd)
    cudnn.benchmark = True
    model_ssd = model_ssd.cuda()
    loss_ssd       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)

    with open(conf.detect_origin_annotation_path) as f:
        lines = f.readlines()
    num_origin = len(lines)

    ## 定义dataloader

    origin_dataset = Dataset(lines, conf.detect_input_shape, num_classes, anchors_yolo, conf.detect_anchors_mask_yolo, anchors_ssd, epoch_length=300, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

    gen = DataLoader(origin_dataset, shuffle=False, batch_size=conf.detect_batch_size, num_workers=conf.detect_num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=None)
    generate_loop(gen, train_util_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd)
    # generate_loop(gen, frcnn, yolo, ssd)