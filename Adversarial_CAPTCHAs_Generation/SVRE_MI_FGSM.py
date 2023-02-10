import torch
import numpy as np
import random

from config import config as conf

def clip_by_value(x, x_min, x_max):
    x = x * (x <= x_max).type(torch.FloatTensor).to(conf.device) + x_max * (x > x_max).type(torch.FloatTensor).to(conf.device)
    x = x * (x >= x_min).type(torch.FloatTensor).to(conf.device) + x_min * (x < x_min).type(torch.FloatTensor).to(conf.device)
    return x

def grad_ensemble(model_frcnn, model_yolo, model_ssd, imgs, y, loss_yolo, loss_ssd, w_frcnn, w_yolo, w_ssd, model_name):
    grad = None
    x_orgin = imgs.detach().type(torch.FloatTensor).to(conf.device)
    x_orgin.requires_grad = True
    if model_name == 'frcnn':
        bboxes_frcnn, labels_frcnn = y[0], y[1]
        rpn_loc, rpn_cls, roi_loc, roi_cls, total = model_frcnn.forward(x_orgin, bboxes_frcnn, labels_frcnn, False)
        # total = rpn_loc + roi_loc
        total.backward()
        grad = x_orgin.grad.data

    elif model_name == 'yolov5':
        bboxes_yolo, labels_yolo = y[2], y[3]
        loss_value_all = 0
        outputs = model_yolo(x_orgin)
        for l in range(len(outputs)):
            loss_item = loss_yolo(l, outputs[l], bboxes_yolo, labels_yolo[l])
            loss_value_all += loss_item
        loss_value = loss_value_all
        loss_value.backward()
        grad = x_orgin.grad.data

    elif model_name == 'ssd':
        bboxes_ssd = y[4]
        out = model_ssd(x_orgin)
        bboxes_ssd = bboxes_ssd.to(conf.device)
        loss = loss_ssd.forward(bboxes_ssd, out)
        loss.backward()
        grad = x_orgin.grad.data

    else:
        ## frcnn loss 
        bboxes_frcnn, labels_frcnn, bboxes_yolo, labels_yolo, bboxes_ssd = y[0], y[1], y[2], y[3], y[4]
        rpn_loc, rpn_cls, roi_loc, roi_cls, frcnn_loss = model_frcnn.forward(x_orgin, bboxes_frcnn, labels_frcnn, False)

        yolo_loss_all = 0
        outputs = model_yolo(x_orgin)
        for l in range(len(outputs)):
            loss_item = loss_yolo(l, outputs[l], bboxes_yolo, labels_yolo[l])
            yolo_loss_all += loss_item
        yolo_loss = yolo_loss_all

        ## sdd loss
        out = model_ssd(x_orgin)
        bboxes_ssd = bboxes_ssd.to(conf.device)
        ssd_loss = loss_ssd.forward(bboxes_ssd, out)

        total_loss = w_yolo*yolo_loss + w_frcnn*frcnn_loss + w_ssd*ssd_loss
        total_loss.backward()
        grad = x_orgin.grad.data
    return grad
    
def get_mask(x, bboxes_cover):
    bboxes_cover = bboxes_cover[0]
    height, width = x.size(2), x.size(3)
    n = x.size(0)
    masks = np.ones((n, 3, height, width))

    
    for i in range(n):
        if conf.detect_AA:
            boxes = bboxes_cover[i]
            m = boxes.shape[0]
            for j in range(m):
                x1, y1, x2, y2 = int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]), int(boxes[j][3])
                masks[i, :, y1:y2, x1:x2] = 0

        if conf.scheme == 'geetest':
            masks[i, :, int(0.9*height):height, :] = 0
    masks = torch.from_numpy(masks)
    masks = masks.to(conf.device)
    return masks

def svre_mi_fgsm(x, y, model_frcnn, model_yolo, model_ssd, loss_yolo, loss_ssd, max_eps, num_iter, momentum, m_svrg, bboxes_cover=None):

    mask = get_mask(x, y)
    eps = max_eps * mask / 255.0
    x_min = torch.clamp(x - max_eps, 0, 1)
    x_max = torch.clamp(x + max_eps, 0, 1)

    alpha = eps / (num_iter * 1.0)

    grad = torch.zeros_like(x)
    x = x.type(torch.FloatTensor).to(conf.device)

    for i in range(num_iter):

        # compute grad to l_fn of img
        x_adv = x

        noise_ensemble = grad_ensemble(model_frcnn, model_yolo, model_ssd, x_adv, y, loss_yolo, loss_ssd, conf.detect_w_frcnn, conf.detect_w_yolo, conf.detect_w_ssd, 'all')
        
        # inner loop
        x_inner = x
        grad_inner = torch.zeros_like(x)
        
        for j in range(m_svrg):

            # choose nets uniformly from nets pool
            model_name = random.choice(conf.detect_model_names)
            noise_x = grad_ensemble(model_frcnn, model_yolo, model_ssd, x, y, loss_yolo, loss_ssd, conf.detect_w_frcnn, conf.detect_w_yolo, conf.detect_w_ssd, model_name)
            noise_x_inner = grad_ensemble(model_frcnn, model_yolo, model_ssd, x_inner, y, loss_yolo, loss_ssd, conf.detect_w_frcnn, conf.detect_w_yolo, conf.detect_w_ssd, model_name)
            noise_inner = noise_x_inner - (noise_x - noise_ensemble)
            noise_inner = noise_inner / torch.mean(torch.abs(noise_inner), (1, 2, 3), keepdims=True)
            grad_inner = momentum * grad_inner + noise_inner

            # update inner adversarial example
            x_inner = x_inner + alpha * torch.sign(grad_inner)
            x_inner = clip_by_value(x_inner, x_min, x_max)

        noise = grad_inner / torch.mean(torch.abs(grad_inner), (1, 2, 3), keepdims=True)
        grad = momentum * grad + noise

        x = x + alpha * torch.sign(grad)
        x = clip_by_value(x, x_min, x_max)
    return x
