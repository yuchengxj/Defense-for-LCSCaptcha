from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from config import config as conf
import sys
sys.path.append(r"./Dataset_and_Model_Preparation/Model_Library_Building")
from yolov5.utils.utils import cvtColor, preprocess_input


class Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors_yolo, anchors_mask, anchors_ssd, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7, overlap_threshold=0.5):
        super(Dataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors_yolo       = anchors_yolo
        self.anchors_mask       = anchors_mask
        self.anchors_ssd        = anchors_ssd
        self.num_anchors_ssd     = len(anchors_ssd)
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4
        self.overlap_threshold  = overlap_threshold

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
            if self.mixup and self.rand() < self.mixup_prob:
                lines           = sample(self.annotation_lines, 1)
                image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box, location, path = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))

        #---------------------------------------------------#
        #   获取yolov5的标签
        #---------------------------------------------------#
        
        boxes_yolo, labels_yolo = self.get_yolo_label(box)
        boxes_frcnn, labels_frcnn = self.get_frcnn_label(box)
        boxes_ssd = self.get_ssd_label(box)

        return image, boxes_frcnn, labels_frcnn, boxes_yolo, labels_yolo, boxes_ssd, location, path


    def get_yolo_label(self, box):
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        y_true = self.get_target(box)
        return box, y_true

    def get_frcnn_label(self, y):
        box_data    = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box         = box_data[:, :4]
        label       = box_data[:, -1]
        return box, label

    def get_ssd_label(self, box):
        if len(box)!=0:
            boxes               = np.array(box[:,:4] , dtype=np.float32)
            # 进行归一化，调整到0-1之间
            boxes[:, [0, 2]]    = boxes[:,[0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]]    = boxes[:,[1, 3]] / self.input_shape[0]
            # 对真实框的种类进行one hot处理
            one_hot_label   = np.eye(self.num_classes - 1)[np.array(box[:,4], np.int32)]
            box             = np.concatenate([boxes, one_hot_label], axis=-1)

        box = self.assign_boxes(box)
        return np.array(box, np.float32)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            location = (dx, dy, dx + nw, dy + nh)

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box, location, line[0]
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def iou(self, box):
        #---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        #---------------------------------------------#
        inter_upleft    = np.maximum(self.anchors_ssd[:, :2], box[:2])
        inter_botright  = np.minimum(self.anchors_ssd[:, 2:4], box[2:])

        inter_wh    = inter_botright - inter_upleft
        inter_wh    = np.maximum(inter_wh, 0)
        inter       = inter_wh[:, 0] * inter_wh[:, 1]
        #---------------------------------------------# 
        #   真实框的面积
        #---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        #---------------------------------------------#
        #   先验框的面积
        #---------------------------------------------#
        area_gt = (self.anchors_ssd[:, 2] - self.anchors_ssd[:, 0])*(self.anchors_ssd[:, 3] - self.anchors_ssd[:, 1])
        #---------------------------------------------#
        #   计算iou
        #---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances = [0.1, 0.1, 0.2, 0.2]):
        #---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        #---------------------------------------------#
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors_ssd, 4 + return_iou))
        
        #---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        #---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        #---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        #---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        
        #---------------------------------------------#
        #   利用iou进行赋值 
        #---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        #---------------------------------------------#
        #   找到对应的先验框
        #---------------------------------------------#
        assigned_anchors = self.anchors_ssd[assign_mask]

        #---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        #---------------------------------------------#
        box_center  = 0.5 * (box[:2] + box[2:])
        box_wh      = box[2:] - box[:2]
        #---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        #---------------------------------------------#
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh     = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])
        
        #------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        #------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        #---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        #---------------------------------------------------#
        assignment          = np.zeros((self.num_anchors_ssd, 4 + self.num_classes + 1))
        assignment[:, 4]    = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        encoded_boxes   = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        #---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        #---------------------------------------------------#
        encoded_boxes   = encoded_boxes.reshape(-1, self.num_anchors_ssd, 5)
        
        #---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        #---------------------------------------------------#
        best_iou        = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx    = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask   = best_iou > 0
        best_iou_idx    = best_iou_idx[best_iou_mask]
        
        #---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        #---------------------------------------------------#
        assign_num      = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes   = encoded_boxes[:, best_iou_mask, :]
        #---------------------------------------------------#
        #   编码后的真实框的赋值
        #---------------------------------------------------#
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        #----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        #----------------------------------------------------------#
        assignment[:, 4][best_iou_mask]     = 0
        assignment[:, 5:-1][best_iou_mask]  = boxes[best_iou_idx, 4:]
        #----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        #----------------------------------------------------------#
        assignment[:, -1][best_iou_mask]    = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w      = grid_shapes[l]
            anchors         = np.array(self.anchors_yolo) / {0:32, 1:16, 2:8, 3:4}[l]
            
            batch_target = np.zeros_like(targets)
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
            max_ratios           = np.max(ratios, axis = -1)
            
            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
                        
        return y_true
    
# DataLoader中collate_fn使用
def dataset_collate(batch):
    images  = []
    boxes_frcnn  = []
    labels_frcnn = []
    boxes_yolo = []
    labels_yolo = [[] for _ in batch[0][4]]
    boxes_ssd = []
    locations = []
    paths = []
    for img, box_frcnn, label_frcnn, box_yolo, label_yolo, box_ssd, location, path in batch:
        images.append(img)
        boxes_frcnn.append(box_frcnn)
        labels_frcnn.append(label_frcnn)
        boxes_yolo.append(box_yolo)
        boxes_ssd.append(box_ssd)
        locations.append(location)
        paths.append(path)
        for i, sub_y_true in enumerate(label_yolo):
            labels_yolo[i].append(sub_y_true)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    boxes_yolo  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in boxes_yolo]
    labels_yolo = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in labels_yolo]
    boxes_ssd = torch.from_numpy(np.array(boxes_ssd)).type(torch.FloatTensor)
    targets = [boxes_frcnn, labels_frcnn, boxes_yolo, labels_yolo, boxes_ssd]
    return images, targets, locations, paths
