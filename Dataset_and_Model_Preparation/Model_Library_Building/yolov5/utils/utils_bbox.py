import numpy as np
import torch
from torch import Tensor
from torchvision.ops import nms


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        #-----------------------------------------------------------#
        #   20x20的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   40x40的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   80x80的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors_mask   = anchors_mask

    def decode_box_ori(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            #-----------------------------------------------#
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)

            #-----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            #-------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            #-------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            #-----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #-----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            #-----------------------------------------------#
            #   先验框的宽高调整参数
            #-----------------------------------------------#
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])
            #-----------------------------------------------#
            #   获得置信度，是否有物体
            #-----------------------------------------------#
            conf        = torch.sigmoid(prediction[..., 4])
            #-----------------------------------------------#
            #   种类置信度
            #-----------------------------------------------#
            pred_cls    = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            #----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            #----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,20,20
            #----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #----------------------------------------------------------#
            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1]  = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2]  = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3]  = (h.data * 2) ** 2 * anchor_h

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def decode_box_single(self, input, i):

        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size = 1
        #   batch_size, 3 * (4 + 1 + 80), 20, 20
        #   batch_size, 255, 40, 40
        #   batch_size, 255, 80, 80
        #-----------------------------------------------#
        batch_size      = input.size(0)
        input_height    = input.size(2)
        input_width     = input.size(3)

        #-----------------------------------------------#
        #   输入为640x640时
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------#
        stride_h = self.input_shape[0] / input_height
        stride_w = self.input_shape[1] / input_width
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 20, 20, 85
        #   batch_size, 3, 40, 40, 85
        #   batch_size, 3, 80, 80, 85
        #-----------------------------------------------#
        prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        #-----------------------------------------------#
        #   先验框的中心位置的调整参数
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #-----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        #-----------------------------------------------#
        #   获得置信度，是否有物体
        #-----------------------------------------------#
        conf        = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   种类置信度
        #-----------------------------------------------#
        pred_cls    = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   batch_size,3,20,20
        #----------------------------------------------------------#

        #----------------------------                      s1gh在这之后Function就断掉了                         ------------------------------#

        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size,3,20,20
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
        #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
        #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
        #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
        #----------------------------------------------------------#
        pred_boxes          = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0]  = x * 2. - 0.5 + grid_x
        pred_boxes[..., 1]  = y * 2. - 0.5 + grid_y
        pred_boxes[..., 2]  = (w * 2) ** 2 * anchor_w
        pred_boxes[..., 3]  = (h * 2) ** 2 * anchor_h

        #----------------------------------------------------------#
        #   将输出结果归一化成小数的形式
        #----------------------------------------------------------#
        _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        # outputs.append(output.data)
        return output

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            #-----------------------------------------------#
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)

            #-----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            #-------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            #-------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            #-----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #-----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            #-----------------------------------------------#
            #   先验框的宽高调整参数
            #-----------------------------------------------#
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])


            ids = torch.arange(0, 3 * input_height * input_width, 1).view(w.unsqueeze(4).shape).to(torch.device('cuda:0') )

            prediction = torch.cat([prediction, ids], dim=4)


            #-----------------------------------------------#
            #   获得置信度，是否有物体
            #-----------------------------------------------#
            conf        = torch.sigmoid(prediction[..., 4])
            #-----------------------------------------------#
            #   种类置信度
            #-----------------------------------------------#
            pred_cls    = torch.sigmoid(prediction[..., 5])

            index = prediction[..., 6]










            # ----------------------------                      s1gh在这之后Function就断掉了                         ------------------------------#










            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            #----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # idx = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            #     batch_size * len(self.anchors_mask[i]), 1, 1).view(input.shape).type(FloatTensor)
            #----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,20,20
            #----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #----------------------------------------------------------#
            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x * 2. - 0.5 + grid_x
            pred_boxes[..., 1]  = y * 2. - 0.5 + grid_y
            pred_boxes[..., 2]  = (w * 2) ** 2 * anchor_w
            pred_boxes[..., 3]  = (h * 2) ** 2 * anchor_h

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs


    def non_max_suppression_single(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        image_pred = prediction
        # k = prediction[:, index, :]

        # output = [None for _ in range(len(prediction))]
        # for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#


        # class_pred是叶子节点，生成于条件判断
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()


        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        # if debug_i == 0:
        #     if i == 1:
        #         print()

        # if not image_pred.size(0):
        #     continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        # detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        detections = torch.cat((image_pred[:, :5], class_conf.float()), 1)

        return detections

    def bbox_iou(self, box1, box2):
        """
        Returns the IoU of two bounding boxes
        得到bbox的坐标
        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Intersection area
        if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,
                                   torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
                inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
        else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
                inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou


    def non_max_suppression_changed(self, prediction, num_classes, input_shape, image_shape, letterbox_image,
                            conf_thres=0.5, nms_thres=0.4, debug_i=0):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        # k = prediction[:, index, :]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#

            # class_pred是叶子节点，生成于条件判断
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if debug_i == 0:
                if i == 1:
                    print()

            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            # detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            detections = torch.cat((image_pred[:, :5], class_conf.float()), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            # unique_labels = detections[:, -1].cpu().unique()
            #
            # if prediction.is_cuda:
            #     unique_labels = unique_labels.cuda()
            #     detections = detections.cuda()

            # for c in unique_labels:
            #     #------------------------------------------#
            #     #   获得某一类得分筛选后全部的预测结果
            #     #------------------------------------------#
            #     detections_class = detections[detections[:, -1] == c]
            #
            #     #------------------------------------------#
            #     #   使用官方自带的非极大抑制会速度更快一些！
            #     #   筛选出一定区域内，属于同一种类得分最大的框
            #     #------------------------------------------#

            keep = nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thres
            )
            max_detections = detections[keep]

            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        # return output
        #     if output[i] is not None:
        #         output[i]           = output[i].cpu().numpy()
        #         box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
        #         output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

    def non_max_suppression_HA(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.5, ):
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            obj_conf, class_pred = torch.max(image_pred[:, 4:5], 1, keepdim=True)
            # class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            # obj_conf = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            # if not image_pred.size(0):
            #     continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        if detections == None:
            return None
        else:
            return detections

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            #----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            #----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            #------------------------------------------#
            #   获得预测结果中包含的所有种类
            #------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #   筛选出一定区域内，属于同一种类得分最大的框
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    # -----------------------------------------------------------------#
    #   由groundtruth获取tensor形式的坐标，同时映射到640*640（映射成功否存疑），用于通过IOU筛选
    # -----------------------------------------------------------------#
    def yolo_boxes2tensor(self, id, input_shape, image_shape, letterbox_image):
        import os
        list_xy = []
        with open(os.path.join("img/1001.txt"), "r") as new_f:
            for line in new_f.readlines():
                tmp = line.split(' ')

                list_xy.append(float(tmp[2]))
                list_xy.append(float(tmp[1]))
                list_xy.append(float(tmp[4]))
                list_xy.append(float(tmp[3]))
            a = new_f.readlines()

        boxes = np.array(list_xy).reshape(-1, 4)


        boxes /= np.concatenate([image_shape, image_shape], axis=-1)

        box_mins = boxes[:, 0:2]
        box_maxes = boxes[:, 2:4]

        box_yx = (box_mins + box_maxes)/2.
        box_hw = box_maxes - box_mins

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            # box_yx = (box_yx - offset) * scale
            # box_hw *= scale

            box_hw /= scale
            box_yx /= scale
            box_yx += offset

        box_xy = box_yx[..., ::-1]
        box_wh = box_hw[..., ::-1]

        aaa = (box_xy * 2. - box_wh) / 2.
        bbb = (box_xy * 2. + box_wh) / 2.


        # aaa = aaa[:, :-1]
        # bbb = bbb[:, :-1]

        output = np.concatenate((aaa, bbb), axis=1)

        return output

    def box_area(self, boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, boxes1, boxes2):
        area1 = self.box_area(boxes1)  # 每个框的面积
        area2 = self.box_area(boxes2)

        boxes1.to(torch.device('cuda'))
        boxes2.to(torch.device('cuda'))

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    def nms(self, boxes, scores, iou_threshold):

        keep = []  # 最终保留的结果， 在boxes中对应的索引；
        idxs = scores.argsort()  # 值从小到大的 索引

        while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
            # 得分最大框对应的索引, 以及对应的坐标
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]  # [1, 4]
            keep.append(max_score_index)
            if idxs.size(0) == 1:  # 就剩余一个框了；
                break
            idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
            other_boxes = boxes[idxs]  # [?, 4]
            ious = self.box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
            idxs = idxs[ious[0] <= iou_threshold]

        keep = idxs.new(keep)  # Tensor
        return keep

    def m_bbox(self, boxes_pred, boxes_real, scores, iou_threshold, k):

        """
        :param boxes: [N, 4] N = 25200
        :param scores: [N]
        :param iou_threshold: 0.5 取iou>iou_threshold
        :param k: 10 取前conf前k
        :return: 选出来的索引
        """
        scores_obj = boxes_pred[:, 5]
        boxes_pred = boxes_pred[:, :4]

        scores_obj = scores_obj.reshape(-1)

        boxes_real = torch.from_numpy(boxes_real)
        boxes_real = torch.tensor(boxes_real, dtype=torch.float)
        boxes_real = boxes_real.to(torch.device('cuda'))
        keep_ious = []  # 最终保留的结果， 在boxes中对应的索引；
        idxs = scores_obj.argsort()  # 值从小到大的 索引

        while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
            # 得分最大框对应的索引, 以及对应的坐标
            for single_real_box in boxes_real:
                single_real_box = single_real_box.view(1, 4)

                ious = self.box_iou(single_real_box, boxes_pred)
                # 取大于阈值的idxs
                keep_ious.append(idxs[ious[0] >= iou_threshold])
                scores_obj_tmp = scores_obj[idxs[ious[0] >= iou_threshold]]
            keep_t = torch.tensor([], dtype=torch.int64).to(torch.device('cuda'))
            for t in keep_ious:
                keep_t = torch.cat((keep_t, t), dim=0)
            boxes_ious = boxes_pred[keep_t]
            print()




if __name__ == "__main__":

    pass
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # #---------------------------------------------------#
    # #   将预测值的每个特征层调成真实值
    # #---------------------------------------------------#
    # def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
    #     #-----------------------------------------------#
    #     #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
    #     #-----------------------------------------------#
    #     batch_size      = input.size(0)
    #     input_height    = input.size(2)
    #     input_width     = input.size(3)
    #
    #     #-----------------------------------------------#
    #     #   输入为640x640时 input_shape = [640, 640]  input_height = 20, input_width = 20
    #     #   640 / 20 = 32
    #     #   stride_h = stride_w = 32
    #     #-----------------------------------------------#
    #     stride_h = input_shape[0] / input_height
    #     stride_w = input_shape[1] / input_width
    #     #-------------------------------------------------#
    #     #   此时获得的scaled_anchors大小是相对于特征层的
    #     #   anchor_width, anchor_height / stride_h, stride_w
    #     #-------------------------------------------------#
    #     scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[2]]]
    #
    #     #-----------------------------------------------#
    #     #   batch_size, 3 * (4 + 1 + num_classes), 20, 20 =>
    #     #   batch_size, 3, 5 + num_classes, 20, 20  =>
    #     #   batch_size, 3, 20, 20, 4 + 1 + num_classes
    #     #-----------------------------------------------#
    #     prediction = input.view(batch_size, len(anchors_mask[2]),
    #                             num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
    #
    #     #-----------------------------------------------#
    #     #   先验框的中心位置的调整参数
    #     #-----------------------------------------------#
    #     x = torch.sigmoid(prediction[..., 0])
    #     y = torch.sigmoid(prediction[..., 1])
    #     #-----------------------------------------------#
    #     #   先验框的宽高调整参数
    #     #-----------------------------------------------#
    #     w = torch.sigmoid(prediction[..., 2])
    #     h = torch.sigmoid(prediction[..., 3])
    #     #-----------------------------------------------#
    #     #   获得置信度，是否有物体 0 - 1
    #     #-----------------------------------------------#
    #     conf        = torch.sigmoid(prediction[..., 4])
    #     #-----------------------------------------------#
    #     #   种类置信度 0 - 1
    #     #-----------------------------------------------#
    #     pred_cls    = torch.sigmoid(prediction[..., 5:])
    #
    #     FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    #     LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    #
    #     #----------------------------------------------------------#
    #     #   生成网格，先验框中心，网格左上角
    #     #   batch_size,3,20,20
    #     #   range(20)
    #     #   [
    #     #       [0, 1, 2, 3 ……, 19],
    #     #       [0, 1, 2, 3 ……, 19],
    #     #       …… （20次）
    #     #       [0, 1, 2, 3 ……, 19]
    #     #   ] * (batch_size * 3)
    #     #   [batch_size, 3, 20, 20]
    #     #
    #     #   [
    #     #       [0, 1, 2, 3 ……, 19],
    #     #       [0, 1, 2, 3 ……, 19],
    #     #       …… （20次）
    #     #       [0, 1, 2, 3 ……, 19]
    #     #   ].T * (batch_size * 3)
    #     #   [batch_size, 3, 20, 20]
    #     #----------------------------------------------------------#
    #     grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
    #         batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
    #     grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
    #         batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)
    #
    #     #----------------------------------------------------------#
    #     #   按照网格格式生成先验框的宽高
    #     #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
    #     #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
    #     #----------------------------------------------------------#
    #     anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    #     anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    #     anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
    #     anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
    #
    #     #----------------------------------------------------------#
    #     #   利用预测结果对先验框进行调整
    #     #   首先调整先验框的中心，从先验框中心向右下角偏移
    #     #   再调整先验框的宽高。
    #     #   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
    #     #   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
    #     #   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
    #     #   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h
    #     #----------------------------------------------------------#
    #     pred_boxes          = FloatTensor(prediction[..., :4].shape)
    #     pred_boxes[..., 0]  = x.data * 2. - 0.5 + grid_x
    #     pred_boxes[..., 1]  = y.data * 2. - 0.5 + grid_y
    #     pred_boxes[..., 2]  = (w.data * 2) ** 2 * anchor_w
    #     pred_boxes[..., 3]  = (h.data * 2) ** 2 * anchor_h
    #
    #     point_h = 5
    #     point_w = 5
    #
    #     box_xy          = pred_boxes[..., 0:2].cpu().numpy() * 32
    #     box_wh          = pred_boxes[..., 2:4].cpu().numpy() * 32
    #     grid_x          = grid_x.cpu().numpy() * 32
    #     grid_y          = grid_y.cpu().numpy() * 32
    #     anchor_w        = anchor_w.cpu().numpy() * 32
    #     anchor_h        = anchor_h.cpu().numpy() * 32
    #
    #     fig = plt.figure()
    #     ax  = fig.add_subplot(121)
    #     from PIL import Image
    #     img = Image.open("img/street.jpg").resize([640, 640])
    #     plt.imshow(img, alpha=0.5)
    #     plt.ylim(-30, 650)
    #     plt.xlim(-30, 650)
    #     plt.scatter(grid_x, grid_y)
    #     plt.scatter(point_h * 32, point_w * 32, c='black')
    #     plt.gca().invert_yaxis()
    #
    #     anchor_left = grid_x - anchor_w / 2
    #     anchor_top  = grid_y - anchor_h / 2
    #
    #     rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w],anchor_top[0, 0, point_h, point_w]], \
    #         anchor_w[0, 0, point_h, point_w],anchor_h[0, 0, point_h, point_w],color="r",fill=False)
    #     rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w],anchor_top[0, 1, point_h, point_w]], \
    #         anchor_w[0, 1, point_h, point_w],anchor_h[0, 1, point_h, point_w],color="r",fill=False)
    #     rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w],anchor_top[0, 2, point_h, point_w]], \
    #         anchor_w[0, 2, point_h, point_w],anchor_h[0, 2, point_h, point_w],color="r",fill=False)
    #
    #     ax.add_patch(rect1)
    #     ax.add_patch(rect2)
    #     ax.add_patch(rect3)
    #
    #     ax  = fig.add_subplot(122)
    #     plt.imshow(img, alpha=0.5)
    #     plt.ylim(-30, 650)
    #     plt.xlim(-30, 650)
    #     plt.scatter(grid_x, grid_y)
    #     plt.scatter(point_h * 32, point_w * 32, c='black')
    #     plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
    #     plt.gca().invert_yaxis()
    #
    #     pre_left    = box_xy[...,0] - box_wh[...,0] / 2
    #     pre_top     = box_xy[...,1] - box_wh[...,1] / 2
    #
    #     rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],\
    #         box_wh[0, 0, point_h, point_w,0], box_wh[0, 0, point_h, point_w,1],color="r",fill=False)
    #     rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],\
    #         box_wh[0, 1, point_h, point_w,0], box_wh[0, 1, point_h, point_w,1],color="r",fill=False)
    #     rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],\
    #         box_wh[0, 2, point_h, point_w,0], box_wh[0, 2, point_h, point_w,1],color="r",fill=False)
    #
    #     ax.add_patch(rect1)
    #     ax.add_patch(rect2)
    #     ax.add_patch(rect3)
    #
    #     plt.show()
    #     #
    # feat            = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    # anchors         = np.array([[116, 90], [156, 198], [373, 326], [30,61], [62,45], [59,119], [10,13], [16,30], [33,23]])
    # anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
