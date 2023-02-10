import torch
import os
from PIL import Image
import numpy as np
import json
import jsonpath
import torchvision.transforms as T

class data_parser(object):
    def __init__(self, captcha_path, json_path, class2idx_path, input_shape):

        class2idx = dict()
        with open(class2idx_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        f.close()
        for l in lines:
            char_class, idx = l.strip('\n').split(' ')
            class2idx[char_class] = idx

        captcha_list = []  # 验证码图片
        captcha_names = []
        captcha_char_list = []  # 每幅图片对应的字符和box索引
        char_imgs = []  # 字符图片
        char_classes = []  # 每一张字符图片的类别名
        char_boxes = []  # 每一张字符图片的box
        k = 0
        for img, label in zip(os.listdir(captcha_path), os.listdir(json_path)):

            img_path = os.path.join(captcha_path, img)
            label_path = os.path.join(json_path, label)
            cap_img = Image.open(img_path).convert('RGB')
            w, h = cap_img.size
            captcha_list.append(cap_img)
            captcha_names.append(img)
            idxs = []
            with open(label_path, 'r', encoding='utf-8') as f:
                json_dic = json.load(f)
                char_names = jsonpath.jsonpath(json_dic, '$..label')
                boxes = jsonpath.jsonpath(json_dic, '$..points')

                for box in boxes:
                    x0, y0, x1, y1 = max(box[0][0], 1), max(box[0][1], 1), min(box[1][0], w - 1), min(box[1][1], h - 1)
                    if x1 < x0:
                        x0, x1 = x1, x0
                    if y1 < y0:
                        y0, y1 = y1, y0
                    b = x0, y0, x1, y1
                    char_img = cap_img.crop(b)
                    transform = T.Compose([
                        T.Resize(input_shape[1:]),
                        T.ToTensor(),  # totensor normalization
                    ])

                    char_imgs.append(torch.unsqueeze(transform(char_img), 0))
                    char_boxes.append(b)

                for name in char_names:
                    char_classes.append(name)
                    idxs.append(k)
                    k += 1
                captcha_char_list.append(idxs)

        char_ids = [int(class2idx[char_class]) for char_class in char_classes]
        self.captcha_names = captcha_names
        self.char_ids = char_ids
        self.char_imgs = char_imgs
        self.char_boxes = char_boxes
        self.captcha_list = captcha_list
        self.captcha_char_list = captcha_char_list

    def get_chars(self):
        return self.char_imgs

    def get_ids(self):
        return self.char_ids

    def save_captcha(self, x_adv, output_dir):
        k = 0
        for cap, char_list in zip(self.captcha_list, self.captcha_char_list):
            for idx in char_list:
                box = self.char_boxes[idx]
                char_image = x_adv[idx].cpu()
                x0, y0, x1, y1 = box
                width = x1 - x0
                height = y1 - y0

                char_image = char_image.detach().numpy()
                char_image = np.transpose(char_image, (1, 2, 0))
                char_image = char_image * 255.0
                char_image = Image.fromarray(char_image.astype(np.uint8))
                char_image = char_image.resize((int(width), int(height)))
                cap.paste(char_image, (int(x0), int(y0)))
            cap_path = os.path.join(output_dir, self.captcha_names[k])
            cap.save(cap_path)
            k += 1

        