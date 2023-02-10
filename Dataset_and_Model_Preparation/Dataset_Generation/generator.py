import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import atexit
import random

from config import config as conf


def add_noise(img, mean, sigma):
    """
            Args:
                img (PIL Image): PIL Image
            Returns:
                PIL Image: PIL image.
            """
    # 将图片灰度标准化
    img_ = np.array(img).copy()
    img_ = img_ / 255.0
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img_.shape)
    # 将噪声和图片叠加
    gaussian_out = img_ + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return Image.fromarray(gaussian_out).convert('RGBA')


def get_color(color_path):

    color_list = []
    with open(color_path, 'r', encoding='utf-8') as f:
        for color in f.readlines():
            c = color.strip('(').strip('\n').strip(')')
            rgb_list = c.split(',')
            r, g, b = rgb_list[0], rgb_list[1], rgb_list[2]
            color_list.append((int(r), int(g), int(b)))
        f.close()
    return color_list


def get_char_map(char_num, char_path):
    with open(char_path, 'r', encoding='utf-8') as f:
        chars_txt = f.read()
        chars = list(chars_txt)
        char_nums = [char_num] * len(chars)
        char_map = dict(zip(chars, char_nums))
    return char_map


def get_fonts_path(font_dir):
    return [os.path.join(font_dir, name) for name in os.listdir(font_dir)]


def get_background_path_list(path):
    return [os.path.join(path, name) for name in os.listdir(path)]


def add_blur(img, box, blur):
    char_img = img.crop(box)
    char_img = char_img.filter(blur)
    char_img = add_noise(char_img, conf.noise_mean, conf.noise_sigma)
    x, y, _, _ = box
    img.paste(char_img, (x, y))
    return img


class generator(object):
    # 初始化一下生成器的参数
    def __init__(self):
        self.colors = get_color(conf.color_path)  # 随机选择颜色
        self.background_height, self.background_width = conf.background_height, conf.background_width
        self.background_path = get_background_path_list(
            conf.background_path)  # 背景地址列表
        self.char_map = get_char_map(conf.char_every_char, conf.char_set_path)  # 统计字符数量，每一个出现多少次，可以保存
        self.char_num_range = conf.char_num_range  # 字符数量的范围
        self.char_size_range = conf.char_size_range  # 字符尺寸范围 可调
        self.max_angle = conf.max_angle  # 最大旋转角度 极验的比较小
        self.fonts = get_fonts_path(conf.font_dir)  # 字体
        self.blur = conf.blur  # 随机增加一些噪声
        self.captcha_path = conf.captcha_path
        self.json_path = conf.json_path
        self.char_path = conf.char_path
        self.map_txt = conf.map_save_path  # 退出时char_map的保存路径，可以不用一次执行到结束
        self.up_sampling = conf.up_sampling  # 上采样，可以使得清晰度变高，但是速度会变慢
        self.color_noise = conf.color_noise # 颜色波动
    def get_background(self):
        back_ground = random.choice(self.background_path)
        back_ground_img = Image.open(back_ground)
        back_ground_img.resize((self.background_width, self.background_height))
        back_ground_img = back_ground_img.convert('RGBA')
        if int(back_ground[-5])%2 == 0:#light background
            return back_ground_img, 0
        elif int(back_ground[-5])%2 == 1:
            return back_ground_img, 1
        else:
            print("baidu 背景与字体颜色配对时出错")

    def save_char_state(self):
        with open(self.map_txt, 'a', encoding='utf-8') as f:
            for key, value in zip(self.char_map.keys(), self.char_map.values()):
                to_save = str(key) + ' ' + str(value)
                f.write(to_save)
        f.close()

    def load_char_state(self):
        new_dict = dict()
        if os.path.exists(self.map_txt):
            with open(self.map_txt, 'r', encoding='utf-8') as f:
                txt = f.readlines()
                for line in txt:
                    key, value = line.strip(' ')
                    new_dict[key] = int(value)
            self.char_map = new_dict

    def update_map(self, char_list):
        pop_list = []
        for key in self.char_map.keys():
            if key in char_list:
                self.char_map[key] -= 1
            if self.char_map[key] == 0:
                pop_list.append(key)
        for item in pop_list:
            self.char_map.pop(item)

    def add_a_char(self, img, char, canvas):
        x_non_zero, y_non_zero = np.nonzero(canvas)
        max_size = self.char_size_range[-1]
        w, h = self.background_width, self.background_height
        # setup
        # 在画布为1的位置选择一个点绘制字符
        location_non_zero = list(zip(x_non_zero, y_non_zero))

        x, y = random.choice(list(location_non_zero))

        # 限定一些位置不能再有字符
        canvas[max(0, x - max_size): x + max_size, max(0, y - max_size): y + max_size] = 0

        # 角度
        angle = random.randint(-self.max_angle, self.max_angle)
        # 两层颜色
        color_out = random.choice(self.colors)
        # color_in = (0,0,0,0)
        color_in = random.choice(self.colors)
        # color_out = random.c
        # hoice(self.colors)
        # color_in = random.choice(self.colors)
        # 字体

        # 字符大小 两层

        size = random.choice(self.char_size_range)



        # 随机选择一个过滤器
        blur = random.choice(self.blur)

        x1 = max(0, x - 3 - random.randint(0, 3))
        y1 = max(0, y - 3 - random.randint(0, 3))
        x2 = min(w, x + max_size + 3 + random.randint(0, 3))
        y2 = min(h, y + max_size + 3 + random.randint(0, 3))
        box = (x1, y1, x2, y2)
        # print(box)
        # print(font_geetest)
        # print(char)

        if self.up_sampling > 1:
            up_sample = self.up_sampling
            x, y = x * up_sample, y * up_sample
            size = size * up_sample
            w, h = w * up_sample, h * up_sample
            img = img.resize((w, h))

        im0 = Image.new(mode='RGBA', size=(w, h), color=(0, 0, 0, 0))
        #空心字符处理s1gh
        font = random.choice(self.fonts)

        ft = ImageFont.truetype(font, size)
        draw = ImageDraw.Draw(im0)
        color2put = (color_out[0] + random.choice(self.color_noise), color_out[1] + random.choice(self.color_noise), color_out[2] + random.choice(self.color_noise))
        draw.text((x, y), char, font=ft, fill=color2put)
        im0 = im0.rotate(angle=angle, center=(x + size // 2, y + size // 2))
        img = Image.alpha_composite(img, im0)
        img = img.resize((self.background_width, self.background_height))
        img = add_blur(img, box, blur)



        return img, box

    def add_chars(self, plaint_background, type_background):
        #字数，固定
        char_num = random.choice(self.char_num_range)
        #字数，不固定
        # char_num = random.choice(self.char_num_range)
        canvas = np.zeros((self.background_width, self.background_height))
        canvas[4:canvas.shape[0] - 55 - 4, 4:canvas.shape[1] - 55 - 4] = 1
        canvas[canvas.shape[0] - 55 - 4 - 120:canvas.shape[0] - 55 - 4, 4:64] = 0
        canvas[canvas.shape[0] - 55 - 4 -40:canvas.shape[0] - 55 - 4, canvas.shape[1] - 55 - 4 - 40:canvas.shape[1] - 55 - 4] = 0
        # canvas[8:canvas.shape[0] - self.char_size_range[-1] -20, 8:canvas.shape[1] - self.char_size_range[-1] -20] = 1
        # canvas[20:canvas.shape[0] -self.char_size_range[-1], canvas.shape[1] - 20:canvas.shape[1] - self.char_size_range[-1]] = 1
        # # 极验的底部没有字
        # canvas[0:canvas.shape[0], 300: canvas.shape[1]] = 0
        boxes = []
        chars = []
        try:
            chars = random.sample(self.char_map.keys(), char_num)
        except:
            print('已经生成所有字符%d次' % conf.char_every_char)
            exit(0)

        self.update_map(chars)

        captcha = plaint_background

        for char in chars:
            captcha, box = self.add_a_char(captcha, char, canvas)
            boxes.append(box)

        return captcha, chars, boxes

    def save_data(self, img, chars, boxes):
        shapes = []
        for char, box in zip(chars, boxes):
            char_img = img.crop(box)
            char_dir = os.path.join(self.char_path, char)
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
            num = len(os.listdir(char_dir)) + 1
            char_path = os.path.join(char_dir, str(num) + '.jpg')
            char_img = char_img.convert('RGB')
            char_img.save(char_path)
            x1, y1, x2, y2 = box
            shape = {'label': char, 'points': [[str(x1), str(y1)], [str(x2), str(y2)]]}
            shapes.append(shape)

        n = len(os.listdir(self.json_path)) + 1
        json_path = os.path.join(self.json_path, str(n) + '.json')
        img_path = os.path.join(self.captcha_path, str(n) + '.jpg')
        json_text = {'shapes': shapes}
        json_data = json.dumps(json_text, indent=4, separators=(',', ': '))
        f = open(json_path, 'w', encoding='utf-8')
        f.write(json_data)
        f.close()
        img.convert("RGB").save(img_path)

    def generate(self, n=99999999999):
        self.load_char_state()
        for i in range(n):
            num = int(len(os.listdir(self.captcha_path)) / 2) + 1
            captcha,type_background = self.get_background()
            captcha, chars, boxes = self.add_chars(captcha, type_background)
            self.save_data(captcha, chars, boxes)
            # captcha.show()
            print(i)
            # # 保存单个字符图片，标签
            # self.save_char(captcha, chars, boxes, num, self.captcha_path)
            # captcha = captcha.convert("RGB")
            # # 保存图片
            # captcha.save(self.captcha_path + "/captcha_" + str(num) + '.jpg')
            # print(i + 1)


if __name__ == '__main__':
    g = generator()
    g.generate(30000)
