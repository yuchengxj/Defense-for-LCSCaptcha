import os
import numpy as np
import torch
from tqdm import tqdm

from config import config as conf
from charloader import data_parser
from M_VNI_CT_FGSM import m_vni_ct_fgsm

if not os.path.exists(conf.reco_output_path):
    os.mkdir(conf.reco_output_path)


class evaluator(object):
    def __init__(self):
        self.white_box_path = conf.reco_white_box_model_path
        self.black_box_paths = conf.reco_black_box_model_path
        self.black_box_names = conf.reco_black_box_models
        self.origin_x = None
        self.adv_x = None
        self.y_true = None

    def add(self, x, x_adv, y):
        if self.origin_x is None:
            self.origin_x = x.clone()
        else:
            self.origin_x = torch.cat([self.origin_x, x], axis=0)
        if self.adv_x is None:
            self.adv_x = x_adv.clone()
        else:
            self.adv_x = torch.cat([self.adv_x, x_adv], axis=0)

        if self.y_true is None:
            self.y_true = y.clone()
        else:
            self.y_true = torch.cat([self.y_true, y], axis=0)

    def evaluate_models(self, model):
        # evaluate white box
        acc, acc_adv, acc_drop = self.evaluate_adv(model)
        print(f'White box nets IncResV2 accuracy = {acc}')
        print(f'White box nets IncResV2 adv accuracy = {acc_adv}')
        print(f'Accuracy drop = {acc_drop}')
        print('*'*50)

        # evaluate black box
        for model_path, model_name in zip(self.black_box_paths, self.black_box_names):
            model = torch.load(model_path)
            model = model.module
            model.eval()
            acc, acc_adv, acc_drop = self.evaluate_adv(model)
            print(f'Black box nets {model_path} accuracy = {acc}')
            print(f'Black box nets {model_path} adv accuracy = {acc_adv}')
            print(f'Accuracy drop = {acc_drop}')
            print('*'*50)

    def evaluate_adv(self, model):
        n = self.adv_x.size(0)
        x_correct = 0
        x_adv_correct = 0

        origin_x_split = torch.split(self.origin_x, conf.reco_batch_size)
        adv_x_split = torch.split(self.adv_x, conf.reco_batch_size)
        y_true_split = torch.split(self.y_true, conf.reco_batch_size)
        for i in range(len(origin_x_split)):
            x = origin_x_split[i]
            x_adv = adv_x_split[i]
            y = y_true_split[i]
            x = x.to(conf.device)
            x_adv = x_adv.type(torch.FloatTensor).to(conf.device)
            out = model(x)
            out_adv = model(x_adv)
            for j in range(y.size(0)):
                y_true = y[j].item()
                y_pred = out.argmax(1)[j].item()
                y_adv = out_adv.argmax(1)[j].item()
                if y_true == y_pred:
                    x_correct += 1
                if y_true == y_adv:
                    x_adv_correct += 1
        return x_correct/n, x_adv_correct/n, x_correct/n - x_adv_correct/n


Evaluator = evaluator()


if __name__ == "__main__":
    dp = data_parser(conf.capthca_path, conf.json_path, conf.class2idx_path, conf.reco_input_shape)
    chars = dp.get_chars()
    labels = dp.get_ids()

    chars = torch.cat(chars, 0)

    model = torch.load(conf.reco_white_box_model_path)
    model = model.module
    model.eval()

    loop = tqdm(range(0, chars.size(0), conf.reco_batch_size),
                total=chars.size(0)//conf.reco_batch_size+1)
    for i in loop:
    
        end = min(i+conf.reco_batch_size, len(chars))
        x = chars[i:end].to(conf.device)
        y = torch.from_numpy(np.array(labels[i:end])).to(conf.device)
        x_adv = m_vni_ct_fgsm(x, y, model, max_epsilon=conf.reco_max_eps, central_eps=conf.central_eps,
                              num_iter=conf.reco_iter, momentum=conf.reco_momentum, beta=conf.reco_beta, N=conf.reco_N)
        Evaluator.add(x, x_adv, y)
        # Evaluator.evaluate_models(nets)
        chars[i:end] = x_adv.cpu()

    Evaluator.evaluate_models(model)
    dp.save_captcha(chars, conf.reco_output_path)
