import random
import cv2
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from torch.nn.functional import conv2d
from torchvision import transforms

from config import config as conf

class GradCamModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.pretrained = model
        # if conf.model_name == 'inception_resnet_v2':
        self.layerhook.append(
            self.pretrained.mixed_6a.branch1[0].register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

def create_bio_attention_H(gcmodel, x, y):
    out, acts = gcmodel(x)
    acts = acts.detach().cpu()
    loss = torch.nn.CrossEntropyLoss()(out, y).to(conf.device)
    loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[2, 3]).detach().cpu()
    pooled_grads_value = np.expand_dims(pooled_grads, axis=(2, 3))
    heatmap = acts * pooled_grads_value
    heatmap_j = torch.mean(heatmap, dim=1).squeeze()
    Z = torch.abs(heatmap_j).max(axis=1, keepdim=True)[
        0].max(axis=2, keepdim=True)[0]
    heatmap_j /= Z

    H = torch.abs(heatmap_j)
    return H


def create_attention_mask(gcmodel, x, y, max_epsilon, min_epsilon):
    out, acts = gcmodel(x)
    acts = acts.detach().cpu()
    loss = torch.nn.CrossEntropyLoss()(out, y).to(conf.device)
    loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[2, 3]).detach().cpu()
    pooled_grads_value = np.expand_dims(pooled_grads, axis=(2, 3))
    heatmap = acts * pooled_grads_value
    heatmap_j = torch.mean(heatmap, dim=1).squeeze()
    Z = torch.abs(heatmap_j).max(axis=1, keepdim=True)[
        0].max(axis=2, keepdim=True)[0]
    heatmap_j /= Z

    masks = np.zeros((heatmap.shape[0], 3, conf.reco_height, conf.reco_width))
    heatmap = heatmap_j.numpy()
    # output_heatmap(heatmap)
    for i in range(heatmap.shape[0]):
        heat_img = heatmap[i]
        mask = cv2.resize(heat_img, (conf.reco_height, conf.reco_width))
        stack_mask = np.stack([mask, mask, mask])
        stack_mask = np.clip(stack_mask, 0, None)
        # stack_mask = np.abs(stack_mask)
        stack_mask = (max_epsilon - min_epsilon) * stack_mask + min_epsilon
        masks[i] = stack_mask
    masks = torch.from_numpy(masks)
    masks = masks.to(conf.device)
    return masks

def input_diversity(input_tensor):
    p = torch.rand(1).item()
    if p < conf.reco_prob:
        rnd = torch.randint(
            conf.reco_input_shape[-1], conf.reco_diver_image_resize, (),  dtype=torch.int32)
        resize = transforms.Resize([rnd, rnd])
        resized = resize(input_tensor)
        h_rem = conf.reco_diver_image_resize - rnd
        w_rem = conf.reco_diver_image_resize - rnd
        pad_top = int(random.uniform(0, h_rem))
        pad_bottom = h_rem - pad_top
        pad_left = int(random.uniform(0, w_rem))
        pad_right = w_rem - pad_left
        padded = torch.nn.functional.pad(
            resized, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
        padded = padded.view(input_tensor.size(
            0), 3, conf.reco_diver_image_resize, conf.reco_diver_image_resize)
        resize = transforms.Resize([conf.reco_height, conf.reco_width])
        ret = resize(padded).type(torch.FloatTensor)
        return ret.to(conf.device)
    else:
        return input_tensor.type(torch.FloatTensor).to(conf.device)


def compute_global_grad_sim(x, y_batch, alpha, N, model, split_num):
    grad = torch.zeros_like(x)
    for i in range(N):
        x_neighbor = x + (torch.rand(x.size(0), 3, conf.reco_height,
                          conf.reco_width) * alpha).to(conf.device)
        x_neighbor_2 = 1 / 2. * x_neighbor
        x_neighbor_4 = 1 / 4. * x_neighbor
        x_neighbor_8 = 1 / 8. * x_neighbor
        x_neighbor_16 = 1 / 16. * x_neighbor

        x_res = torch.cat((x_neighbor, x_neighbor_2, x_neighbor_4,
                          x_neighbor_8, x_neighbor_16), axis=0)

        loss_fn = nn.CrossEntropyLoss()
        x_res.requires_grad = True
        output = model(input_diversity(x_res))
        loss = loss_fn(output, y_batch)
        loss.backward()
        grad_res = torch.sum(torch.stack(torch.split(x_res.grad.data, split_num)) * torch.tensor
                             ([1, 1/2., 1/4., 1/8., 1/16.])[:, None, None, None, None].to(conf.device), axis=0)
        grad += grad_res
    grad /= (1.0*N)
    return grad

    #   Conv2d-331          [-1, 256, 13, 13]         589,824

def get_kernel(kernel_len=15, nsig=6):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernel_len)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = np.expand_dims(kernel, 0)
    kernel = np.stack([kernel]*3)
    return kernel


def clip_by_value(x, x_min, x_max):
    x = x * (x <= x_max).type(torch.FloatTensor).to(conf.device) + \
        x_max * (x > x_max).type(torch.FloatTensor).to(conf.device)
    x = x * (x >= x_min).type(torch.FloatTensor).to(conf.device) + \
        x_min * (x < x_min).type(torch.FloatTensor).to(conf.device)
    return x



def m_vni_ct_fgsm(x, y_true, model, max_epsilon, central_eps, num_iter, momentum, beta, N):
    gcmodel = GradCamModel(model).to(conf.device)
    mask = create_attention_mask(gcmodel, x, y_true, central_eps, max_epsilon)
    eps = mask / 255.0
    max_epsilon = max_epsilon / 255.0
    x_min = torch.clamp(x - max_epsilon, 0, 1)
    x_max = torch.clamp(x + max_epsilon, 0, 1)

    alpha = eps / (num_iter * 1.0)

    current_gradient = torch.zeros_like(x)
    current_variance = torch.zeros_like(x)

    for i in range(num_iter):
        x_nes = x + momentum * alpha * current_gradient

        split_num = min(conf.reco_batch_size, x_nes.size(0))
        x_batch = torch.cat(
            (x_nes, x_nes/2., x_nes/4., x_nes/8., x_nes/16.), axis=0)
        y_batch = torch.cat([y_true] * 5, axis=0)
        # compute grad to l_fn of img
        loss_fn = nn.CrossEntropyLoss()
        x_batch.requires_grad = True
        output = model(input_diversity(x_batch))
        loss = loss_fn(output, y_batch)
        loss.backward()
        grad = torch.sum(torch.stack(torch.split(x_batch.grad.data, split_num)) * torch.tensor([1, 1/2., 1/4., 1/8., 1/16.])[:, None,
                                                                                                                                   None, None, None].to(conf.device), axis=0)
        global_grad = compute_global_grad_sim(
            x, y_batch, max_epsilon/255.0 * beta, N, model, split_num)

        grad_sum = grad + current_variance
        grad_sum = grad_sum.type(torch.FloatTensor).to(conf.device)
        kernel = get_kernel(15, 6).astype(np.float32)
        kernel = torch.from_numpy(kernel).to(conf.device)
        grad_sum = conv2d(grad_sum, kernel, stride=(1, 1),
                          padding=(7, 7), groups=3)

        current_gradient = momentum * current_gradient + grad_sum / torch.mean(
            torch.abs(grad_sum), (1, 2, 3), keepdims=True)
        current_variance = global_grad - grad
        x = x + alpha * torch.sign(current_gradient)
        x = clip_by_value(x, x_min, x_max)

    return x