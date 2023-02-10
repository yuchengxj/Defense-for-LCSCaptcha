from codecs import ignore_errors
import random
import time
import pickle
from tokenize import Ignore

from tqdm import tqdm

from PIL import Image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.attention_ocr import OCR
from utils.dataset import Image_OCR_Dataset
from utils.train_util import train_batch, eval_batch

def main():
    # Loading Json File
    json_file = "/root/autodl-tmp/Attention-OCR-pytorch-main/config_train.json"
    f = open(json_file,encoding='utf-8')
    # f = open(json_file,encoding='gb18030')
    config_train = json.load(f)

    img_width = config_train["img_width"]
    img_height = config_train["img_height"]
    max_len = config_train["max_len"]
    nh = config_train["nh"]
    teacher_forcing_ratio = config_train["teacher_forcing_ratio"]
    train_batch_size = config_train["train_batch_size"]
    val_batch_size = config_train["val_batch_size"]
    lr = config_train["lr"]
    n_epoch = config_train["n_epoch"]
    n_works = config_train["n_works"]
    train_dir=config_train["train_dir"]
    train_file=config_train["train_file"]
    val_dir=config_train["val_dir"]
    val_file=config_train["val_file"]
    # device=config_train["device"]
    chars=config_train["chars"]
    checkpoint_dir=config_train["chk_point_dir"]
    best_model_dir=config_train["best_mod_dir"]
    data_file_separator=config_train["data_file_separator"]
    resume_training=config_train["resume_training"]
    cnn_option=config_train["cnn_option"]
    cnn_backbone=config_train["cnn_backbone_model"][str(cnn_option)] # list containing nets, model_weight

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    device = torch.device(device)



    ds_train = Image_OCR_Dataset(train_dir,train_file,img_width, img_height, data_file_separator,max_len,chars=chars)
    ds_val = Image_OCR_Dataset(val_dir,val_file,img_width, img_height,data_file_separator, max_len,chars=chars)

    tokenizer = ds_train.tokenizer

    train_loader = DataLoader(ds_train, batch_size=train_batch_size, shuffle=True, num_workers=n_works)
    val_loader = DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=n_works)

    model = OCR(img_width, img_height, nh, tokenizer.n_token,
                max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token,cnn_backbone[0]).to(device=device)


    # Loading Checkpoint file
    def load_ckp(checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']


    load_weights = torch.load(cnn_backbone[1])
    names = set()
    for k, w in model.model_arch.named_children():
        names.add(k)

    weights = {}
    for k, w in load_weights.items():
        if k.split('.')[0] in names:
            weights[k] = w

    model.model_arch.load_state_dict(weights)

    # resume_training ==True
    if resume_training == 1:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        ckp_path = checkpoint_dir + "/checkpoint.pt"
        model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    crit = nn.NLLLoss().to(device=device)

    def train_epoch():
        sum_loss_train = 0
        n_train = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = train_batch(x, y, model, optimizer,
                                                  crit, teacher_forcing_ratio, max_len,
                                                  tokenizer,device)

            sum_loss_train += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_train += 1

        return sum_loss_train / n_train, sum_acc / n_train, sum_sentence_acc / n_train

    def eval_epoch():
        sum_loss_eval = 0
        n_eval = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(val_loader)):
            x, y = batch
            x = x.to(device=device)
            y = y.to(device=device)

            loss, acc, sentence_acc = eval_batch(x, y, model, crit, max_len, tokenizer,device)

            sum_loss_eval += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_eval += 1
            model.to(device=device)
            model.eval()
            #with torch.no_grad():
            pred = model(x[0].unsqueeze(0))
            # print("This is a pred.",pred)

            print("PREDICTION ---> ",tokenizer.translate(pred.squeeze(0).argmax(1)))



        return sum_loss_eval / n_eval, sum_acc / n_eval, sum_sentence_acc / n_eval


    LOWEST_LOSS=99999

    # Saving nets with checkpoint and best nets
    def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
        f_path = checkpoint_dir + '/checkpoint.pt'
        torch.save(state, f_path)
        if is_best:
            best_fpath = best_model_dir + '/best_model.pth'
            torch.save(state["state_dict"],best_fpath)

    for epoch in range(n_epoch):
        train_loss, train_acc, train_sentence_acc = train_epoch()
        eval_loss, eval_acc, eval_sentence_acc = eval_epoch()
        is_best = eval_loss < LOWEST_LOSS
        LOWEST_LOSS = min(eval_loss, LOWEST_LOSS)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint, is_best, checkpoint_dir, best_model_dir)

        if resume_training==1:
            new_epoch_start = start_epoch + epoch + 1
            print("Epoch %d" % new_epoch_start)
        else:
            print("Epoch %d" % epoch)

        print('train_loss: %.4f, train_acc: %.4f, train_sentence: %.4f' % (train_loss, train_acc, train_sentence_acc))
        print('eval_loss:  %.4f, eval_acc:  %.4f, eval_sentence:  %.4f' % (eval_loss, eval_acc, eval_sentence_acc))






if __name__ == '__main__':
    main()
