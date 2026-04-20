#!/usr/bin/python
# -*- encoding: utf-8 -*-

from datasets import (
    get_dataset_config,
    build_fusion_dataset,
    build_seg_dataset,
    get_fusion_output_dir,
)

from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from model_TII import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# def parse_args():
#     parse = argparse.ArgumentParser()

#     parser.add_argument('--dataset_name', default='flame3', type=str, help='flame3 or msrs')
#     parser.add_argument('--dataset_root', default='./FLAME3', type=str, help='dataset root path')

#     return parse.parse_args()

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='flame3', type=str, help='flame3 or msrs')
    parser.add_argument('--dataset_root', default='./FLAME3', type=str, help='dataset root path')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader workers')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs or outer loop iterations')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id, use -1 for cpu')
    return parser.parse_args()


# def RGB2YCrCb(input_im):
#     im_flat = input_im.transpose(1, 3).transpose(
#         1, 2).reshape(-1, 3)  # (nhw,c)
#     R = im_flat[:, 0]
#     G = im_flat[:, 1]
#     B = im_flat[:, 2]
#     Y = 0.299 * R + 0.587 * G + 0.114 * B
#     Cr = (R - Y) * 0.713 + 0.5
#     Cb = (B - Y) * 0.564 + 0.5
#     Y = torch.unsqueeze(Y, 1)
#     Cr = torch.unsqueeze(Cr, 1)
#     Cb = torch.unsqueeze(Cb, 1)
#     temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
#     out = (
#         temp.reshape(
#             list(input_im.size())[0],
#             list(input_im.size())[2],
#             list(input_im.size())[3],
#             3,
#         )
#         .transpose(1, 3)
#         .transpose(2, 3)
#     )
#     return out

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def RGB2YCrCb(input_im):
    device = input_im.device
    dtype = input_im.dtype
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device=device, dtype=dtype)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

# def YCrCb2RGB(input_im):
#     im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
#     mat = torch.tensor(
#         [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
#     ).cuda()
#     bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
#     temp = (im_flat + bias).mm(mat).cuda()
#     out = (
#         temp.reshape(
#             list(input_im.size())[0],
#             list(input_im.size())[2],
#             list(input_im.size())[3],
#             3,
#         )
#         .transpose(1, 3)
#         .transpose(2, 3)
#     )
#     return out

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def YCrCb2RGB(input_im):
    device = input_im.device
    dtype = input_im.dtype
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]],
        device=device,
        dtype=dtype,
    )
    bias = torch.tensor([0.0 / 255, -0.5, -0.5], device=device, dtype=dtype)
    temp = (im_flat + bias).mm(mat)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

# 原train_seg函数，保留以供参考
# def train_seg(i=0, logger=None, args=None):
#     load_path = './model/Fusion/model_final.pth'
#     modelpth = './model'
#     Method = 'Fusion'
#     modelpth = os.path.join(modelpth, Method)
#     os.makedirs(modelpth, mode=0o777, exist_ok=True)
#     # if logger == None:
#     #     logger = logging.getLogger()
#     #     setup_logger(modelpth)

#     # dataset
#     n_classes = 9
#     n_img_per_gpu = args.batch_size
#     n_workers = 4
#     # # 原cropsize时写死的，建议改为从配置中读取，以适配不同数据集的需求。
#     # cropsize = [640, 480]
#     # ds = CityScapes('./MSRS/', cropsize=cropsize, mode='train', Method=Method)
#     # 新增：从配置中读取cropsize，并且使用build_seg_dataset函数构建数据集，以适配不同数据集的需求。
#     cropsize = cfg.cropsize
#     ds = build_seg_dataset(cfg, split='train', method=cfg.method_name)

#     dl = DataLoader(
#         ds,
#         batch_size=n_img_per_gpu,
#         shuffle=False,
#         num_workers=n_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     # model
#     ignore_idx = 255
#     net = BiSeNet(n_classes=n_classes)
#     if i>0:
#         net.load_state_dict(torch.load(load_path))
#     net.cuda()
#     net.train()
#     print('Load Pre-trained Segmentation Model:{}!'.format(load_path))
#     score_thres = 0.7

#     # 原n_min的计算方式是基于固定的cropsize和batch_size的，建议改为动态计算，以适配不同配置的需求。
#     n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
#     # 新增：动态计算n_min，并且从配置中读取ignore_index，以适配不同数据集的需求。
#     crop_w, crop_h = cfg.cropsize
#     n_min = args.batch_size * crop_w * crop_h // 16

#     criteria_p = OhemCELoss(
#         thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#     criteria_16 = OhemCELoss(
#         thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#     # optimizer
#     momentum = 0.9
#     weight_decay = 5e-4
#     lr_start = 1e-2
#     max_iter = 80000
#     power = 0.9
#     warmup_steps = 1000
#     warmup_start_lr = 1e-5
#     it_start = i*20000
#     iter_nums=20000

#     optim = Optimizer(
#         model=net,
#         lr0=lr_start,
#         momentum=momentum,
#         wd=weight_decay,
#         warmup_steps=warmup_steps,
#         warmup_start_lr=warmup_start_lr,
#         max_iter=max_iter,
#         power=power,
#         it=it_start,
#     )

#     # train loop
#     msg_iter = 10
#     loss_avg = []
#     st = glob_st = time.time()
#     diter = iter(dl)
#     epoch = 0
#     for it in range(iter_nums):
#         try:
#             im, lb, _ = next(diter)
#             if not im.size()[0] == n_img_per_gpu:
#                 raise StopIteration
#         except StopIteration:
#             epoch += 1
#             # sampler.set_epoch(epoch)
#             diter = iter(dl)
#             im, lb, _ = next(diter)
#         im = im.cuda()
#         lb = lb.cuda()
#         lb = torch.squeeze(lb, 1)

#         optim.zero_grad()
#         out, mid = net(im)
#         lossp = criteria_p(out, lb)
#         loss2 = criteria_16(mid, lb)
#         loss = lossp + 0.75 * loss2
#         loss.backward()
#         optim.step()

#         loss_avg.append(loss.item())
#         # print training log message
#         if (it + 1) % msg_iter == 0:
#             loss_avg = sum(loss_avg) / len(loss_avg)

#             lr = optim.lr
#             ed = time.time()
#             t_intv, glob_t_intv = ed - st, ed - glob_st
#             eta = int(( max_iter - it) * (glob_t_intv / it))
#             eta = str(datetime.timedelta(seconds=eta))
#             msg = ', '.join(
#                 [
#                     'it: {it}/{max_it}',
#                     'lr: {lr:4f}',
#                     'loss: {loss:.4f}',
#                     'eta: {eta}',
#                     'time: {time:.4f}',
#                 ]
#             ).format(
#                 it=it_start+it + 1, max_it= max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
#             )
#             logger.info(msg)
#             loss_avg = []
#             st = ed
#     # dump the final model
#     save_pth = osp.join(modelpth, 'model_final.pth')
#     net.cpu()
#     state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
#     torch.save(state, save_pth)
#     logger.info(
#         'Segmentation Model Training done~, The Model is saved to: {}'.format(
#             save_pth)
#     )
#     logger.info('\n')

# # 新加入模块化版本，然后切换调用。
# def train_seg(i=0, logger=None, args=None, cfg=None):
#     save_pth = osp.join('./model/Fusion/model_seg')
#     os.makedirs(save_pth, exist_ok=True)
#     logger = logger

#     n_classes = cfg.num_classes
#     cropsize = list(cfg.cropsize)
#     ds = build_seg_dataset(cfg, split='train', method=cfg.method_name)
#     dl = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     net = BiSeNet(n_classes=n_classes)
#     if i > 0:
#         net.load_state_dict(torch.load('./model/Fusion/model_final.pth'))

#     net.cuda()
#     net.train()

#     score_thres = 0.75
#     n_min = args.batch_size * cropsize[0] * cropsize[1] // 16
#     criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min).cuda()
#     criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min).cuda()

#     optimizer = Optimizer(
#         model=net,
#         lr0=1e-2,
#         momentum=0.9,
#         wd=5e-4,
#         warmup_steps=1000,
#         warmup_start_lr=1e-5,
#         max_iter=args.epochs,
#         power=0.9,
#     )

#     diter = iter(dl)
#     for it in range(args.epochs):
#         try:
#             im, lb, _ = next(diter)
#             if not im.size()[0] == args.batch_size:
#                 raise StopIteration
#         except StopIteration:
#             diter = iter(dl)
#             im, lb, _ = next(diter)

#         im = im.cuda()
#         lb = torch.squeeze(lb, 1).cuda()

#         optimizer.zero_grad()
#         out, mid = net(im)
#         lossp = criteria_p(out, lb)
#         loss2 = criteria_16(mid, lb)
#         loss = lossp + loss2
#         loss.backward()
#         optimizer.step()

#         if logger is not None:
#             logger.info(f'[seg round {i}] iter={it}, loss={loss.item():.6f}')

#     torch.save(net.state_dict(), osp.join(save_pth, 'model_final.pth'))
#     net.cpu()

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def train_seg(i=0, logger=None, args=None, cfg=None):
    save_pth = osp.join('./model/Fusion/model_seg')
    os.makedirs(save_pth, exist_ok=True)
    load_path = osp.join(save_pth, 'model_final.pth')

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    n_classes = cfg.num_classes
    cropsize = cfg.cropsize
    ds = build_seg_dataset(cfg, split='train', method=cfg.method_name)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ignore_idx = cfg.ignore_index
    net = BiSeNet(n_classes=n_classes)
    if i > 0 and os.path.isfile(load_path):
        net.load_state_dict(torch.load(load_path, map_location=device))
    net = net.to(device)
    net.train()

    crop_w, crop_h = cfg.cropsize
    n_min = args.batch_size * crop_w * crop_h // 16

    criteria_p = OhemCELoss(thresh=0.7, n_min=n_min, ignore_lb=ignore_idx).to(device)
    criteria_16 = OhemCELoss(thresh=0.7, n_min=n_min, ignore_lb=ignore_idx).to(device)

    optim = Optimizer(
        model=net,
        lr0=1e-2,
        momentum=0.9,
        wd=5e-4,
        warmup_steps=1000,
        warmup_start_lr=1e-5,
        max_iter=80000,
        power=0.9,
    )

    diter = iter(dl)
    for it in range(20000):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == args.batch_size:
                raise StopIteration
        except StopIteration:
            diter = iter(dl)
            im, lb, _ = next(diter)

        im = im.to(device, non_blocking=True)
        lb = torch.squeeze(lb, 1).to(device, non_blocking=True)

        optim.zero_grad()
        out, out16 = net(im)
        loss_p = criteria_p(out, lb)
        loss_16 = criteria_16(out16, lb)
        loss = loss_p + loss_16
        loss.backward()
        optim.step()

        if logger is not None:
            logger.info(f"[train_seg] round={i}, iter={it}, loss={loss.item():.6f}")

    torch.save(net.state_dict(), load_path)
    net.cpu()


# 原train_fusion函数，保留以供参考，建议在其下方加入新的train_fusion函数以适配新的配置系统，并且调整了日志输出格式以包含更多细节。

# def train_fusion(num=0, logger=None, args=None):
#     # num: control the segmodel 
#     lr_start = 0.001
#     modelpth = './model'
#     Method = 'Fusion'
#     modelpth = os.path.join(modelpth, Method)
#     fusionmodel = eval('FusionNet')(output=1)
#     fusionmodel.cuda()
#     fusionmodel.train()
#     optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
#     if num>0:
#         n_classes = 9
#         segmodel = BiSeNet(n_classes=n_classes)
#         save_pth = osp.join(modelpth, 'model_final.pth')
#         if logger == None:
#             logger = logging.getLogger()
#             setup_logger(modelpth)
#         segmodel.load_state_dict(torch.load(save_pth))
#         segmodel.cuda()
#         segmodel.eval()
#         for p in segmodel.parameters():
#             p.requires_grad = False
#         print('Load Segmentation Model {} Sucessfully~'.format(save_pth))
    
#     train_dataset = Fusion_dataset('train')
#     print("the training dataset is length:{}".format(train_dataset.length))
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         drop_last=True,
#     )
#     train_loader.n_iter = len(train_loader)
#     # 
#     if num>0:
#         score_thres = 0.7
#         ignore_idx = 255
#         n_min = 8 * 640 * 480 // 8
#         criteria_p = OhemCELoss(
#             thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#         criteria_16 = OhemCELoss(
#             thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#     criteria_fusion = Fusionloss()

#     epoch = 10
#     st = glob_st = time.time()
#     logger.info('Training Fusion Model start~')
#     for epo in range(0, epoch):
#         # print('\n| epo #%s begin...' % epo)
#         lr_start = 0.001
#         lr_decay = 0.75
#         lr_this_epo = lr_start * lr_decay ** (epo - 1)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr_this_epo
#         for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
#             fusionmodel.train()
#             image_vis = Variable(image_vis).cuda()
#             image_vis_ycrcb = RGB2YCrCb(image_vis)
#             image_ir = Variable(image_ir).cuda()
#             label = Variable(label).cuda()
#             logits = fusionmodel(image_vis_ycrcb, image_ir)
#             fusion_ycrcb = torch.cat(
#                 (logits, image_vis_ycrcb[:, 1:2, :, :],
#                  image_vis_ycrcb[:, 2:, :, :]),
#                 dim=1,
#             )
#             fusion_image = YCrCb2RGB(fusion_ycrcb)

#             ones = torch.ones_like(fusion_image)
#             zeros = torch.zeros_like(fusion_image)
#             fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
#             fusion_image = torch.where(
#                 fusion_image < zeros, zeros, fusion_image)
#             lb = torch.squeeze(label, 1)
#             optimizer.zero_grad()
#             # seg loss
#             if num>0:
#                 out, mid = segmodel(fusion_image)
#                 lossp = criteria_p(out, lb)
#                 loss2 = criteria_16(mid, lb)
#                 seg_loss = lossp + 0.1 * loss2
#             # fusion loss
#             loss_fusion, loss_in, loss_grad = criteria_fusion(
#                 image_vis_ycrcb, image_ir, label, logits,num
#             )
#             if num>0:
#                 loss_total = loss_fusion + (num) * seg_loss
#             else:
#                 loss_total = loss_fusion
#             loss_total.backward()
#             optimizer.step()
#             ed = time.time()
#             t_intv, glob_t_intv = ed - st, ed - glob_st
#             now_it = train_loader.n_iter * epo + it + 1
#             eta = int((train_loader.n_iter * epoch - now_it)
#                       * (glob_t_intv / (now_it)))
#             eta = str(datetime.timedelta(seconds=eta))
#             if now_it % 10 == 0:
#                 if num>0:
#                     loss_seg=seg_loss.item()
#                 else:
#                     loss_seg=0
#                 msg = ', '.join(
#                     [
#                         'step: {it}/{max_it}',
#                         'loss_total: {loss_total:.4f}',
#                         'loss_in: {loss_in:.4f}',
#                         'loss_grad: {loss_grad:.4f}',
#                         'loss_seg: {loss_seg:.4f}',
#                         'eta: {eta}',
#                         'time: {time:.4f}',
#                     ]
#                 ).format(
#                     it=now_it,
#                     max_it=train_loader.n_iter * epoch,
#                     loss_total=loss_total.item(),
#                     loss_in=loss_in.item(),
#                     loss_grad=loss_grad.item(),
#                     loss_seg=loss_seg,
#                     time=t_intv,
#                     eta=eta,
#                 )
#                 logger.info(msg)
#                 st = ed
#     fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
#     torch.save(fusionmodel.state_dict(), fusion_model_file)
#     logger.info("Fusion Model Save to: {}".format(fusion_model_file))
#     logger.info('\n')

# # 替换后的train_fusion函数
# def train_fusion(num=0, logger=None, args=None, cfg=None):
#     lr_start = 0.001
#     modelpth = './model/Fusion/fusion_model.pth'
#     Method = cfg.method_name

#     fusionmodel = eval('FusionNet')(output=1)
#     fusionmodel.cuda()
#     fusionmodel.train()

#     if num > 0:
#         segmodel = BiSeNet(n_classes=cfg.num_classes)
#         segmodel.cuda()
#         segmodel.load_state_dict(torch.load('./model/Fusion/model_seg/model_final.pth'))
#         segmodel.eval()
#     else:
#         segmodel = None

#     optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
#     train_dataset = build_fusion_dataset(cfg, split='train', require_label=True)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     train_loader.n_iter = len(train_loader)
#     criteria_fusion = Fusionloss()
#     criteria_seg = OhemCELoss(thresh=0.75, n_min=args.batch_size * cfg.cropsize[0] * cfg.cropsize[1] // 16).cuda()

#     for epo in range(args.epochs):
#         for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
#             image_vis = Variable(image_vis).cuda()
#             image_vis_ycrcb = RGB2YCrCb(image_vis)
#             image_ir = Variable(image_ir).cuda()
#             label = Variable(label).cuda()

#             logits = fusionmodel(image_vis_ycrcb, image_ir)
#             fusion_y = logits
#             fusion_image = YCrCb2RGB(
#                 torch.cat(
#                     (fusion_y, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]),
#                     dim=1,
#                 )
#             )

#             optimizer.zero_grad()
#             loss_fusion, loss_in, loss_grad = criteria_fusion(
#                 image_vis_ycrcb, image_ir, label, logits, num
#             )

#             if num > 0 and segmodel is not None:
#                 out, mid = segmodel(fusion_image)
#                 lb = label.long()
#                 loss_seg = criteria_seg(out, lb)
#                 seg_loss_value = 0.1 * loss_seg
#             else:
#                 seg_loss_value = 0.0

#             loss_total = loss_fusion + seg_loss_value
#             loss_total.backward()
#             optimizer.step()

#             if logger is not None:
#                 logger.info(
#                     f'[fusion round {num}] epoch={epo}, iter={it}, '
#                     f'loss_total={loss_total.item():.6f}, '
#                     f'loss_fusion={loss_fusion.item():.6f}'
#                 )

#     torch.save(fusionmodel.state_dict(), modelpth)
#     fusionmodel.cpu()
#     if segmodel is not None:
#         segmodel.cpu()

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def train_fusion(num=0, logger=None, args=None, cfg=None):
    lr_start = 0.001
    modelpth = './model/Fusion/fusion_model.pth'
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    fusionmodel = FusionNet(output=1).to(device)
    fusionmodel.train()

    segmodel = None
    if num > 0:
        seg_path = './model/Fusion/model_seg/model_final.pth'
        segmodel = BiSeNet(n_classes=cfg.num_classes)
        segmodel.load_state_dict(torch.load(seg_path, map_location=device))
        segmodel = segmodel.to(device)
        segmodel.eval()

    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = build_fusion_dataset(cfg, split='train', require_label=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    criteria_fusion = Fusionloss()
    crop_w, crop_h = cfg.cropsize
    n_min = args.batch_size * crop_w * crop_h // 16
    criteria_seg = OhemCELoss(thresh=0.75, n_min=n_min, ignore_lb=cfg.ignore_index).to(device)

    for epo in range(args.epochs):
        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            image_vis = image_vis.to(device, non_blocking=True)
            image_ir = image_ir.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            image_vis_ycrcb = RGB2YCrCb(image_vis)
            logits = fusionmodel(image_vis_ycrcb, image_ir)
            fusion_y = logits
            fusion_image = YCrCb2RGB(
                torch.cat(
                    (fusion_y, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
            )

            optimizer.zero_grad()
            loss_fusion, loss_in, loss_grad = criteria_fusion(
                image_vis_ycrcb, image_ir, label, logits, num
            )

            seg_loss_value = 0.0
            if num > 0 and segmodel is not None:
                out, mid = segmodel(fusion_image)
                loss_seg = criteria_seg(out, label.long())
                seg_loss_value = 0.1 * loss_seg

            loss_total = loss_fusion + seg_loss_value
            loss_total.backward()
            optimizer.step()

            if logger is not None:
                logger.info(
                    f"[train_fusion] round={num}, epoch={epo}, iter={it}, "
                    f"loss_total={loss_total.item():.6f}, loss_fusion={loss_fusion.item():.6f}"
                )

    torch.save(fusionmodel.state_dict(), modelpth)
    fusionmodel.cpu()
    if segmodel is not None:
        segmodel.cpu()

# # 原run_fusion函数，保留以供参考，建议在其下方加入新的run_fusion函数以适配新的配置系统。
# def run_fusion(type='train'):
#     fusion_model_path = './model/Fusion/fusion_model.pth'
#     fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
#     os.makedirs(fused_dir, mode=0o777, exist_ok=True)
#     fusionmodel = eval('FusionNet')(output=1)
#     fusionmodel.eval()
#     if args.gpu >= 0:
#         fusionmodel.cuda(args.gpu)
#     fusionmodel.load_state_dict(torch.load(fusion_model_path))
#     print('done!')
#     test_dataset = Fusion_dataset(type)
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )
#     test_loader.n_iter = len(test_loader)
#     with torch.no_grad():
#         for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
#             images_vis = Variable(images_vis)
#             images_ir = Variable(images_ir)
#             labels = Variable(labels)
#             if args.gpu >= 0:
#                 images_vis = images_vis.cuda(args.gpu)
#                 images_ir = images_ir.cuda(args.gpu)
#                 labels = labels.cuda(args.gpu)
#             images_vis_ycrcb = RGB2YCrCb(images_vis)
#             logits = fusionmodel(images_vis_ycrcb, images_ir)
#             fusion_ycrcb = torch.cat(
#                 (logits, images_vis_ycrcb[:, 1:2, :,
#                  :], images_vis_ycrcb[:, 2:, :, :]),
#                 dim=1,
#             )
#             fusion_image = YCrCb2RGB(fusion_ycrcb)

#             ones = torch.ones_like(fusion_image)
#             zeros = torch.zeros_like(fusion_image)
#             fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
#             fusion_image = torch.where(
#                 fusion_image < zeros, zeros, fusion_image)
#             fused_image = fusion_image.cpu().numpy()
#             fused_image = fused_image.transpose((0, 2, 3, 1))
#             fused_image = (fused_image - np.min(fused_image)) / (
#                 np.max(fused_image) - np.min(fused_image)
#             )
#             fused_image = np.uint8(255.0 * fused_image)
#             for k in range(len(name)):
#                 image = fused_image[k, :, :, :]
#                 image = image.squeeze()
#                 image = Image.fromarray(image)
#                 save_path = os.path.join(fused_dir, name[k])
#                 image.save(save_path)
#                 print('Fusion {0} Sucessfully!'.format(save_path))

# # 替换后的run_fusion函数
# def run_fusion(type='train', logger=None, args=None, cfg=None):
#     fusionmodel = eval('FusionNet')(output=1)
#     fusionmodel.load_state_dict(torch.load('./model/Fusion/fusion_model.pth'))
#     fusionmodel.cuda()
#     fusionmodel.eval()

#     test_dataset = build_fusion_dataset(cfg, split=type, require_label=False)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )

#     fused_dir = get_fusion_output_dir(cfg, type, cfg.method_name)

#     with torch.no_grad():
#         for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
#             images_vis = Variable(images_vis).cuda()
#             images_vis_ycrcb = RGB2YCrCb(images_vis)
#             images_ir = Variable(images_ir).cuda()

#             logits = fusionmodel(images_vis_ycrcb, images_ir)
#             fusion_y = logits
#             fusion_image = YCrCb2RGB(
#                 torch.cat(
#                     (fusion_y, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
#                     dim=1,
#                 )
#             )

#             fusion_image = fusion_image.cpu().numpy()
#             for k in range(fusion_image.shape[0]):
#                 fi = fusion_image[k]
#                 fi = np.transpose(fi, (1, 2, 0))
#                 fi = np.clip(fi * 255.0, 0, 255).astype(np.uint8)
#                 save_name = name[k] if isinstance(name, list) else name
#                 save_path = os.path.join(fused_dir, save_name)
#                 Image.fromarray(fi).save(save_path)

#     fusionmodel.cpu()
#     if logger is not None:
#         logger.info(f'Fusion results saved to: {fused_dir}')

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def run_fusion(type='train', logger=None, args=None, cfg=None):
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    fusionmodel = FusionNet(output=1)
    fusionmodel.load_state_dict(torch.load('./model/Fusion/fusion_model.pth', map_location=device))
    fusionmodel = fusionmodel.to(device)
    fusionmodel.eval()

    test_dataset = build_fusion_dataset(cfg, split=type, require_label=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    fused_dir = get_fusion_output_dir(cfg, type, cfg.method_name)

    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            images_vis = images_vis.to(device, non_blocking=True)
            images_ir = images_ir.to(device, non_blocking=True)

            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_y = logits
            fusion_image = YCrCb2RGB(
                torch.cat(
                    (fusion_y, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
            )

            fusion_image = fusion_image.cpu().numpy()
            for k in range(fusion_image.shape[0]):
                fi = fusion_image[k]
                fi = np.transpose(fi, (1, 2, 0))
                fi = np.clip(fi * 255.0, 0, 255).astype(np.uint8)
                save_name = name[k] if isinstance(name, list) else name
                save_path = os.path.join(fused_dir, save_name)
                Image.fromarray(fi).save(save_path)

    fusionmodel.cpu()
    if logger is not None:
        logger.info(f"Fusion results saved to: {fused_dir}")

# 原主入口函数，保留以供参考，建议在其下方加入新的主入口函数以适配新的配置系统。
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train with pytorch')
#     parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
#     parser.add_argument('--batch_size', '-B', type=int, default=16)
#     parser.add_argument('--gpu', '-G', type=int, default=0)
#     parser.add_argument('--num_workers', '-j', type=int, default=8)
#     args = parser.parse_args()
#     # modelpth = './model'
#     # Method = 'Fusion'
#     # modelpth = os.path.join(modelpth, Method)
#     logpath='./logs'
#     logger = logging.getLogger()
#     setup_logger(logpath)
#     for i in range(4):
#         train_fusion(i, logger, args)  
#         print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
#         run_fusion('train')  
#         print("|{0} Fusion Image Sucessfully~!".format(i + 1))
#         train_seg(i, logger, args)
#         print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
#     print("training Done!")

# # 新主入口函数
# if __name__ == '__main__':
#     args = parse_args()
#     cfg = get_dataset_config(args.dataset_name, args.dataset_root)

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     os.makedirs('./logs', exist_ok=True)
#     fh = logging.FileHandler('./logs/train.log')
#     logger.addHandler(fh)

#     for i in range(4):
#         train_fusion(num=i, logger=logger, args=args, cfg=cfg)
#         run_fusion(type='train', logger=logger, args=args, cfg=cfg)
#         train_seg(i=i, logger=logger, args=args, cfg=cfg)

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset_name, args.dataset_root)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs('./logs', exist_ok=True)
    fh = logging.FileHandler('./logs/train.log')
    logger.addHandler(fh)

    for i in range(4):
        train_fusion(num=i, logger=logger, args=args, cfg=cfg)
        run_fusion(type='train', logger=logger, args=args, cfg=cfg)
        train_seg(i=i, logger=logger, args=args, cfg=cfg)