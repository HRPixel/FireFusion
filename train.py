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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='flame3', type=str, help='flame3 or msrs')
    parser.add_argument('--dataset_root', default='./FLAME3', type=str, help='dataset root path')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader workers')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs or outer loop iterations')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id, use -1 for cpu')
    return parser.parse_args()

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

def train_seg(i=0, logger=None, args=None, cfg=None):
    save_pth = osp.join('./model/Fusion/model_seg')
    os.makedirs(save_pth, exist_ok=True)
    load_path = osp.join(save_pth, 'model_final.pth')

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    n_classes = cfg.num_classes
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

    it_start = i * 20000

    optim = Optimizer(
        model=net,
        lr0=1e-2,
        momentum=0.9,
        wd=5e-4,
        warmup_steps=1000,
        warmup_start_lr=1e-5,
        max_iter=80000,
        power=0.9,
        it=it_start,
    )

    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()

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
        loss = loss_p + 0.75 * loss_16
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        if (it + 1) % msg_iter == 0:
            ed = time.time()
            t_intv = ed - st
            glob_t_intv = ed - glob_st
            eta_sec = int((20000 - (it + 1)) * (glob_t_intv / (it + 1)))
            eta = str(datetime.timedelta(seconds=eta_sec))

            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:.6f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=20000,
                lr=optim.lr,
                loss=sum(loss_avg) / len(loss_avg),
                eta=eta,
                time=t_intv,
            )

            if logger is not None:
                logger.info(msg)

            loss_avg = []
            st = ed

    torch.save(net.state_dict(), load_path)
    if logger is not None:
        logger.info(f'Segmentation Model Training done~, The Model is saved to: {load_path}')
        logger.info('\n')

    net.cpu()

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
    train_loader.n_iter = len(train_loader)

    criteria_fusion = Fusionloss()
    crop_w, crop_h = cfg.cropsize
    n_min = args.batch_size * crop_w * crop_h // 16
    criteria_seg = OhemCELoss(
        thresh=0.75,
        n_min=n_min,
        ignore_lb=cfg.ignore_index
    ).to(device)

    msg_iter = 10
    loss_total_avg = []
    loss_fusion_avg = []
    loss_seg_avg = []
    st = glob_st = time.time()

    if logger is not None:
        logger.info(f"Training Fusion Model start~ [round={num}]")

    for epo in range(args.epochs):
        lr_this_epo = lr_start * (0.75 ** max(epo - 1, 0))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            fusionmodel.train()

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

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            optimizer.zero_grad()
            loss_fusion, loss_in, loss_grad = criteria_fusion(
                image_vis_ycrcb, image_ir, label, logits, num
            )

            seg_loss_value = torch.tensor(0.0, device=device)
            if num > 0 and segmodel is not None:
                out, mid = segmodel(fusion_image)
                loss_seg = criteria_seg(out, label.long())
                seg_loss_value = 0.1 * loss_seg

            loss_total = loss_fusion + seg_loss_value
            loss_total.backward()
            optimizer.step()

            loss_total_avg.append(loss_total.item())
            loss_fusion_avg.append(loss_fusion.item())
            loss_seg_avg.append(seg_loss_value.item())

            now_it = train_loader.n_iter * epo + it + 1
            max_it = train_loader.n_iter * args.epochs

            if now_it % msg_iter == 0:
                ed = time.time()
                t_intv = ed - st
                glob_t_intv = ed - glob_st
                eta_sec = int((max_it - now_it) * (glob_t_intv / max(now_it, 1)))
                eta = str(datetime.timedelta(seconds=eta_sec))

                msg = ', '.join([
                    'step: {it}/{max_it}',
                    'lr: {lr:.6f}',
                    'loss_total: {loss_total:.4f}',
                    'loss_fusion: {loss_fusion:.4f}',
                    'loss_seg: {loss_seg:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it=now_it,
                    max_it=max_it,
                    lr=optimizer.param_groups[0]['lr'],
                    loss_total=sum(loss_total_avg) / len(loss_total_avg),
                    loss_fusion=sum(loss_fusion_avg) / len(loss_fusion_avg),
                    loss_seg=sum(loss_seg_avg) / len(loss_seg_avg),
                    eta=eta,
                    time=t_intv,
                )

                if logger is not None:
                    logger.info(msg)

                loss_total_avg = []
                loss_fusion_avg = []
                loss_seg_avg = []
                st = ed

    torch.save(fusionmodel.state_dict(), modelpth)
    if logger is not None:
        logger.info(f"Fusion Model Save to: {modelpth}")
        logger.info('\n')

    fusionmodel.cpu()
    if segmodel is not None:
        segmodel.cpu()

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

if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset_name, args.dataset_root)

    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    logger.info('Training start!')
    logger.info(f'dataset_name: {cfg.dataset_name}')
    logger.info(f'dataset_root: {cfg.root}')
    logger.info(f'cropsize: {cfg.cropsize}')
    logger.info(f'num_classes: {cfg.num_classes}')
    logger.info('\n')

    for i in range(4):
        logger.info(f'========== Round {i + 1}/4 : train_fusion ==========')
        train_fusion(i, logger, args, cfg)
        print("|{0} Train Fusion Model Successfully~!".format(i + 1))

        logger.info(f'========== Round {i + 1}/4 : run_fusion(train) ==========')
        run_fusion('train', logger, args, cfg)
        print("|{0} Fusion Image Successfully~!".format(i + 1))

        logger.info(f'========== Round {i + 1}/4 : train_seg ==========')
        train_seg(i, logger, args, cfg)
        print("|{0} Train Segmentation Model Successfully~!".format(i + 1))

    logger.info("Training Done!")
    print("training Done!")