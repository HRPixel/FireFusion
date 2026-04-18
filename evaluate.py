# 原evaluate.py文件，评测脚本，包含了多尺度评测的实现，以及结果的可视化和mIoU的计算。

# #!/usr/bin/python
# # -*- encoding: utf-8 -*-

# from logger import setup_logger
# from model_TII import BiSeNet
# from MSRS import MSRS
# # import cv2
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.distributed as dist
# from sklearn.metrics import confusion_matrix
# import os
# import os.path as osp
# import logging
# import numpy as np
# from tqdm import tqdm
# import math
# from PIL import Image
# from utils import SegmentationMetric

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# class MscEval(object):
#     def __init__(
#         self,
#         model,
#         dataloader,
#         # scales=[1],
#         scales=[0.75, 0.9, 1, 1.1, 1.2, 1.25],
#         # scales=[1, 1.2],
#         n_classes=9,
#         lb_ignore=255,
#         cropsize=480,
#         flip=True,
#         *args,
#         **kwargs
#     ):
#         self.scales = scales
#         self.n_classes = n_classes
#         self.lb_ignore = lb_ignore
#         self.flip = flip
#         self.cropsize = cropsize
#         ## dataloader
#         self.dl = dataloader
#         self.net = model
#         print(self.scales)

#     def pad_tensor(self, inten, size):
#         N, C, H, W = inten.size()
#         outten = torch.zeros(N, C, size[0], size[1]).cuda()
#         outten.requires_grad = False
#         margin_h, margin_w = size[0] - H, size[1] - W
#         hst, hed = margin_h // 2, margin_h // 2 + H
#         wst, wed = margin_w // 2, margin_w // 2 + W
#         outten[:, :, hst:hed, wst:wed] = inten
#         return outten, [hst, hed, wst, wed]

#     def get_palette(self):
#         unlabelled = [0, 0, 0]
#         car = [64, 0, 128]
#         person = [64, 64, 0]
#         bike = [0, 128, 192]
#         curve = [0, 0, 192]
#         car_stop = [128, 128, 0]
#         guardrail = [64, 64, 128]
#         color_cone = [192, 128, 128]
#         bump = [192, 64, 0]
#         palette = np.array(
#             [
#                 unlabelled,
#                 car,
#                 person,
#                 bike,
#                 curve,
#                 car_stop,
#                 guardrail,
#                 color_cone,
#                 bump,
#             ]
#         )
#         return palette

#     def visualize(self, save_name, predictions):
#         palette = self.get_palette()
#         # print(predictions.shape)
#         # 遍历predictions
#         # for (i, pred) in enumerate(predictions):
#         pred = predictions
#         img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#         for cid in range(1, int(predictions.max())):
#             img[pred == cid] = palette[cid]
#         img = Image.fromarray(np.uint8(img))
#         img.save(save_name)

#     def eval_chip(self, crop):
#         with torch.no_grad():
#             out = self.net(crop)[0]
#             prob = F.softmax(out, 1)
#             if self.flip:
#                 crop = torch.flip(crop, dims=(3,))
#                 out = self.net(crop)[0]
#                 out = torch.flip(out, dims=(3,))
#                 prob += F.softmax(out, 1)
#             prob = torch.exp(prob)
#         return prob

#     def crop_eval(self, im):
#         cropsize = self.cropsize
#         stride_rate = 5 / 6.0
#         N, C, H, W = im.size()
#         long_size, short_size = (H, W) if H > W else (W, H)
#         if long_size < cropsize:
#             im, indices = self.pad_tensor(im, (cropsize, cropsize))
#             prob = self.eval_chip(im)
#             prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
#         else:
#             stride = math.ceil(cropsize * stride_rate)
#             if short_size < cropsize:
#                 if H < W:
#                     im, indices = self.pad_tensor(im, (cropsize, W))
#                 else:
#                     im, indices = self.pad_tensor(im, (H, cropsize))
#             N, C, H, W = im.size()
#             n_x = math.ceil((W - cropsize) / stride) + 1
#             n_y = math.ceil((H - cropsize) / stride) + 1
#             prob = torch.zeros(N, self.n_classes, H, W).cuda()
#             prob.requires_grad = False
#             for iy in range(n_y):
#                 for ix in range(n_x):
#                     hed, wed = (
#                         min(H, stride * iy + cropsize),
#                         min(W, stride * ix + cropsize),
#                     )
#                     hst, wst = hed - cropsize, wed - cropsize
#                     chip = im[:, :, hst:hed, wst:wed]
#                     prob_chip = self.eval_chip(chip)
#                     prob[:, :, hst:hed, wst:wed] += prob_chip
#             if short_size < cropsize:
#                 prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
#         return prob

#     def scale_crop_eval(self, im, scale):
#         N, C, H, W = im.size()
#         new_hw = [int(H * scale), int(W * scale)]
#         im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
#         prob = self.crop_eval(im)
#         prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
#         return prob

#     def compute_hist(self, pred, lb):
#         n_classes = self.n_classes
#         ignore_idx = self.lb_ignore
#         keep = np.logical_not(lb == ignore_idx)
#         merge = pred[keep] * n_classes + lb[keep]
#         hist = np.bincount(merge, minlength=n_classes ** 2)
#         hist = hist.reshape((n_classes, n_classes))
#         return hist

#     def evaluate(self, Method='NestFuse'):
#         ## evaluate
#         n_classes = self.n_classes
#         hist = np.zeros((n_classes, n_classes), dtype=np.float32)
#         device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
#         lb_ignore = [255]
#         seg_metric = SegmentationMetric(n_classes, device=device)          
#         dloader = tqdm(self.dl)
#         if dist.is_initialized() and not dist.get_rank() == 0:
#             dloader = self.dl
#         for i, (imgs, label, fn) in enumerate(dloader):
#             # if not fn[0] == '00037N.png':
#             #     continue
#             # print(fn[0])
#             N, _, H, W = label.shape
#             probs = torch.zeros((N, self.n_classes, H, W))
#             probs.requires_grad = False
#             imgs = imgs.cuda()
#             probs = torch.zeros((N, self.n_classes, H, W))
#             probs.requires_grad = False
#             probs_torch = torch.zeros((N, self.n_classes, H, W))
#             probs_torch = probs_torch.to(device)
#             probs_torch.requires_grad = False
#             imgs = imgs.cuda()
#             for sc in self.scales:
#                 prob = self.scale_crop_eval(imgs, sc)
#                 probs_torch += prob
#                 probs += prob.detach().cpu()            
#             seg_results = torch.argmax(probs_torch, dim=1, keepdim=True)
#             seg_metric.addBatch(seg_results, label.to(device), lb_ignore)
#             probs = probs.data.numpy()
#             preds = np.argmax(probs, axis=1)
#             for i in range(1):
#                 outpreds = preds[i]
#                 name = fn[i]
#                 folder_path = os.path.join('BANet', Method)
#                 if not os.path.exists(folder_path):
#                     os.makedirs(folder_path)
#                 file_path = os.path.join(folder_path, name)
#                 # img = Image.fromarray(np.uint8(outpreds))
#                 # img.save(file_path)
#                 self.visualize(file_path, outpreds)
#             hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
#             hist = hist + hist_once
#         IOUs = np.diag(hist) / (
#             np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist)
#         )
#         mIOU = np.mean(IOUs)
#         mIOU = mIOU
#         mIoU_torch = np.array(seg_metric.meanIntersectionOverUnion().item())
#         IoU_list = IOUs.tolist()
#         IoU_list.append(mIOU)
#         IoU_list = [round(100 * i, 2) for i in IoU_list]
#         # print('{} | IoU:{}, mIoU:{:.4f}'.format(Method, IoU_list, mIoU_torch))
#         return mIOU, IoU_list


# def evaluate(respth='./res_MSRS', dspth='/data/timer/BiSeNet/MSRS', Method=None, save_pth=None):
#     ## logger
#     logger = logging.getLogger()
#     respth = os.path.join(respth, Method)
#     ## model
#     logger.info('\n')
#     logger.info('====' * 4)
#     logger.info('evaluating the model ...\n')
#     # logger.info('setup and restore model')
#     n_classes = 9
#     net = BiSeNet(n_classes=n_classes)
#     if save_pth==None:
#         save_pth = osp.join(respth, 'model_final.pth')
#     net.load_state_dict(torch.load(save_pth))
#     # torch.save(net.state_dict(), save_pth, file_serialization=False)
#     net.cuda()
#     net.eval()

#     ## dataset
#     batchsize = 1
#     n_workers = 2
#     dsval = MSRS(dspth, mode='test', Method=Method)
#     dl = DataLoader(
#         dsval,
#         batch_size=batchsize,
#         shuffle=False,
#         num_workers=n_workers,
#         drop_last=False,
#     )

#     ## evaluator
#     logger.info('compute the mIOU')
#     evaluator = MscEval(net, dl)

#     ## eval
#     mIOU, IoU_list = evaluator.evaluate(Method=Method)
#     logger.info('{} | IoU:{}, mIoU:{:.4f}'.format(Method, IoU_list, mIOU))
#     return mIOU


# if __name__ == "__main__":
#     # setup_logger('./res_Infrared')
#     save_name = 'SeAFusion'
#     Method = 'SeAFusion'
#     respth = './model'
#     log_dir = os.path.join(respth, save_name)
#     os.makedirs(log_dir, exist_ok=True)
#     datapth='./datasets/MSRS'
#     setup_logger(log_dir)
#     evaluate(respth=respth, dspth=datapth, Method=Method, save_pth='./model/Fusion/model_final.pth')

# 新evaluate.py

# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import math
import logging
import argparse
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from model_TII import BiSeNet
from utils import SegmentationMetric
from logger import setup_logger

from datasets import (
    get_dataset_config,
    build_seg_dataset,
)


class MscEval(object):
    def __init__(
        self,
        model,
        dataloader,
        n_classes=3,
        lb_ignore=255,
        cropsize=512,
        flip=True,
        scales: Optional[List[float]] = None,
        palette: Optional[np.ndarray] = None,
        pred_save_dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.scales = scales if scales is not None else [0.75, 0.9, 1.0, 1.1, 1.2, 1.25]
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        self.dl = dataloader
        self.net = model
        self.palette = palette
        self.pred_save_dir = pred_save_dir

        if self.pred_save_dir is not None:
            os.makedirs(self.pred_save_dir, exist_ok=True)

        print("Eval scales:", self.scales)

    def pad_tensor(self, inten, size):
        """
        inten: [N, C, H, W]
        size : (target_h, target_w)
        """
        n, c, h, w = inten.size()
        device = inten.device
        outten = torch.zeros(n, c, size[0], size[1], device=device, dtype=inten.dtype)
        outten.requires_grad = False

        margin_h, margin_w = size[0] - h, size[1] - w
        hst, hed = margin_h // 2, margin_h // 2 + h
        wst, wed = margin_w // 2, margin_w // 2 + w

        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def get_default_palette(self) -> np.ndarray:
        """
        若没有 info_json，则回退到一个通用调色板。
        前 4 类优先覆盖 FLAME3:
            0 background -> black
            1 flame      -> red
            2 smoke      -> gray
            3 ignore/other -> white
        其余类循环补色。
        """
        palette = np.array(
            [
                [0, 0, 0],         # 0 background
                [255, 0, 0],       # 1 flame
                [128, 128, 128],   # 2 smoke
                [255, 255, 255],   # 3 other/ignore
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [128, 64, 0],
                [64, 0, 128],
                [64, 64, 0],
            ],
            dtype=np.uint8,
        )
        return palette

    def visualize(self, save_name, prediction):
        """
        prediction: [H, W]
        """
        palette = self.palette if self.palette is not None else self.get_default_palette()
        pred = prediction.astype(np.int64)

        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        max_color_idx = min(len(palette) - 1, int(pred[pred != self.lb_ignore].max()) if np.any(pred != self.lb_ignore) else 0)
        for cid in range(0, max_color_idx + 1):
            img[pred == cid] = palette[cid]

        # ignore 区域单独置白
        img[pred == self.lb_ignore] = np.array([255, 255, 255], dtype=np.uint8)

        Image.fromarray(img).save(save_name)

    def eval_chip(self, crop):
        """
        crop: [N, C, H, W]
        """
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, dim=1)

            if self.flip:
                crop_flip = torch.flip(crop, dims=(3,))
                out_flip = self.net(crop_flip)[0]
                out_flip = torch.flip(out_flip, dims=(3,))
                prob += F.softmax(out_flip, dim=1)

            # 与原实现保持近似逻辑
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        n, c, h, w = im.size()
        long_size, short_size = (h, w) if h > w else (w, h)

        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
            return prob

        stride = math.ceil(cropsize * stride_rate)

        if short_size < cropsize:
            if h < w:
                im, indices = self.pad_tensor(im, (cropsize, w))
            else:
                im, indices = self.pad_tensor(im, (h, cropsize))

        n, c, h, w = im.size()
        n_x = math.ceil((w - cropsize) / stride) + 1
        n_y = math.ceil((h - cropsize) / stride) + 1

        prob = torch.zeros(n, self.n_classes, h, w, device=im.device, dtype=torch.float32)
        prob.requires_grad = False

        for iy in range(n_y):
            for ix in range(n_x):
                hed = min(h, stride * iy + cropsize)
                wed = min(w, stride * ix + cropsize)
                hst = hed - cropsize
                wst = wed - cropsize
                chip = im[:, :, hst:hed, wst:wed]
                prob_chip = self.eval_chip(chip)
                prob[:, :, hst:hed, wst:wed] += prob_chip

        if short_size < cropsize:
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]

        return prob

    def scale_crop_eval(self, im, scale):
        """
        im: [N, C, H, W]
        """
        n, c, h, w = im.size()
        new_hw = [int(h * scale), int(w * scale)]
        im = F.interpolate(im, new_hw, mode="bilinear", align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (h, w), mode="bilinear", align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        """
        pred: [N, H, W]
        lb  : [N, H, W]
        """
        n_classes = self.n_classes
        keep = np.logical_not(lb == self.lb_ignore)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self, method_name="Fusion"):
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float64)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seg_metric = SegmentationMetric(n_classes, device=device)
        dloader = tqdm(self.dl)

        if dist.is_initialized() and dist.get_rank() != 0:
            dloader = self.dl

        for batch in dloader:
            imgs, label, fn = batch
            # label shape should be [N,1,H,W]
            n, _, h, w = label.shape

            imgs = imgs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            probs_torch = torch.zeros((n, self.n_classes, h, w), device=device, dtype=torch.float32)
            probs_torch.requires_grad = False

            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs_torch += prob

            seg_results = torch.argmax(probs_torch, dim=1, keepdim=True)  # [N,1,H,W]
            seg_metric.addBatch(seg_results, label, [self.lb_ignore])

            preds = seg_results.detach().cpu().numpy().squeeze(1)   # [N,H,W]
            lbs = label.detach().cpu().numpy().squeeze(1)           # [N,H,W]

            hist_once = self.compute_hist(preds, lbs)
            hist += hist_once

            if self.pred_save_dir is not None:
                names = list(fn) if not isinstance(fn, list) else fn
                for i in range(len(names)):
                    save_path = os.path.join(self.pred_save_dir, names[i])
                    self.visualize(save_path, preds[i])

        ious = np.diag(hist) / (
            np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist) + 1e-12
        )
        miou_np = np.nanmean(ious)

        miou_torch = seg_metric.meanIntersectionOverUnion().item()
        iou_list = [round(100.0 * x, 2) for x in ious.tolist()]
        miou_percent = round(100.0 * miou_np, 2)

        return {
            "mIoU_numpy": miou_np,
            "mIoU_torch": miou_torch,
            "IoU_percent_list": iou_list,
            "mIoU_percent": miou_percent,
        }


def load_palette_from_info_json(info_json_path: Optional[str], fallback_num_classes: int) -> np.ndarray:
    """
    从 info_json 中读取颜色。若文件不存在则回退默认调色板。
    """
    default_palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [128, 128, 128],
            [255, 255, 255],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 64, 0],
            [64, 0, 128],
            [64, 64, 0],
        ],
        dtype=np.uint8,
    )

    if info_json_path is None or (not os.path.isfile(info_json_path)):
        return default_palette

    with open(info_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_train_id = 0
    for item in data:
        train_id = int(item.get("trainId", item.get("id", 0)))
        if train_id != 255:
            max_train_id = max(max_train_id, train_id)

    palette_len = max(max_train_id + 1, fallback_num_classes)
    palette = np.zeros((palette_len, 3), dtype=np.uint8)

    for item in data:
        train_id = int(item.get("trainId", item.get("id", 0)))
        color = item.get("color", None)
        if color is None:
            continue
        if train_id == 255:
            continue
        if 0 <= train_id < palette_len:
            palette[train_id] = np.array(color, dtype=np.uint8)

    # 若某些类颜色未填，补默认色
    for i in range(min(len(default_palette), palette_len)):
        if np.all(palette[i] == 0) and i != 0:
            palette[i] = default_palette[i]

    return palette


def evaluate(
    dataset_name="flame3",
    dataset_root="./FLAME3",
    split="test",
    method_name="Fusion",
    model_path="./model/Fusion/model_seg/model_final.pth",
    pred_save_dir=None,
    log_dir="./eval_logs",
    cropsize=512,
    scales=None,
    flip=True,
    batch_size=1,
    num_workers=2,
    gpu=0,
):
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)

    logger = logging.getLogger()
    logger.info("")
    logger.info("====" * 4)
    logger.info("Evaluating segmentation model ...")

    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu"
    )

    cfg = get_dataset_config(dataset_name, dataset_root)

    logger.info(f"dataset_name: {cfg.dataset_name}")
    logger.info(f"dataset_root: {cfg.root}")
    logger.info(f"split       : {split}")
    logger.info(f"method_name : {method_name}")
    logger.info(f"model_path  : {model_path}")
    logger.info(f"num_classes : {cfg.num_classes}")
    logger.info(f"ignore_index: {cfg.ignore_index}")

    net = BiSeNet(n_classes=cfg.num_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()

    dsval = build_seg_dataset(cfg, split=split, method=method_name)
    dl = DataLoader(
        dsval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    palette = load_palette_from_info_json(
        getattr(cfg, "info_json", None),
        fallback_num_classes=cfg.num_classes,
    )

    evaluator = MscEval(
        model=net,
        dataloader=dl,
        scales=scales if scales is not None else [0.75, 0.9, 1.0, 1.1, 1.2, 1.25],
        n_classes=cfg.num_classes,
        lb_ignore=cfg.ignore_index,
        cropsize=cropsize,
        flip=flip,
        palette=palette,
        pred_save_dir=pred_save_dir,
    )

    results = evaluator.evaluate(method_name=method_name)

    logger.info(
        f"{method_name} | IoU(%): {results['IoU_percent_list']}, "
        f"mIoU(%): {results['mIoU_percent']:.2f}, "
        f"mIoU(torch): {results['mIoU_torch']:.6f}"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on FLAME3 or MSRS")

    parser.add_argument("--dataset_name", type=str, default="flame3", choices=["flame3", "msrs"])
    parser.add_argument("--dataset_root", type=str, default="./FLAME3")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--method_name", type=str, default="Fusion")

    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/Fusion/model_seg/model_final.pth",
        help="Path to trained segmentation model",
    )
    parser.add_argument(
        "--pred_save_dir",
        type=str,
        default="./eval_preds/Fusion",
        help="Directory to save colorized prediction maps",
    )
    parser.add_argument("--log_dir", type=str, default="./eval_logs/Fusion")

    parser.add_argument("--cropsize", type=int, default=480)
    parser.add_argument("--flip", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)

    # 可通过命令行传入多尺度，例如:
    # --scales 0.75 0.9 1.0 1.1 1.25
    parser.add_argument("--scales", nargs="+", type=float, default=[0.75, 0.9, 1.0, 1.1, 1.2, 1.25])

    args = parser.parse_args()

    evaluate(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        split=args.split,
        method_name=args.method_name,
        model_path=args.model_path,
        pred_save_dir=args.pred_save_dir,
        log_dir=args.log_dir,
        cropsize=args.cropsize,
        scales=args.scales,
        flip=args.flip,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu=args.gpu,
    )