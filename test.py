# 原始版本test.py，适配旧版 MSRS 数据集和 TaskFusion_dataset.py 中的修改。
# # coding:utf-8
# import os
# import argparse
# from utils import *
# import torch
# from torch.utils.data import DataLoader
# from datasets import Fusion_dataset
# from FusionNet import FusionNet
# from tqdm import tqdm

# # To run, set the fused_dir, and the val path in the TaskFusionDataset.py
# def main(ir_dir='./test_imgs/ir', vi_dir='./test_imgs/vi', save_dir='./SeAFusion', fusion_model_path='./model/Fusion/fusionmodel_final.pth'):
#     fusionmodel = FusionNet(output=1)
#     device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
#     fusionmodel.load_state_dict(torch.load(fusion_model_path))
#     fusionmodel = fusionmodel.to(device)
#     print('fusionmodel load done!')
#     test_dataset = Fusion_dataset('val', ir_path=ir_dir, vi_path=vi_dir)
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )
#     test_loader.n_iter = len(test_loader)
#     test_bar = tqdm(test_loader)
#     with torch.no_grad():
#         for it, (img_vis, img_ir, name) in enumerate(test_bar):
#             img_vis = img_vis.to(device)
#             img_ir = img_ir.to(device)
#             vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
#             vi_Y = vi_Y.to(device)
#             vi_Cb = vi_Cb.to(device)
#             vi_Cr = vi_Cr.to(device)
#             fused_img = fusionmodel(vi_Y, img_ir)
#             fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
#             for k in range(len(name)):
#                 img_name = name[k]
#                 save_path = os.path.join(save_dir, img_name)
#                 save_img_single(fused_img[k, ::], save_path)
#                 test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
#     ## model
#     parser.add_argument('--model_path', '-M', type=str, default='./model/Fusion/fusionmodel_final.pth')
#     ## dataset
#     parser.add_argument('--ir_dir', '-ir_dir', type=str, default='./test_imgs/ir')
#     parser.add_argument('--vi_dir', '-vi_dir', type=str, default='./test_imgs/vi')
#     parser.add_argument('--save_dir', '-save_dir', type=str, default='./SeAFusion')
#     parser.add_argument('--batch_size', '-B', type=int, default=1)
#     parser.add_argument('--gpu', '-G', type=int, default=0)
#     parser.add_argument('--num_workers', '-j', type=int, default=8)
#     args = parser.parse_args()
#     os.makedirs(args.save_dir, exist_ok=True)
#     print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
#     main(ir_dir=args.ir_dir, vi_dir=args.vi_dir, save_dir=args.save_dir, fusion_model_path=args.model_path)

# 新版 FLAME3 版 test.py，适配 datasets/flame3_seg_dataset.py 和 configs/flame3_config.py 中的修改。
# -*- coding: utf-8 -*-
import os
import argparse
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from FusionNet import FusionNet
from datasets import (
    get_dataset_config,
    build_fusion_dataset,
    get_fusion_output_dir,
)


def RGB2YCrCb(input_im: torch.Tensor) -> torch.Tensor:
    """
    输入:
        input_im: [B, 3, H, W], RGB, range [0,1]
    输出:
        [B, 3, H, W], 通道顺序为 [Y, Cr, Cb]
    """
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)

    r = im_flat[:, 0]
    g = im_flat[:, 1]
    b = im_flat[:, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5

    y = torch.unsqueeze(y, 1)
    cr = torch.unsqueeze(cr, 1)
    cb = torch.unsqueeze(cb, 1)

    temp = torch.cat((y, cr, cb), dim=1).to(device)
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


def YCrCb2RGB(input_im: torch.Tensor) -> torch.Tensor:
    """
    输入:
        input_im: [B, 3, H, W], 通道顺序 [Y, Cr, Cb]
    输出:
        [B, 3, H, W], RGB, 理论值范围近似 [0,1]
    """
    device = input_im.device
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)

    mat = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.403, -0.714, 0.0],
            [0.0, -0.344, 1.773],
        ],
        device=device,
        dtype=im_flat.dtype,
    )
    bias = torch.tensor([0.0 / 255, -0.5, -0.5], device=device, dtype=im_flat.dtype)

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


def save_rgb_tensor_image(img_tensor: torch.Tensor, save_path: str) -> None:
    """
    输入:
        img_tensor: [3, H, W], float tensor, range约在[0,1]
        save_path: 保存路径
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def unpack_batch(batch):
    """
    兼容两种可能的数据集返回:
    1) FLAME3模块化版: (image_vis, image_ir, label, name)
    2) 某些旧版测试集: (image_vis, image_ir, name)
    """
    if len(batch) == 4:
        img_vis, img_ir, _, name = batch
        return img_vis, img_ir, name
    if len(batch) == 3:
        img_vis, img_ir, name = batch
        return img_vis, img_ir, name
    raise ValueError(f"Unsupported batch format, len(batch)={len(batch)}")


def build_test_loader(args, cfg):
    """
    构造测试 dataloader。
    对 FLAME3，默认从 cfg.root 下的 Visible/Infrared/(Label) 读取。
    """
    test_dataset = build_fusion_dataset(
        cfg=cfg,
        split=args.split,
        require_label=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_loader


def resolve_save_dir(args, cfg) -> str:
    """
    若用户手动指定 save_dir，则优先使用。
    否则按 builder 规则写入数据集内部 Fusion/<split>/。
    """
    if args.save_dir is not None and len(args.save_dir.strip()) > 0:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    return get_fusion_output_dir(cfg, args.split, cfg.method_name)


def load_model(args, device: torch.device) -> torch.nn.Module:
    fusion_model = FusionNet(output=1)
    state = torch.load(args.model_path, map_location=device)
    fusion_model.load_state_dict(state)
    fusion_model = fusion_model.to(device)
    fusion_model.eval()
    return fusion_model


def main(args):
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    cfg = get_dataset_config(args.dataset_name, args.dataset_root)
    save_dir = resolve_save_dir(args, cfg)
    test_loader = build_test_loader(args, cfg)
    fusion_model = load_model(args, device)

    print(f"| testing {args.dataset_name} on device: {device}")
    print(f"| model path: {args.model_path}")
    print(f"| save dir  : {save_dir}")
    print(f"| split     : {args.split}")
    print(f"| dataset size: {len(test_loader.dataset)}")

    test_bar = tqdm(test_loader)

    with torch.no_grad():
        for batch in test_bar:
            img_vis, img_ir, name = unpack_batch(batch)

            img_vis = img_vis.to(device, non_blocking=True)
            img_ir = img_ir.to(device, non_blocking=True)

            # 与修改后的 train.py 保持一致:
            # 先转 YCrCb，再把完整3通道张量送入 FusionNet，
            # FusionNet 内部实际只取可见光的第1通道(Y)。
            img_vis_ycrcb = RGB2YCrCb(img_vis)
            fused_y = fusion_model(img_vis_ycrcb, img_ir)

            fused_ycrcb = torch.cat(
                (
                    fused_y,
                    img_vis_ycrcb[:, 1:2, :, :],
                    img_vis_ycrcb[:, 2:3, :, :],
                ),
                dim=1,
            )
            fused_rgb = YCrCb2RGB(fused_ycrcb)
            fused_rgb = torch.clamp(fused_rgb, min=0.0, max=1.0)

            # name 可能是 list[str]，也可能是 tuple[str]
            names = list(name) if not isinstance(name, list) else name

            for k, img_name in enumerate(names):
                save_path = os.path.join(save_dir, img_name)
                save_rgb_tensor_image(fused_rgb[k], save_path)
                test_bar.set_description(f"Fusion {img_name} successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SeAFusion on FLAME3 or MSRS")

    # 模型
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/Fusion/fusion_model.pth",
        help="Path to trained fusion model",
    )

    # 数据集
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="flame3",
        choices=["flame3", "msrs"],
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./FLAME3",
        help="Dataset root path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to run inference on",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Optional manual output directory; if empty, use dataset Fusion/<split>/",
    )

    # 运行参数
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)

    # ==============================
# 兼容补丁说明（保留原代码）
# ==============================
#
# 当前 test.py 已基本兼容 Python 3.12 / PyTorch 2.8 / CUDA 12.8
# 若仍报错，优先检查以下几项：
# 1. configs/flame3_config.py 的 dataset_name 是否与 builder.py 分支一致
# 2. dataset_root 是否包含 Visible / Infrared / Label / Fusion 完整目录
# 3. model_path 是否存在，且对应的是新的 fusion_model.pth
# 4. FLAME3 图像尺寸是否在 dataloader 中保持 RGB / IR / Label 三者一致
# 5. 若 split='test' 且无标签，请确认 synthesize_dummy_label=True