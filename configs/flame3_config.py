# configs/flame3_config.py

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig:
    # ---- 基本标识 ----
    dataset_name: str              # 'flame3'
    root: str                      # './FLAME3'

    # ---- 目录命名 ----
    visible_dir: str               # 'Visible'
    infrared_dir: str              # 'Infrared'
    label_dir: str                 # 'Label'
    fusion_dir: str                # 'Fusion'
    method_name: str               # 'Fusion'

    # ---- 标签与类别 ----
    num_classes: int               # 3
    ignore_index: int              # 255
    info_json: str                 # './datasets/meta/flame3_info.json'
    assume_train_ids: bool         # True，表示标签PNG直接存0/1/2/255

    # ---- 训练相关 ----
    cropsize: Tuple[int, int]      # (640, 480)
    splits: Tuple[str, ...]        # ('train', 'val', 'test')
    image_suffixes: Tuple[str, ...]
    synthesize_dummy_label: bool   # test阶段若无label则补255 mask

@dataclass
class MSRSConfig:
    dataset_name: str
    root: str
    method_name: str
    num_classes: int
    ignore_index: int
    cropsize: Tuple[int, int]


# def get_flame3_config(root: str = "./FLAME3") -> DatasetConfig:
#     return DatasetConfig(
#         dataset_name="FLAME3",
#         root=root,
#         visible_dir="Visible",
#         infrared_dir="Infrared",
#         label_dir="Label",
#         fusion_dir="Fusion",
#         method_name="Fusion",
#         num_classes=3,
#         ignore_index=255,
#         info_json="./datasets/meta/flame3_info.json",
#         assume_train_ids=True,
#         cropsize=(640, 512),
#         splits=("train", "val", "test"),
#         image_suffixes=(".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"),
#         synthesize_dummy_label=True,
#     )


# def get_msrs_config(root: str = "./MSRS") -> MSRSConfig:
#     return MSRSConfig(
#         dataset_name="MSRS",
#         root=root,
#         method_name="Fusion",
#         num_classes=9,
#         ignore_index=255,
#         cropsize=(640, 512),
#     )

# ==============================
# 兼容补丁说明（保留原代码）
# ==============================

def get_flame3_config(root: str = "./FLAME3") -> DatasetConfig:
    return DatasetConfig(
        dataset_name="flame3",
        root=root,
        visible_dir="Visible",
        infrared_dir="Infrared",
        label_dir="Label",
        fusion_dir="Fusion",
        method_name="Fusion",
        num_classes=3,
        ignore_index=255,
        info_json="./datasets/meta/flame3_info.json",
        assume_train_ids=True,
        cropsize=(640, 512),
        splits=("train", "val", "test"),
        image_suffixes=(".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"),
        synthesize_dummy_label=True,
    )


def get_msrs_config(root: str = "./MSRS") -> MSRSConfig:
    return MSRSConfig(
        dataset_name="msrs",
        root=root,
        method_name="Fusion",
        num_classes=9,
        ignore_index=255,
        cropsize=(640, 480),
    )