import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import transform as T


def list_supported_files(
    dir_path: str,
    suffixes: Tuple[str, ...] = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"),
) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    paths: List[str] = []
    for file_name in os.listdir(dir_path):
        fp = os.path.join(dir_path, file_name)
        if os.path.isfile(fp) and Path(file_name).suffix.lower() in suffixes:
            paths.append(os.path.abspath(fp))

    paths.sort()
    return paths


def build_stem_to_path_map(file_list: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for fp in file_list:
        stem = Path(fp).stem
        if stem in mapping:
            raise ValueError(f"Duplicate stem detected: {stem}")
        mapping[stem] = fp
    return mapping


def load_label_mapping(info_json_path: str) -> Dict[int, int]:
    with open(info_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(el["id"]): int(el["trainId"]) for el in data}


def convert_label_ids(label: np.ndarray, lb_map: Dict[int, int]) -> np.ndarray:
    out = label.copy()
    for old_id, train_id in lb_map.items():
        out[label == old_id] = train_id
    return out


def build_fusion_label_pairs(
    fusion_dir: str,
    label_dir: str,
    suffixes: Tuple[str, ...] = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"),
) -> List[Tuple[str, str, str]]:
    fusion_files = list_supported_files(fusion_dir, suffixes)
    label_files = list_supported_files(label_dir, suffixes)

    fusion_map = build_stem_to_path_map(fusion_files)
    label_map = build_stem_to_path_map(label_files)

    keys = sorted(list(set(fusion_map.keys()) & set(label_map.keys())))
    if len(keys) == 0:
        raise RuntimeError("No matched fusion-label pairs found.")

    pairs: List[Tuple[str, str, str]] = []
    for key in keys:
        pairs.append((fusion_map[key], label_map[key], os.path.basename(fusion_map[key])))
    return pairs


class Flame3SegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        method: str = "Fusion",
        cropsize: Tuple[int, int] = (640, 480),
        info_json_path: str = "./datasets/meta/flame3_info.json",
        assume_train_ids: bool = True,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.root = root
        self.split = split
        self.method = method
        self.cropsize = cropsize
        self.info_json_path = info_json_path
        self.assume_train_ids = assume_train_ids
        self.ignore_index = ignore_index
        self.to_tensor = transforms.ToTensor()
        self.lb_map = load_label_mapping(info_json_path)

        self.fusion_dir = os.path.join(root, "Fusion", split)
        self.label_dir = os.path.join(root, "Label", split)
        self.samples = build_fusion_label_pairs(self.fusion_dir, self.label_dir)
        self.trans_train = self._build_transforms()

    def _build_transforms(self):
        if self.split == "train":
            return T.Compose([
                T.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                ),
                T.HorizontalFlip(),
                T.RandomScale((0.75, 1.0, 1.25, 1.5)),
                T.RandomCrop(self.cropsize),
            ])
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fusion_path, label_path, fn = self.samples[idx]

        img = Image.open(fusion_path).convert("RGB")
        label = np.array(Image.open(label_path))

        if label.ndim == 3:
            label = label[..., 0]

        if not self.assume_train_ids:
            label = convert_label_ids(label, self.lb_map)

        if self.trans_train is not None:
            img, label = self.trans_train(img, label)

        img = self.to_tensor(img)
        label = label.astype(np.int64)
        label = np.expand_dims(label, axis=0)
        return img, label, fn
