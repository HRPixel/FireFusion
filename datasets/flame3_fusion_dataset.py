import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


def intersect_sample_keys(
    vis_map: Dict[str, str],
    ir_map: Dict[str, str],
    label_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    keys = set(vis_map.keys()) & set(ir_map.keys())
    if label_map is not None:
        keys = keys & set(label_map.keys())
    keys = sorted(list(keys))
    if len(keys) == 0:
        raise RuntimeError("No matched samples found across required modalities.")
    return keys


def make_dummy_label(height: int, width: int, fill_value: int = 255) -> np.ndarray:
    return np.full((height, width), fill_value, dtype=np.uint8)


def read_visible_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)


def read_infrared_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    return transforms.ToTensor()(img)


def read_label_image(path: str) -> np.ndarray:
    label = np.array(Image.open(path))
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.int64)


class Flame3FusionDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        visible_dir: str = "Visible",
        infrared_dir: str = "Infrared",
        label_dir: str = "Label",
        suffixes: Tuple[str, ...] = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"),
        require_label: bool = True,
        synthesize_dummy_label: bool = False,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.root = root
        self.split = split
        self.require_label = require_label
        self.synthesize_dummy_label = synthesize_dummy_label
        self.ignore_index = ignore_index
        self.suffixes = suffixes

        self.visible_path = os.path.join(root, visible_dir, split)
        self.infrared_path = os.path.join(root, infrared_dir, split)
        self.label_path = os.path.join(root, label_dir, split)

        self.samples = self._build_samples()

    def _build_samples(self):
        vis_files = list_supported_files(self.visible_path, self.suffixes)
        ir_files = list_supported_files(self.infrared_path, self.suffixes)

        vis_map = build_stem_to_path_map(vis_files)
        ir_map = build_stem_to_path_map(ir_files)

        label_map = None
        if os.path.isdir(self.label_path):
            label_files = list_supported_files(self.label_path, self.suffixes)
            if len(label_files) > 0:
                label_map = build_stem_to_path_map(label_files)

        if self.require_label:
            if label_map is None:
                raise FileNotFoundError(f"Label directory is required but missing or empty: {self.label_path}")
            keys = intersect_sample_keys(vis_map, ir_map, label_map)
        else:
            keys = intersect_sample_keys(vis_map, ir_map, label_map if label_map is not None else None)
            if label_map is None:
                keys = intersect_sample_keys(vis_map, ir_map, None)

        samples = []
        for key in keys:
            samples.append(
                {
                    "key": key,
                    "vis_path": vis_map[key],
                    "ir_path": ir_map[key],
                    "label_path": None if label_map is None or key not in label_map else label_map[key],
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image_vis = read_visible_image(sample["vis_path"])
        image_ir = read_infrared_image(sample["ir_path"])

        if image_vis.shape[1:] != image_ir.shape[1:]:
            raise ValueError(
                f"Shape mismatch between visible and infrared: "
                f"{image_vis.shape} vs {image_ir.shape} for key={sample['key']}"
            )

        h, w = image_vis.shape[1], image_vis.shape[2]

        if sample["label_path"] is not None:
            label = read_label_image(sample["label_path"])
        else:
            if not self.synthesize_dummy_label:
                raise FileNotFoundError(
                    f"Label missing for sample={sample['key']} and synthesize_dummy_label=False"
                )
            label = make_dummy_label(h, w, self.ignore_index)

        if label.shape != (h, w):
            raise ValueError(
                f"Label size mismatch: expected {(h, w)}, got {label.shape} for key={sample['key']}"
            )

        label = torch.from_numpy(label).long()
        name = os.path.basename(sample["vis_path"])
        return image_vis, image_ir, label, name
