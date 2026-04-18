import os
from types import SimpleNamespace
from typing import Any

from configs.flame3_config import get_flame3_config, get_msrs_config
from datasets.flame3_fusion_dataset import Flame3FusionDataset
from datasets.flame3_seg_dataset import Flame3SegDataset

from TaskFusion_dataset import Fusion_dataset as MSRSFusionDataset
from cityscapes import CityScapes as MSRSSegDataset


def get_dataset_config(dataset_name: str, dataset_root: str | None = None) -> Any:
    dataset_name = dataset_name.lower()
    if dataset_name == "flame3":
        return get_flame3_config(root=dataset_root or "./FLAME3")
    if dataset_name == "msrs":
        return get_msrs_config(root=dataset_root or "./MSRS")
    raise NotImplementedError(f"Unsupported dataset_name: {dataset_name}")


def build_fusion_dataset(cfg, split: str, require_label: bool = True):
    if cfg.dataset_name == "flame3":
        return Flame3FusionDataset(
            root=cfg.root,
            split=split,
            visible_dir=cfg.visible_dir,
            infrared_dir=cfg.infrared_dir,
            label_dir=cfg.label_dir,
            suffixes=cfg.image_suffixes,
            require_label=require_label,
            synthesize_dummy_label=cfg.synthesize_dummy_label,
            ignore_index=cfg.ignore_index,
        )
    if cfg.dataset_name == "msrs":
        return MSRSFusionDataset(split)
    raise NotImplementedError(f"Unsupported dataset_name: {cfg.dataset_name}")


def build_seg_dataset(cfg, split: str, method: str = "Fusion"):
    if cfg.dataset_name == "flame3":
        return Flame3SegDataset(
            root=cfg.root,
            split=split,
            method=method,
            cropsize=cfg.cropsize,
            info_json_path=cfg.info_json,
            assume_train_ids=cfg.assume_train_ids,
            ignore_index=cfg.ignore_index,
        )
    if cfg.dataset_name == "msrs":
        return MSRSSegDataset(
            cfg.root,
            cropsize=list(cfg.cropsize),
            mode=split,
            Method=method,
        )
    raise NotImplementedError(f"Unsupported dataset_name: {cfg.dataset_name}")


def get_fusion_output_dir(cfg, split: str, method: str = "Fusion") -> str:
    if cfg.dataset_name == "flame3":
        out_dir = os.path.join(cfg.root, cfg.fusion_dir, split)
    elif cfg.dataset_name == "msrs":
        out_dir = os.path.join(cfg.root, "Fusion", split, "MSRS")
    else:
        raise NotImplementedError(f"Unsupported dataset_name: {cfg.dataset_name}")

    os.makedirs(out_dir, exist_ok=True)
    return out_dir
