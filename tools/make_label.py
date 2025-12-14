#!/usr/bin/env python3
# make_label_dict_total_with_check.py
import torch
from pathlib import Path
from tqdm import tqdm
import os

def make_label_dict_total_with_check(root_dir, save_path):
    root_dir = Path(root_dir)
    target_dirs = list(root_dir.rglob("0_real")) + list(root_dir.rglob("1_fake"))

    label_dict = {}
    from collections import defaultdict
    stats = defaultdict(lambda: {"real": 0, "fake": 0, "total": 0})

    for td in tqdm(target_dirs, desc="Scan dirs"):
        sub = td.parts[root_dir.parts.index(root_dir.name) + 1]  
        for img in td.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = str(img.relative_to(root_dir))  
                label = 0 if "0_real" in img.parts else 1
                label_dict[key] = label
                if "0_real" in img.parts:
                    stats[sub]["real"] += 1
                else:
                    stats[sub]["fake"] += 1
                stats[sub]["total"] += 1

    for sub, cnt in stats.items():
        print(f"[Check] {sub} | real={cnt['real']} | fake={cnt['fake']} | total={cnt['total']}")

    torch.save(label_dict, save_path)
    print(f"[Done] 标签字典完成，共 {len(label_dict)} 张 -> {save_path}")

if __name__ == "__main__":
    os.makedirs("/data1/zhanghongji/datasets", exist_ok=True)
    make_label_dict_total_with_check(
        "/data1/zhanghongji/datasets/AIDE/train/progan",
        "/data1/zhanghongji/datasets/label_progan_tr.pt"
    )
    make_label_dict_total_with_check(
        "/data1/zhanghongji/datasets/AIDE/val/progan",
        "/data1/zhanghongji/datasets/label_progan_v.pt"
    )
    make_label_dict_total_with_check(
        "/data1/zhanghongji/datasets/AIDE/val/GenImage",
        "/data1/zhanghongji/datasets/label_GenImage_te.pt"
    )