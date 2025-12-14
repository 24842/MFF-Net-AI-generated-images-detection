#!/usr/bin/env python3
# extract_semantic_dinov3.py
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DINOv3 代码仓库的本地路径
# 请确保这个路径是 `git clone https://github.com/facebookresearch/dinov3.git` 下载的文件夹
DINOV3_REPO_PATH = '/data1/zhanghongji/datasets/github/dinov3'

# DINOv3 ViT-L/16 权重文件的本地路径
DINOV3_WEIGHTS_PATH = '/data1/zhanghongji/datasets/github/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth' 

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

try:
    model = torch.hub.load(DINOV3_REPO_PATH, 'dinov3_vitl16', source='local', weights=DINOV3_WEIGHTS_PATH)
    model.eval().to(device)
    print("DINOv3 ViT-L/16 模型加载成功。")
except Exception as e:
    print(f"模型加载失败，请检查 DINOV3 仓库路径和权重路径是否正确。错误: {e}")
    exit()

original_sdpa = F.scaled_dot_product_attention

def extract_semantic_dinov3(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)  # [1, 1024]
    return feat.cpu().squeeze(0)  # [1024]

def batch_extract_recursive(root_dir, save_path):
    root_dir = Path(root_dir)
    feats = {}
    all_imgs = list(root_dir.rglob("*"))
    img_paths = [p for p in all_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    
    print(f"在 {root_dir} 中找到 {len(img_paths)} 张图片...")
    
    for img_path in tqdm(img_paths, ncols=80):
        key = str(img_path.relative_to(root_dir)).replace("\\", "/") # 统一路径分隔符
        try:
            feat = extract_semantic_dinov3(img_path)
            feats[key] = feat
        except Exception as e:
            print(f"跳过损坏或无法识别的图片: {img_path}，原因: {e}")
            continue
            
    torch.save(feats, save_path)
    print(f"[DINOv3] 递归提取完成，共 {len(feats)} 张，已保存到 {save_path}")

if __name__ == "__main__":
    os.makedirs("/data1/zhanghongji/datasets", exist_ok=True)

    save_path_train = "/data1/zhanghongji/datasets/semantic_dinov3_AG_train.pt"
    save_path_val = "/data1/zhanghongji/datasets/semantic_dinov3_AG_val.pt"
    save_path_chameleon = "/data1/zhanghongji/datasets/semantic_dinov3_Gen.pt"

    batch_extract_recursive("/data1/zhanghongji/datasets/AIDE/train/GenImage", save_path_train)
    batch_extract_recursive("/data1/zhanghongji/datasets/AIDE/val/GenImage",   save_path_val)
    batch_extract_recursive("/data1/zhanghongji/datasets/AIDE/val/GenImage",  save_path_chameleon)