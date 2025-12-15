#!/usr/bin/env python3
# extract_semantic_dinov3.py
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
import sys
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Local path to the DINOv3 code repository.
# Please ensure this path points to the folder cloned from `git clone https://github.com/facebookresearch/dinov3.git`
DINOV3_REPO_PATH = '/data1/zhanghongji/datasets/github/dinov3'

# Local path to the DINOv3 ViT-L/16 weights file
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
    print("DINOv3 ViT-L/16 model loaded successfully.")
except Exception as e:
    print(f"Failed to load model. Please check the DINOV3 repository path and weights path. Error: {e}")
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
    
    print(f"Found {len(img_paths)} images in {root_dir}...")
    
    for img_path in tqdm(img_paths, ncols=80):
        key = str(img_path.relative_to(root_dir)).replace("\\", "/") # Normalize path separators
        try:
            feat = extract_semantic_dinov3(img_path)
            feats[key] = feat
        except Exception as e:
            print(f"Skipping corrupted or unrecognizable image: {img_path}, reason: {e}")
            continue
            
    torch.save(feats, save_path)
    print(f"[DINOv3] Recursive extraction completed, total {len(feats)} images, saved to {save_path}")
    
if __name__ == "__main__":
    os.makedirs("/data1/zhanghongji/datasets", exist_ok=True)

    save_path_train = "/data1/zhanghongji/datasets/semantic_dinov3_AG_train.pt"
    save_path_val = "/data1/zhanghongji/datasets/semantic_dinov3_AG_val.pt"
    save_path_chameleon = "/data1/zhanghongji/datasets/semantic_dinov3_Gen.pt"

    batch_extract_recursive("/data1/zhanghongji/datasets/AIDE/train/GenImage", save_path_train)
    batch_extract_recursive("/data1/zhanghongji/datasets/AIDE/val/GenImage",   save_path_val)
    batch_extract_recursive("/data1/zhanghongji/datasets/AIDE/val/GenImage",  save_path_chameleon)