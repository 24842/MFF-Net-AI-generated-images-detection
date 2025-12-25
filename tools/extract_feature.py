#!/usr/bin/env python3
# extract_lh_vgg_enhanced.py
import sys
import os
# Ensure project root is on sys.path so local package imports like `models` work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, vgg16
import tqdm
import numpy as np

from models.vqvae import VQVAE
from models.srm import EnhancedSRMWithWavelet
from freq_dec import freq_decompose

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

vqvae = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32,
              embed_dim=64, n_embed=512, decay=0.99).to(device)
ckpt = torch.load('/data1/zhanghongji/datasets/github/vqvae_560.pt', weights_only=False, map_location=device)
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
vqvae.load_state_dict(ckpt, strict=True)
vqvae.eval()

class VGGPerceptualExtractor(nn.Module):
   
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(4): 
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):  
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):  
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):     
            self.slice4.add_module(str(x), vgg[x])
        for x in range(23, 30): 
            self.slice5.add_module(str(x), vgg[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            
    def forward(self, x):
        x = self.normalize(x)
        
        h1 = self.slice1(x)   # [B, 64, H, W]
        h2 = self.slice2(h1)  # [B, 128, H/2, W/2]
        h3 = self.slice3(h2)  # [B, 256, H/4, W/4]
        h4 = self.slice4(h3)  # [B, 512, H/8, W/8]
        h5 = self.slice5(h4)  # [B, 512, H/16, W/16]
        
        return [h1, h2, h3, h4, h5]

preprocess = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])

class EnhancedChannelAdapter(nn.Module):
    def __init__(self, max_channels=512, out_channels=3):
        super().__init__()
        self.adapters = nn.ModuleDict({
            '32': nn.Conv2d(32, out_channels, kernel_size=1),
            '64': nn.Conv2d(64, out_channels, kernel_size=1),
            '96': nn.Conv2d(96, out_channels, kernel_size=1),
            '128': nn.Conv2d(128, out_channels, kernel_size=1),
            '256': nn.Conv2d(256, out_channels, kernel_size=1),
            '512': nn.Conv2d(512, out_channels, kernel_size=1)
        })

    def forward(self, x):
        channels = x.shape[1]
        if channels <= 32:
            key = '32'
        elif channels <= 64:
            key = '64'
        elif channels <= 96:
            key = '96'
        elif channels <= 128:
            key = '128'
        elif channels <= 256:
            key = '256'
        else:
            key = '512'

        if channels != int(key):
            if channels > int(key):
                x = x[:, :int(key), :, :]  
            else:
                pad_channels = int(key) - channels
                padding = torch.zeros(x.shape[0], pad_channels, x.shape[2], x.shape[3]).to(x.device)
                x = torch.cat([x, padding], dim=1)

        return self.adapters[key](x)

resnet_lr = resnet50(pretrained=True)
resnet_lr.fc = nn.Identity()
resnet_lr.eval().to(device)

resnet_hr = resnet50(pretrained=True)
resnet_hr.fc = nn.Identity()
resnet_hr.eval().to(device)

enhanced_srm = EnhancedSRMWithWavelet(out_channels=32).eval().to(device)
vgg_extractor = VGGPerceptualExtractor().eval().to(device)
channel_adapter = EnhancedChannelAdapter(max_channels=512, out_channels=3).eval().to(device)

class SimpleDCTScoring:
    def __init__(self, patch_size=32, num_patches=2):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def extract_patches(self, image):
        batch_size, channels, height, width = image.shape
        patches = []

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size

        for i in range(h_patches):
            for j in range(w_patches):
                start_i = i * self.patch_size
                start_j = j * self.patch_size
                patch = image[:, :, start_i:start_i + self.patch_size, start_j:start_j + self.patch_size]
                patches.append(patch)

        return patches

    def compute_dct_scores(self, patches):
        scores = []

        for patch in patches:
            patch_np = patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
            total_energy = 0
            for c in range(patch_np.shape[2]):
                channel_dct = np.fft.fft2(patch_np[:, :, c])
                energy = np.sum(np.abs(channel_dct[1:, 1:]) ** 2)
                total_energy += energy

            scores.append(total_energy)

        return scores

    def select_extreme_patches(self, image):
        patches = self.extract_patches(image)

        if not patches:
            return [], []

        scores = self.compute_dct_scores(patches)
        sorted_indices = np.argsort(scores)
        lowest_freq_indices = sorted_indices[:self.num_patches]
        highest_freq_indices = sorted_indices[-self.num_patches:]

        lowest_patches = [patches[i] for i in lowest_freq_indices]
        highest_patches = [patches[i] for i in highest_freq_indices]

        return highest_patches, lowest_patches

def extract_lpips_frequency_features(original_img, recon_img, vgg_extractor, enhanced_srm):
    vgg_high_features = []
    vgg_low_features = []
    vgg_features_original = vgg_extractor(original_img)
    vgg_features_recon = vgg_extractor(recon_img)
    
    for i, (feat_orig, feat_rec) in enumerate(zip(vgg_features_original, vgg_features_recon)):
        perceptual_diff = torch.abs(feat_orig - feat_rec)
        diff_gray = perceptual_diff.mean(dim=1, keepdim=True)
        diff_resized = F.interpolate(diff_gray, size=(256, 256), 
                                   mode='bilinear', align_corners=False)
        diff_low, diff_high = freq_decompose(diff_resized)
        low_noise = enhanced_srm(diff_low)
        high_noise = enhanced_srm(diff_high)
        vgg_low_features.append(low_noise)
        vgg_high_features.append(high_noise)
    
    return vgg_high_features, vgg_low_features

@torch.no_grad()
def extract_lr_hr_vqvae_with_vgg_lpips(img_path):
    image = Image.open(img_path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)  # [1,3,H,W]
    gray = x.mean(dim=1, keepdim=True)  # [1,1,H,W]
    dct_scorer = SimpleDCTScoring(patch_size=32, num_patches=2)
    high_freq_patches, low_freq_patches = dct_scorer.select_extreme_patches(x)
    quant_t, quant_b, *_ = vqvae.encode(x)
    rec = vqvae.decode(quant_t, quant_b)
    residual = gray - rec.mean(dim=1, keepdim=True)
    residual_3ch = residual.repeat(1, 3, 1, 1)
    res_high_freq_patches, res_low_freq_patches = dct_scorer.select_extreme_patches(residual_3ch)
    vgg_high_features, vgg_low_features = extract_lpips_frequency_features(x, rec, vgg_extractor, enhanced_srm)
    lr_features = []

    for patch in low_freq_patches:
        patch_resized = F.interpolate(patch, size=(256, 256), mode='bilinear', align_corners=False)
        patch_gray = patch_resized.mean(dim=1, keepdim=True)
        patch_low, _ = freq_decompose(patch_gray)
        patch_noise = enhanced_srm(patch_low)
        lr_features.append(patch_noise)

    for patch in res_low_freq_patches:
        patch_resized = F.interpolate(patch, size=(256, 256), mode='bilinear', align_corners=False)
        patch_gray = patch_resized.mean(dim=1, keepdim=True)
        patch_low, _ = freq_decompose(patch_gray)
        patch_noise = enhanced_srm(patch_low)
        lr_features.append(patch_noise)
        
    lr_features.extend(vgg_low_features)

    hr_features = []
    for patch in high_freq_patches:
        patch_resized = F.interpolate(patch, size=(256, 256), mode='bilinear', align_corners=False)
        patch_gray = patch_resized.mean(dim=1, keepdim=True)
        _, patch_high = freq_decompose(patch_gray)
        patch_noise = enhanced_srm(patch_high)
        hr_features.append(patch_noise)
    for patch in res_high_freq_patches:
        patch_resized = F.interpolate(patch, size=(256, 256), mode='bilinear', align_corners=False)
        patch_gray = patch_resized.mean(dim=1, keepdim=True)
        _, patch_high = freq_decompose(patch_gray)
        patch_noise = enhanced_srm(patch_high)
        hr_features.append(patch_noise)
    hr_features.extend(vgg_high_features)

    def merge_features(feature_list, target_channels=512):
        if not feature_list:
            return torch.zeros(1, target_channels, 256, 256).to(device)

        merged = torch.cat(feature_list, dim=1)

        current_channels = merged.shape[1]
        if current_channels < target_channels:
            padding = torch.zeros(1, target_channels - current_channels, 256, 256).to(device)
            merged = torch.cat([merged, padding], dim=1)
        elif current_channels > target_channels:
            merged = merged[:, :target_channels, :, :]

        return merged

    total_channels = (2 + 2 + 5) * 32  
    lr_combined = merge_features(lr_features, target_channels=total_channels)
    hr_combined = merge_features(hr_features, target_channels=total_channels)

    lr_3ch = channel_adapter(lr_combined)  # [1,3,H,W]
    hr_3ch = channel_adapter(hr_combined)  # [1,3,H,W]

    lr_feat = resnet_lr(lr_3ch).squeeze(0)  # [2048]
    hr_feat = resnet_hr(hr_3ch).squeeze(0)  # [2048]

    return lr_feat, hr_feat

@torch.no_grad()
def extract_lr_hr_vqvae(img_path):
    return extract_lr_hr_vqvae_with_vgg_lpips(img_path)

def batch_extract_target_dirs(root_dir, save_path):
    root_dir = Path(root_dir)
    feats = {}
    all_imgs = list(root_dir.rglob("*"))
    img_paths = [p for p in all_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for img_path in tqdm.tqdm(img_paths, ncols=80):
        key = str(img_path.relative_to(root_dir))
        try:
            lr, hr = extract_lr_hr_vqvae(img_path)
            feats[key] = {"lr": lr.cpu(), "hr": hr.cpu()}
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    torch.save(feats, save_path)
    print(f"[VQ-VAE-LR/HR with VGG16] Extraction completed, total {len(feats)} images, saved to {save_path}")
if __name__ == "__main__":
    os.makedirs("/data1/zhanghongji/datasets", exist_ok=True)
    batch_extract_target_dirs("/data1/zhanghongji/datasets/AIDE/train/progan",
                              "/data1/zhanghongji/datasets/lrhr_vgg_progan_train.pt")
    batch_extract_target_dirs("/data1/zhanghongji/datasets/AIDE/val/progan",
                              "/data1/zhanghongji/datasets/lrhr_vgg_progan_val.pt")
    batch_extract_target_dirs("/data1/zhanghongji/datasets/AIDE/test_blur/test_blur_3.0",
                              "/data1/zhanghongji/datasets/lrhr_vgg_blur3.pt")