import torch.nn as nn
import torch
import numpy as np
import pywt
import torch.nn.functional as F


class EnhancedSRMWithWavelet(nn.Module):

    def __init__(self, out_channels=32, wavelet_type='haar'):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.srm_fixed = self._create_srm_filters()  

        self.learnable_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.wavelet_conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.multiscale = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.fusion = nn.Conv2d(3 + 16 + 8 + 4, out_channels, kernel_size=1)

    def _create_srm_filters(self):
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filters = torch.tensor([filter1, filter2, filter3], dtype=torch.float32).unsqueeze(1)
        conv = nn.Conv2d(1, 3, kernel_size=5, padding=2, bias=False)
        conv.weight = nn.Parameter(filters, requires_grad=False)
        return conv

    def wavelet_transform_fixed_size(self, x):

        batch_size, _, h, w = x.shape
        
        coeffs_list = []
        
        for i in range(batch_size):
            img_tensor = x[i, 0]  # [H, W]
            
            img_np = img_tensor.detach().cpu().numpy()
            
            try:
                coeffs2 = pywt.dwt2(img_np, self.wavelet_type)
                cA, (cH, cV, cD) = coeffs2
                cA = cA.astype(np.float32)
                cH = cH.astype(np.float32)
                cV = cV.astype(np.float32)
                cD = cD.astype(np.float32)
                
                cA_tensor = torch.tensor(cA, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
                cH_tensor = torch.tensor(cH, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
                cV_tensor = torch.tensor(cV, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
                cD_tensor = torch.tensor(cD, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
                
                cA_upsampled = F.interpolate(cA_tensor, size=(h, w), mode='bilinear', align_corners=False)
                cH_upsampled = F.interpolate(cH_tensor, size=(h, w), mode='bilinear', align_corners=False)
                cV_upsampled = F.interpolate(cV_tensor, size=(h, w), mode='bilinear', align_corners=False)
                cD_upsampled = F.interpolate(cD_tensor, size=(h, w), mode='bilinear', align_corners=False)
                
                wavelet_feat = torch.cat([cA_upsampled, cH_upsampled, cV_upsampled, cD_upsampled], dim=1)
                coeffs_list.append(wavelet_feat)
                
            except Exception as e:
                print(f"Wavelet transform failed: {e}. Using zero tensor as fallback.")
                zero_feat = torch.zeros(1, 4, h, w, dtype=torch.float32, device=x.device)
                coeffs_list.append(zero_feat)
        
        return torch.cat(coeffs_list, dim=0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        srm_feat = self.srm_fixed(x)  # [B, 3, H, W]

        learnable_feat = self.learnable_conv(x)  # [B, 16, H, W]

        wavelet_feat = self.wavelet_transform_fixed_size(x)  # [B, 4, H, W]
        wavelet_feat = self.wavelet_conv(wavelet_feat)  # [B, 8, H, W]

        multiscale_feat = self.multiscale(x)  # [B, 4, H, W]

        features = [srm_feat, learnable_feat, wavelet_feat, multiscale_feat]
        for i, feat in enumerate(features):
            if feat.shape[2:] != (H, W):
                print(f"Warning: Feature {i} size mismatch {feat.shape[2:]} vs {(H, W)}")
                features[i] = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        
        combined = torch.cat(features, dim=1)
        output = self.fusion(combined)  # [B, out_channels, H, W]

        return output