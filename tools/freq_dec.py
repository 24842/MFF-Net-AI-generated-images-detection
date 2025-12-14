import torch
import torch.fft as fft
import torch.nn.functional as F

def freq_decompose(img, low_freq_ratio=0.3):
    # img: [B, C, H, W]
    B, C, H, W = img.shape

    freq = fft.fftn(img, dim=(-2, -1))
    freq_shift = fft.fftshift(freq, dim=(-2, -1))


    mask = torch.zeros_like(freq_shift)
    center_h, center_w = H // 2, W // 2
    radius = int(min(H, W) * low_freq_ratio / 2)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    mask[:, :, (y - center_h) ** 2 + (x - center_w) ** 2 <= radius ** 2] = 1

    low_freq = freq_shift * mask
    high_freq = freq_shift * (1 - mask)

    low_img = torch.real(fft.ifftn(fft.ifftshift(low_freq, dim=(-2, -1)), dim=(-2, -1)))
    high_img = torch.real(fft.ifftn(fft.ifftshift(high_freq, dim=(-2, -1)), dim=(-2, -1)))
    return low_img, high_img