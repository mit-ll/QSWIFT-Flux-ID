import torch.nn as nn
import torch
import torch.nn.functional as F

# Defines the arcitecture of the CNN. Needed to rebuild the model for use after training as well

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kern: int, stride: int, pad: int, dil: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,  c_out, kernel_size=kern, stride=stride, padding=pad, dilation=dil),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kern, stride=stride, padding=pad, dilation=dil),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class HeatmapUNet(nn.Module):
    """
    Input:  [B,1,H,W]
    Output: [B,1,H,W] logits
    """
    def __init__(self, base: int, kern: int, stride: int, pad: int, dil: int):
        super().__init__()
        self.enc1 = ConvBlock(1, base, kern=kern, stride=stride, pad=pad, dil=dil)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2, kern=kern, stride=stride, pad=pad, dil=dil)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 2, base * 4, kern=kern, stride=stride, pad=pad, dil=dil)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, kern=kern, stride=stride, pad=pad, dil=dil)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, kern=kern, stride=stride, pad=pad, dil=dil)

        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                # [B, base,   H,   W]
        e2 = self.enc2(self.pool1(e1))   # [B, 2base, H/2, W/2]
        m  = self.mid(self.pool2(e2))    # [B, 4base, H/4, W/4]

        # Up + size-match to skip connection (fixes odd H/W mismatches)
        d2 = self.up2(m)  # nominally [B, 2base, ~H/2, ~W/2]
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)  # nominally [B, base, ~H, ~W]
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)

        # Optional: ensure exact match to input spatial size (useful if stride/pad/dil ever change size)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out  # logits