import torch
import torch.nn as nn

# ======================================
# Versi lengkap RIDNet (dengan CA Layer)
# ======================================

class CALayer(nn.Module):
    """Channel Attention Layer"""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ResidualBlock_CA(nn.Module):
    """Residual Block + Channel Attention"""
    def __init__(self, channel):
        super(ResidualBlock_CA, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.ca = CALayer(channel, reduction=16)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        return out + residual


class RIDNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_blocks=8):
        super(RIDNet, self).__init__()
        self.fea_conv = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock_CA(num_features) for _ in range(num_blocks)])
        self.recon_conv = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.fea_conv(x)
        out = self.res_blocks(out)
        out = self.recon_conv(out)
        return x + out  # residual connection


# ======================================
# Test Load Model
# ======================================

model_path = "best_ridnet_epoch30.pth"

try:
    model = RIDNet()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("✅ Model berhasil di-load! RIDNet cocok dengan file .pth kamu 🎉")
except Exception as e:
    print("❌ Masih gagal load model:")
    print(str(e))
