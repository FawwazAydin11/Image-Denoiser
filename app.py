import streamlit as st

st.set_page_config(
    page_title="RIDNet Image Denoising",
    layout="centered",
    page_icon="🧠"
)

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io


# ==========================
class CALayer(nn.Module):
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
    def __init__(self, channel):
        super(ResidualBlock_CA, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.ca = CALayer(channel, reduction=16)

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        return out + res


class RIDNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_blocks=8):
        super(RIDNet, self).__init__()
        self.fea_conv = nn.Conv2d(num_channels, num_features, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock_CA(num_features) for _ in range(num_blocks)])
        self.recon_conv = nn.Conv2d(num_features, num_channels, 3, 1, 1)

    def forward(self, x):
        out = self.fea_conv(x)
        out = self.res_blocks(out)
        out = self.recon_conv(out)
        return x + out


# ==========================
@st.cache_resource
def load_model():
    model = RIDNet()
    state_dict = torch.load("best_ridnet_epoch30.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()


# ==========================
def denoise_image(model, image_pil):
    image = np.array(image_pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    output = output.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)

# ==========================
st.set_page_config(page_title="RIDNet Image Denoising", layout="centered")

st.markdown("""
    <style>
    /* Background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e1f4b 0%, #3c1d63 50%, #2e8bc0 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Animasi fade in */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    h1, h2, h3 {
        color: #ffffff;
        text-align: center;
        animation: fadeIn 1.2s ease-in-out;
        font-weight: 600;
    }

    /* Card animasi */
    .image-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(46, 139, 192, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .image-card:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px rgba(173, 216, 230, 0.9);
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(173, 216, 230, 0.3);
    }
    [data-testid="stFileUploader"]:hover {
        box-shadow: 0 0 25px rgba(0, 191, 255, 0.6);
        transform: scale(1.02);
    }

    /* Tombol download */
    div.stDownloadButton > button {
        background: linear-gradient(90deg, #00b4db, #0083b0);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 15px rgba(0, 191, 255, 0.5);
    }

    div.stDownloadButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0083b0, #00b4db);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.9);
    }

    /* Spinner kustom */
    .stSpinner > div {
        border-top-color: #00b4db !important;
        border-right-color: #00b4db !important;
    }

    /* Footer kecil */
    .footer {
        text-align:center;
        font-size:13px;
        color:#d0e6f7;
        margin-top:30px;
        animation: fadeIn 2s ease;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Image Denoising</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>Hilangkan noise dan tampilkan gambar jernih dengan AI-powered denoising 🌌</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Unggah gambar (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<h3>✨ Before vs After Denoise</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.image(image, caption="Before", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("⚙️ Sedang memproses gambar dengan AI..."):
        denoised_image = denoise_image(model, image)

    with col2:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.image(denoised_image, caption="After", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    buf = io.BytesIO()
    denoised_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Download Hasil Denoise",
        data=byte_im,
        file_name="denoised_result.png",
        mime="image/png",
    )


