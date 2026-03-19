import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from streamlit_image_comparison import image_comparison

from colorizers import eccv16, siggraph17
from colorizers import util

st.markdown("""
<style>
/* Make buttons nice */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
}

/* Add space */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# PAGE CONFIG
# =========================
st.title(" AI Image Colorizer")
st.write("Turn black & white images into color using AI")
st.divider()

# =========================
# DARK UI
# =========================


# =========================
# FIX IMAGE FUNCTION
# =========================
def fix_image(img):
    # Tensor → numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.squeeze(img)

    # If grayscale → convert to 3 channel
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    # Normalize float → uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    return img

# Convert to downloadable bytes
def to_bytes(img):
    img = Image.fromarray(fix_image(img))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model1 = eccv16(pretrained=True).eval()
    model2 = siggraph17(pretrained=True).eval()
    return model1, model2

model_eccv16, model_siggraph17 = load_models()

# =========================
# TITLE
# =========================
st.title("AI Image Colorization Studio")
st.write("Upload images and watch AI bring them to life")

use_gpu = st.checkbox(" Use GPU")
device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

model_eccv16.to(device)
model_siggraph17.to(device)

# =========================
# FILE UPLOAD
# =========================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    uploaded_files = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
# =========================
# MAIN PROCESS
# =========================
if uploaded_files:

    for uploaded_file in uploaded_files:

        st.divider()

        # ---- Load image safely ----
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        st.subheader(f"📷 {uploaded_file.name}")
        st.image(img, width=300)

        # Resize large images
        if img_np.shape[0] > 512:
            img = img.resize((512, 512))
            img_np = np.array(img)

        # ---- Progress ----
        progress = st.progress(0)

        # ---- Preprocess ----
        progress.progress(20)

        img_l = util.preprocess_img(img_np, HW=(256, 256))[0]

        if isinstance(img_l, np.ndarray):
            img_l = torch.from_numpy(img_l)

        # Ensure correct shape [1,1,H,W]
        if img_l.dim() == 2:
            img_l = img_l.unsqueeze(0).unsqueeze(0)
        elif img_l.dim() == 3:
            img_l = img_l.unsqueeze(0)

        img_l = img_l.to(device)

        # ---- Model ----
        progress.progress(60)

        with torch.no_grad():
            out_eccv16 = model_eccv16(img_l)
            out_eccv16 = util.postprocess_tens(img_l.cpu(), out_eccv16.cpu())

            out_siggraph17 = model_siggraph17(img_l)
            out_siggraph17 = util.postprocess_tens(img_l.cpu(), out_siggraph17.cpu())

        progress.progress(100)
        st.success(" Colorization complete!")

        # ---- Fix images for UI ----
        fixed_original = fix_image(img_np)
        fixed_eccv16 = fix_image(out_eccv16)
        fixed_siggraph17 = fix_image(out_siggraph17)

        # =========================
        # SLIDER
        # =========================
        col1, col2, col3 = st.columns([1,4,1])

    with col2:
        image_comparison(
        img1=fixed_original,
        img2=fixed_siggraph17,
        label1="Original",
        label2="Colorized",
        width=900
    )
        

        # =========================
        # SIDE BY SIDE
        # =========================
        st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(fixed_eccv16, caption="ECCV16")

    with col2:
        st.image(fixed_siggraph17, caption="SIGGRAPH17")
        # =========================
        # MODEL INSIGHTS
        # =========================
        st.write("###  Model Comparison")
        st.write("""
        - **ECCV16** → smoother, stable colors  
        - **SIGGRAPH17** → sharper, vibrant colors  
        """)

        # =========================
        # DOWNLOAD
        # =========================
        st.download_button(
            "Download ECCV16",
            data=to_bytes(out_eccv16),
            file_name="eccv16.png"
        )

        st.download_button(
            " Download SIGGRAPH17",
            data=to_bytes(out_siggraph17),
            file_name="siggraph17.png"
        )