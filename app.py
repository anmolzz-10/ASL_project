# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# -------------------------------
# 1. Config
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "swin_temporal_best.pth"   # change if different

# Classes: 0-9 + A-Z
CLASS_NAMES = [str(i) for i in range(10)] + [chr(ord('A')+i) for i in range(26)]

# -------------------------------
# 2. Preprocessing
# -------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -------------------------------
# 3. Model Definition
# -------------------------------
class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load SwinV2-T backbone
        backbone = torch.hub.load("pytorch/vision", "swin_v2_t", weights=None)
        hidden_dim = backbone.head.in_features   # get features BEFORE removing head
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.hidden_dim = hidden_dim   # store it for later

    def forward(self, x):
        return self.backbone(x)

class SwinTemporal(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        hidden_dim = self.frame_encoder.hidden_dim  # now works ✅
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=3072,   # 🔥 match checkpoint
            batch_first=True
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)             # merge batch + time
        feats = self.frame_encoder(x)        # (B*T, hidden_dim)
        feats = feats.view(B, T, -1)         # (B, T, hidden_dim)
        feats = self.temporal(feats)         # (B, T, hidden_dim)
        pooled = feats.mean(1)               # temporal avg
        return self.cls(pooled)


# -------------------------------
# 4. Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = SwinTemporal(num_classes=len(CLASS_NAMES))
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state["model"])   # checkpoint contains "model" key
    model.to(DEVICE).eval()
    return model

model = load_model()

# -------------------------------
# 5. Prediction
# -------------------------------
def predict_clip(images):
    # images: list of PIL Images
    tensors = [transform(img) for img in images]   # [(C,H,W), ...]
    clip_tensor = torch.stack(tensors)             # (T,C,H,W)
    clip_tensor = clip_tensor.unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)

    with torch.no_grad():
        logits = model(clip_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top3_idx = probs.argsort()[-3:][::-1]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]

# -------------------------------
# 6. Streamlit UI
# -------------------------------
st.title("🤟 ASL Prototype Classifier (0-9 & A-Z)")

st.markdown("Upload **5 images** (frames) to form a single clip.")

uploaded_files = st.file_uploader(
    "Upload 5 images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) != 5:
        st.error("Please upload exactly **5 images** for one clip.")
    else:
        images = [Image.open(f).convert("RGB") for f in uploaded_files]
        
        st.subheader("Uploaded Frames")
        st.image(images, width=100)

        # Run prediction
        preds = predict_clip(images)

        st.subheader("Predictions")
        for cls, prob in preds:
            st.write(f"**{cls}** : {prob:.4f}")
