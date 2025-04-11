import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
import requests

st.title("Stroke Patient Pain Intensity Detector")
st.markdown("Upload a full-face image of a stroke patient. The app will detect the affected side and predict pain intensity using the unaffected side.")
st.write("🔧 App started. Preparing to download models...")

@st.cache_resource
def download_models():
    model_urls = {
        "cnn_stroke_model.keras": "https://huggingface.co/AdhamQQ/cnn_stroke_model/resolve/main/cnn_stroke_model.keras",
        "right_side_pain_model.pth": "https://huggingface.co/AdhamQQ/cnn_stroke_model/resolve/main/right_side_pain_model.pth"
    }

    for filename, url in model_urls.items():
        if not os.path.exists(filename):
            st.write(f"📥 Downloading {filename}...")
            r = requests.get(url)
            with open(filename, "wb") as f:
                f.write(r.content)
            st.write(f"✅ {filename} downloaded.")
        else:
            st.write(f"✔️ {filename} already exists.")

    st.write("📦 Loading models...")
    stroke_model = load_model("cnn_stroke_model.keras")

    class PainRegressor(nn.Module):
        def __init__(self):
            super(PainRegressor, self).__init__()
            from torchvision.models import resnet18, ResNet18_Weights
            self.base = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = self.base.fc.in_features
            self.base.fc = nn.Linear(num_features, 1)
        def forward(self, x):
            return self.base(x)

    pain_model = PainRegressor()
    pain_model.load_state_dict(torch.load("right_side_pain_model.pth", map_location=torch.device('cpu')))
    pain_model.eval()
    st.write("✅ Models loaded.")

    return stroke_model, pain_model

stroke_model, pain_model = download_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("📷 Image uploaded. Processing...")
    full_image = Image.open(uploaded_file).convert("RGB")
    st.image(full_image, caption="Uploaded Full-Face Image", use_column_width=True)

    w, h = full_image.size
    mid = w // 2
    left_face = full_image.crop((0, 0, mid, h))
    right_face = full_image.crop((mid, 0, w, h))

    stroke_input = full_image.resize((128, 128))
    stroke_array = np.array(stroke_input).astype("float32") / 255.0
    stroke_array = np.expand_dims(stroke_array, axis=0)
    st.write("🧠 Running stroke model prediction...")
    stroke_pred = stroke_model.predict(stroke_array)
    affected = int(np.round(stroke_pred[0][0]))

    unaffected_face = right_face if affected == 0 else left_face
    unaffected_tensor = transform(unaffected_face).unsqueeze(0)

    st.write("📈 Predicting pain score...")
    with torch.no_grad():
        output = pain_model(unaffected_tensor)
        pspi_score = output.item()

    st.subheader("Prediction Results")
    st.image(unaffected_face, caption="Unaffected Side Used for Pain Detection", width=300)
    st.write(f"**Affected side:** {'left' if affected == 0 else 'right'}")
    st.write(f"**Unaffected side:** {'right' if affected == 0 else 'left'}")
    st.write(f"**Predicted PSPI Pain Score:** {round(pspi_score, 3)}")