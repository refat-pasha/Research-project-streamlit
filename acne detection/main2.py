import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.saving import register_keras_serializable

# ── ENV FLAGS ───────────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ── REGISTER CUSTOM LAMBDA FUNCTION ─────────────────────────
@register_keras_serializable()
def my_lambda_function(x):
    return x

# ── STREAMLIT CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Acne Skin Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
  .result-success { background: #d4edda; color: #155724; padding: 1rem; border-radius:5px; }
  .result-warning { background: #fff3cd; color: #856404; padding: 1rem; border-radius:5px; }
  .result-danger  { background: #f8d7da; color:#721c24; padding:1rem; border-radius:5px; }
  .info-box { background:#e7f3ff; color:#0c5460; padding:1rem; border-radius:5px; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── MODEL LOADING ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    candidates = ["model.keras"]
    for path in candidates:
        if os.path.exists(path):
            try:
                return keras_load_model(path, compile=False)
            except Exception as e:
                st.error(f"Error loading '{path}': {e}")
                return None
    st.error(f"No model file found (tried: {', '.join(candidates)})")
    return None

def preprocess_for_cnn(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)[None, ...]
    return vgg19_preprocess(arr)

def preprocess_for_flat(img: Image.Image, num_features: int) -> np.ndarray:
    # model expects flat vector of length num_features = 784
    size = int(np.sqrt(num_features))
    gray = img.convert("L").resize((size, size))
    arr = np.array(gray, dtype=np.float32).reshape(1, num_features)
    # normalize to [0,1] or as in training
    return arr / 255.0

def predict(model, img: Image.Image):
    input_shape = model.input_shape  # e.g. (None, 224,224,3) or (None,784)
    if len(input_shape) == 4:
        inp = preprocess_for_cnn(img)
    else:
        num_features = int(input_shape[-1])
        inp = preprocess_for_flat(img, num_features)
    preds = model.predict(inp, verbose=0)[0]
    return preds

def display_results(preds):
    st.markdown("## 🔍 Prediction Results")
    # Always treat first two probabilities as Acne vs Healthy
    acne_prob = float(preds[0])
    healthy_prob = float(preds[1])
    if acne_prob >= healthy_prob:
        label, box, icon, desc, conf = (
            "Acne", "result-danger", "⚠️",
            "Signs of acne detected.", acne_prob
        )
    else:
        label, box, icon, desc, conf = (
            "Healthy", "result-success", "✅",
            "No acne detected.", healthy_prob
        )
    st.markdown(f"""
      <div class="{box}">
        <h3>{icon} Prediction: {label} ({conf:.2%})</h3>
        <p>{desc}</p>
      </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**Acne Confidence:** {acne_prob:.2%}   |   **Healthy Confidence:** {healthy_prob:.2%}")
    df = pd.DataFrame({
        "Class": ["Acne", "Healthy"],
        "Probability": [acne_prob, healthy_prob],
        "Percentage": [f"{acne_prob:.2%}", f"{healthy_prob:.2%}"]
    })
    st.markdown("### 📊 Detailed Probabilities")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Class")["Probability"])

def display_disclaimer():
    st.markdown("""<div class="info-box">
      <h4>⚕️ Medical Disclaimer</h4>
      This tool is for informational purposes only. Consult a professional.
    </div>""", unsafe_allow_html=True)

def validate_image(uploaded) -> Image.Image or None:
    if uploaded.size > 10 * 1024 * 1024:
        st.error("❌ File too large (>10 MB).")
        return None
    try:
        img = Image.open(uploaded)
    except:
        st.error("❌ Invalid image.")
        return None
    if img.width < 100 or img.height < 100:
        st.warning("⚠️ Low resolution; results may be less accurate.")
    return img

def main():
    st.title("🩺 Acne Skin Disease Prediction")
    st.markdown("### AI-Powered Skin Analysis with VGG19")

    model = load_model()
    if model is None:
        return

    st.sidebar.header("📋 How to Use")
    st.sidebar.write("1. Upload a JPG/PNG image\n2. Wait for results\n3. Consult a professional")
    st.sidebar.header("ℹ️ About Model")
    st.sidebar.write(f"- Input shape: {model.input_shape}\n- Classes: Acne, Healthy")

    uploaded = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = validate_image(uploaded)
        if img:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            preds = predict(model, img)
            display_results(preds)
            if max(preds[0], preds[1]) < 0.6:
                st.markdown("""<div class="result-warning">
                  <h4>⚠️ Low Confidence</h4>
                  Consider a clearer image or see a dermatologist.
                </div>""", unsafe_allow_html=True)
            if st.button("📥 Download Results"):
                rec = {
                    "Prediction": "Acne" if preds[0] >= preds[1] else "Healthy",
                    "Acne Confidence": f"{preds[0]:.2%}",
                    "Healthy Confidence": f"{preds[1]:.2%}",
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                df = pd.DataFrame([rec])
                st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
    else:
        st.info("👆 Upload an image to begin")

    display_disclaimer()
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#666;'>🤖 Powered by VGG19 & Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
