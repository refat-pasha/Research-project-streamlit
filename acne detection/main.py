import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings

# ── ENV FLAGS (Must be set before importing TensorFlow) ──────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # hide oneDNN info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # hide TF INFO logs

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.saving import register_keras_serializable

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
    .main { padding-top: 2rem; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .result-success { background: #d4edda; color: #155724; padding: 1rem; border-radius:5px; border:1px solid #c3e6cb; }
    .result-warning { background: #fff3cd; color: #856404; padding: 1rem; border-radius:5px; border:1px solid #ffeaa7; }
    .result-danger  { background: #f8d7da; color:#721c24; padding:1rem; border-radius:5px; border:1px solid #f1b0b7; }
    .info-box { background:#e7f3ff; color:#0c5460; padding:1rem; border-radius:5px; border-left:4px solid #b8daff; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── MODEL LOADING ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vgg19_model():
    candidates = ["model.keras", "model.h5", "vgg19_model.keras", "vgg19_model.h5"]
    for path in candidates:
        if os.path.exists(path):
            try:
                model = keras_load_model(path, compile=False)
                st.success(f"✅ Loaded model from '{path}'")
                return model
            except Exception as e:
                st.error(f"❌ Error loading '{path}': {e}")
                return None
    st.error(f"❌ No model file found (tried: {', '.join(candidates)})")
    return None

def preprocess_image(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return vgg19_preprocess(arr)

def reshape_for_flat_model(arr: np.ndarray, input_shape):
    num_features = int(np.prod(input_shape[1:]))
    size = int(np.sqrt(num_features))
    flat_img = tf.image.resize(tf.image.rgb_to_grayscale(arr), (size, size))
    flat_img = tf.reshape(flat_img, (arr.shape[0], num_features))
    return flat_img.numpy()

def predict_acne(model, arr: np.ndarray):
    # if model expects 4D image input
    if len(model.input_shape) == 4:
        inp = arr  # (1,224,224,3)
    else:
        inp = reshape_for_flat_model(arr, model.input_shape)
    preds = model.predict(inp, verbose=0)[0]
    # ensure preds length matches class_names
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    return idx, conf, preds

def display_results(idx, conf, probs, class_names):
    st.markdown("## 🔍 Prediction Results")
    col1, col2 = st.columns([2,1])

    # Map idx to label safely
    if 0 <= idx < len(class_names):
        label = class_names[idx]
    else:
        label = "Unknown"

    with col1:
        if label.lower() == "acne":
            st.markdown(f"""<div class="result-danger">
                <h3>⚠️ Prediction: {label}</h3>
                <p><strong>Confidence:</strong> {conf:.2%}</p>
                <p>Signs of acne detected.</p>
            </div>""", unsafe_allow_html=True)
        elif label.lower() == "healthy":
            st.markdown(f"""<div class="result-success">
                <h3>✅ Prediction: {label}</h3>
                <p><strong>Confidence:</strong> {conf:.2%}</p>
                <p>No acne detected.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-warning">
                <h3>❓ Prediction: {label}</h3>
                <p><strong>Confidence:</strong> {conf:.2%}</p>
                <p>Unable to map prediction to a known class.</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.metric("Confidence Score", f"{conf:.2%}")
        color = "green" if conf > 0.7 else "orange" if conf > 0.5 else "red"
        st.markdown(f"""<div style="background:#f0f0f0;border-radius:10px;padding:5px;">
            <div style="background:{color};width:{conf*100:.1f}%;height:20px;border-radius:5px;text-align:center;color:white;font-weight:bold;line-height:20px;">
                {conf:.1%}
            </div>
        </div>""", unsafe_allow_html=True)

    # Build DataFrame with matching lengths
    n = len(class_names)
    probs_trimmed = probs[:n]
    df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs_trimmed,
        "Percentage": [f"{p:.2%}" for p in probs_trimmed]
    })

    st.markdown("### 📊 Detailed Probabilities")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Class")["Probability"])

def display_medical_disclaimer():
    st.markdown("""<div class="info-box">
        <h4>⚕️ Medical Disclaimer</h4>
        This tool is for informational purposes only. Consult a professional for medical advice.
    </div>""", unsafe_allow_html=True)

def validate_image(file) -> Image.Image or None:
    if file.size > 10*1024*1024:
        st.error("❌ File too large (>10 MB).")
        return None
    try:
        img = Image.open(file)
    except:
        st.error("❌ Invalid image.")
        return None
    if img.width < 100 or img.height < 100:
        st.warning("⚠️ Low resolution; results may be less accurate.")
    return img

# ── MAIN ─────────────────────────────────────────────────────
def main():
    st.title("🩺 Acne Skin Disease Prediction")
    st.markdown("### AI-Powered Skin Analysis with VGG19")

    model = load_vgg19_model()
    if model is None:
        st.stop()

    st.sidebar.header("📋 How to Use")
    st.sidebar.write("1. Upload a JPG/PNG image\n2. Wait for results\n3. Consult a professional")
    st.sidebar.header("ℹ️ About Model")
    st.sidebar.write(f"- Input shape: {model.input_shape}\n- Classes: Acne, Healthy")

    class_names = ["Acne", "Healthy"]
    uploaded = st.file_uploader("Upload Skin Image", type=["jpg","jpeg","png"])
    if uploaded:
        img = validate_image(uploaded)
        if img:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            arr = preprocess_image(img)
            idx, conf, probs = predict_acne(model, arr)
            display_results(idx, conf, probs, class_names)

            if conf < 0.6:
                st.markdown("""<div class="result-warning">
                    <h4>⚠️ Low Confidence</h4>
                    Consider a clearer image or see a dermatologist.
                </div>""", unsafe_allow_html=True)

            if st.button("📥 Download Results"):
                rec = {
                    "Prediction": class_names[idx] if 0 <= idx < len(class_names) else "Unknown",
                    "Confidence": f"{conf:.2%}",
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                df = pd.DataFrame([rec])
                st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
    else:
        st.info("👆 Upload an image to begin")

    display_medical_disclaimer()
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#666;'>🤖 Powered by VGG19 & Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
