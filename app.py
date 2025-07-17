import streamlit as st
import joblib
import string
import pytesseract
from PIL import Image
import numpy as np
import cv2


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# OCR for images
def extract_text_from_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

# Prediction
def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return pred, round(proba * 100, 2)

# App layout
st.set_page_config(page_title="📰 AI Fake News Detector", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #1c1c1c;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #262730;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Fake News Detector (with Confidence & Image Support)")
st.write("Upload a news screenshot, .txt file, or paste the article below.")

# Upload or Text Input
col1, col2 = st.columns(2)

with col1:
    uploaded_img = st.file_uploader("📸 Upload an image (screenshot of article)", type=["png", "jpg", "jpeg"])
    uploaded_txt = st.file_uploader("📄 Or upload a .txt file", type=["txt"])

with col2:
    user_input = st.text_area("✍️ Or paste news text here:", height=300)

text = ""

if uploaded_img:
    img = Image.open(uploaded_img)
    text = extract_text_from_image(img)
    st.image(img, caption="Uploaded Image")
    st.write("📝 Extracted text:", text[:500] + "..." if len(text) > 500 else text)

elif uploaded_txt:
    text = uploaded_txt.read().decode("utf-8")
    st.write("📝 Uploaded text:", text[:500] + "..." if len(text) > 500 else text)

elif user_input.strip():
    text = user_input

# Predict
if text:
    if st.button("🔍 Check News"):
        pred, confidence = predict(text)
        st.markdown(f"### ✅ Prediction: **{pred.upper()}**")
        st.markdown(f"### 📊 Confidence: **{confidence}%**")

        # 🔍 Highlighting common fake cues (basic)
        cues = ["shocking", "unbelievable", "you won't believe", "click here", "must see", "urgent"]
        highlights = [cue for cue in cues if cue in text.lower()]
        if highlights:
            st.warning(f"⚠️ Suspicious keywords detected: {', '.join(highlights)}")
else:
    st.info("Please upload a file or enter text to continue.")
