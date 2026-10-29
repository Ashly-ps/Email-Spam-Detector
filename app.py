import streamlit as st
import joblib
from pathlib import Path

# Find the saved model
MODEL_PATH = Path(__file__).parent / "spam_detector_model.joblib"

st.set_page_config(page_title="Email Spam Detector")

# Application title
st.title("Email Spam Detector")
st.write("Enter a message to check whether it may be spam.")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run train_model.py first.")
    st.stop()

# Input field
message = st.text_area("Enter your email message:")

# Detect spam
if st.button("Detect Spam"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("This message may be SPAM!")
        else:
            st.success("This message appears to be NOT SPAM.")
