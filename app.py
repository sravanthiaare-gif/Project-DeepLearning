import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Consumer Complaint Classifier",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Consumer Complaint Classification")
st.write("Enter a complaint and the model will predict its category.")

# ============================
# Load Label Encoder
# ============================
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ============================
# Load Tokenizer
# ============================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ============================
# Load Model Architecture
# ============================
model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# ============================
# Load Trained Weights
# ============================
model.load_weights("complaint_classifier_weights.h5")

MAX_LEN = 128


# ============================
# Prediction Function
# ============================
def predict(text):

    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

    outputs = model(tokens)

    logits = outputs.logits.numpy()

    predicted_class_id = np.argmax(logits)

    label = label_encoder.inverse_transform([predicted_class_id])[0]

    return label, logits


# ============================
# User Input
# ============================
complaint = st.text_area(
    "Enter Complaint Text",
    height=150
)

# ============================
# Prediction Button
# ============================
if st.button("Predict Category"):

    if complaint.strip() == "":
        st.warning("⚠ Please enter complaint text")

    else:

        label, logits = predict(complaint)

        st.success(f"Predicted Category: **{label}**")

        # Confidence Score
        probs = tf.nn.softmax(logits)[0]

        confidence = np.max(probs)

        st.write(f"Confidence Score: **{confidence:.2f}**")