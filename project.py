
# !!! ROAD DAMAGE CLASSIFICATION !!!

# Import required libraries
import streamlit as st                      # Web app framework
import numpy as np                          # Numerical operations
import tensorflow as tf                     # Load trained model
from PIL import Image                       # Image handling
import matplotlib.pyplot as plt             # Visualization

# 1-LOAD TRAINED MODEL

# Load the saved model file (.h5)
model = tf.keras.models.load_model(
    r"C:\Users\KAVIYA V\final_model_here.h5"
)

# Force model build (avoids lazy initialization issues)
dummy = np.zeros((1, 224, 224, 3))         # Create dummy input
_ = model(dummy, training=False)           # Run one forward pass

# Class labels
CLASS_NAMES = ["Crack", "Manhole", "Pothole"]

# Image size (same as training)
IMG_SIZE = (224, 224)

# 2-RECOMMENDATION FUNCTION

def get_recommendation(label):
    """
    Provide action recommendation based on predicted class
    """
    if label == "Pothole":
        return "⚠ Immediate repair required"             # Critical damage
    elif label == "Crack":
        return "🛠 Monitor and schedule maintenance"     # Moderate damage
    elif label == "Manhole":
        return "🔍 Inspect cover condition"             # Inspection needed
    return "No action"                                  # Default case

# 3-PREDICTION FUNCTION

def predict_image(image):
    """
    Preprocess image and perform prediction
    """
    img = image.resize(IMG_SIZE)                # Resize image
    img_array = np.array(img) / 255.0           # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    preds = model.predict(img_array)            # Predict probabilities
    class_idx = np.argmax(preds)                # Get predicted class index
    confidence = np.max(preds)                  # Get confidence score

    return preds[0], CLASS_NAMES[class_idx], confidence

# 4-SIDEBAR NAVIGATION

st.sidebar.title("Navigation")                  # Sidebar title

# Page selection
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "Results"]
)

# 5-HOME PAGE

if page == "Home":

    st.title("🚧 Road Damage Classification System")  # App title

    # Description
    st.write("""
    This system uses Deep Learning to classify road damages.

    !!! Features:
    - Detect Pothole, Crack, Manhole
    - Confidence Score
    - Recommendation System
    - Visualization Dashboard

    !!! Applications:
    - Smart Cities
    - Road Maintenance
    - Public Safety
    """)

# 6-PREDICTION PAGE

elif page == "Prediction":

    st.title("📸 Upload Road Image")          # Page title

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"]
    )

    # If image uploaded
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")   # Convert to RGB
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction button
        if st.button("Predict"):

            preds, label, confidence = predict_image(image)

            # Store results in session state (for multi-page use)
            st.session_state["preds"] = preds
            st.session_state["label"] = label
            st.session_state["confidence"] = confidence
            st.session_state["image"] = image

            st.success("Prediction completed! Go to Results page ➡️")

# 7-RESULTS PAGE

elif page == "Results":

    st.title("📊 Final Results Dashboard")     # Page title

    # Check if prediction exists
    if "preds" not in st.session_state:
        st.warning("⚠ Please make a prediction first")

    else:
        # Retrieve stored results
        image = st.session_state["image"]
        preds = st.session_state["preds"]
        label = st.session_state["label"]
        confidence = st.session_state["confidence"]

        # Show analyzed image
        st.image(image, caption="Analyzed Image", use_column_width=True)

        # Prediction summary
        st.subheader("Prediction Summary")
        st.write(f"**Detected Damage:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.write(f"**Recommendation:** {get_recommendation(label)}")

        # 7.1-VISUALIZATION: BAR CHART

        st.subheader("Prediction Probabilities")

        fig, ax = plt.subplots()               # Create plot
        ax.bar(CLASS_NAMES, preds)             # Bar chart
        ax.set_ylabel("Probability")           # Y-axis label
        ax.set_title("Model Confidence for Each Class")  # Title

        st.pyplot(fig)                         # Display chart

# 8-FINAL MESSAGE

print("FINAL APP READY")                   # Console message