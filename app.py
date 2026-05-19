import streamlit as st
import tensorflow as tf

from PIL import Image

from utils.preprocess import preprocess_image


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered"
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    base_model = tf.keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False


    model = tf.keras.Sequential([

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation='relu'
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            1,
            activation='sigmoid'
        )

    ])


    model.load_weights(
        "model/mobilenet_weights.weights.h5"
    )

    return model


model = load_model()


# =====================================================
# TITLE
# =====================================================

st.title("Pneumonia Detection Using Chest X-Ray")


st.write(
    """
    Upload a chest X-ray image to predict whether
    the patient has pneumonia or not.
    """
)


# =====================================================
# WARNING
# =====================================================

st.warning(
    """
    This application is for educational purposes only.
    It should not be used for real medical diagnosis.
    """
)


# =====================================================
# CLASS INFO
# =====================================================

st.info(
    """
    Classes:
    - NORMAL
    - PNEUMONIA
    """
)


# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")


    st.image(
        image,
        caption="Uploaded X-Ray Image",
        width=300
    )


    processed_image = preprocess_image(image)


    prediction = model.predict(processed_image)


    probability = float(prediction[0][0])


    threshold = 0.6


    if probability > threshold:

        label = "PNEUMONIA"

        confidence = probability

    else:

        label = "NORMAL"

        confidence = 1 - probability


    # =====================================================
    # RESULTS
    # =====================================================

    st.subheader(f"Prediction: {label}")


    st.write(
        f"Pneumonia Probability: {probability:.4f}"
    )


    st.write(
        f"Confidence: {confidence:.4f}"
    )


    st.progress(float(confidence))


    # =====================================================
    # EXTRA MESSAGE
    # =====================================================

    if label == "PNEUMONIA":

        st.error(
            "Model predicts signs of pneumonia."
        )

    else:

        st.success(
            "Model predicts normal chest X-ray."
        )