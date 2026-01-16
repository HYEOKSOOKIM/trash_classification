# app.py
# Waste Classification Streamlit Demo
# TensorFlow SavedModel (export) vs OpenVINO inference comparison

import streamlit as st
import numpy as np
import time
from PIL import Image

import tensorflow as tf
from openvino.runtime import Core


# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(page_title="Waste Classification Demo", layout="centered")

CLASS_NAMES = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash"
]

IMG_SIZE = (224, 224)


# ==================================================
# 모델 로딩 (캐시)
# ==================================================
@st.cache_resource
def load_tf_models():
    """
    Keras 3 export()로 생성된 TensorFlow SavedModel 로드
    (서빙 전용, signatures 사용)
    """
    mn = tf.saved_model.load("mobilenetv2_infer_savedmodel")
    ef = tf.saved_model.load("efficientnetb0_infer_savedmodel")
    return mn, ef


@st.cache_resource
def load_ov_models():
    """
    OpenVINO IR 모델 로드 및 CPU 컴파일
    """
    ie = Core()

    mn_model = ie.read_model("ov_mobilenet/saved_model.xml")
    ef_model = ie.read_model("ov_efficientnet/saved_model.xml")

    mn_compiled = ie.compile_model(mn_model, "CPU")
    ef_compiled = ie.compile_model(ef_model, "CPU")

    return mn_compiled, ef_compiled


# ==================================================
# 공통 전처리
# ==================================================
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    x = np.array(img).astype(np.float32)
    return x


# ==================================================
# TensorFlow SavedModel 추론 (signatures 방식)
# ==================================================
def predict_tf(model, image: Image.Image, model_name: str):
    """
    TensorFlow SavedModel (export 기반) 추론 + latency 측정
    """
    x = preprocess_image(image)
    x = x[None, ...]

    if model_name == "MobileNetV2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    else:
        x = tf.keras.applications.efficientnet.preprocess_input(x)

    infer = model.signatures["serving_default"]

    start = time.time()
    outputs = infer(tf.constant(x))
    latency = (time.time() - start) * 1000

    preds = list(outputs.values())[0].numpy()[0]
    return preds, latency


# ==================================================
# OpenVINO 추론
# ==================================================
def predict_ov(compiled_model, image: Image.Image):
    x = preprocess_image(image)
    x = x[None, ...]

    input_layer = compiled_model.input(0)

    start = time.time()
    result = compiled_model({input_layer: x})
    latency = (time.time() - start) * 1000

    preds = list(result.values())[0][0]
    return preds, latency


# ==================================================
# UI
# ==================================================
st.title("♻️ 쓰레기 분류 데모")
st.caption("TensorFlow Model vs OpenVINO 추론 비교")

# ---------------- Sidebar ----------------
st.sidebar.header("추론 모델 세팅")

backend = st.sidebar.radio(
    "추론 백엔드",
    ["TensorFlow", "OpenVINO"]
)

model_name = st.sidebar.selectbox(
    "모델",
    ["MobileNetV2", "EfficientNet-B0"]
)

# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드 된 이미지", width=450)

    if st.button("추론 시작!"):
        with st.spinner("추론 중..."):

            if backend == "TensorFlow":
                mn, ef = load_tf_models()
                model = mn if model_name == "MobileNetV2" else ef
                preds, latency = predict_tf(model, image, model_name)

            else:
                mn_ov, ef_ov = load_ov_models()
                model = mn_ov if model_name == "MobileNetV2" else ef_ov
                preds, latency = predict_ov(model, image)

        # ---------------- Result ----------------
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        st.success(f"추론 결과: **{CLASS_NAMES[pred_idx]}**")
        st.metric("신뢰도", f"{confidence:.2%}")
        st.metric("추론 시간", f"{latency:.2f} ms")

        st.subheader("클래스 확률")
        st.bar_chart(
            {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        )

else:
    st.info("추론을 위해 이미지를 업로드 해주세요.")
