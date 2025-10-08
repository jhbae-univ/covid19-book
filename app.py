# =========================
# app.py (전체 소스)
# =========================

# Core Pkgs
import streamlit as st
st.set_page_config(
    page_title="Covid19 Detection Tool",
    page_icon="covid19.jpg",            # 파일과 동일 확장자 사용
    layout="centered",
    initial_sidebar_state="auto",
)

import time
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf

# ----- add imports for model patching -----
from pathlib import Path
import h5py, json
# -----------------------------------------

# ====== 5D 입력으로 저장된 .h5를 4D로 자동 패치해 로드하는 유틸 ======

def _fix_h5_input_to_4d(src_path: Path, dst_path: Path):
    """
    src_path(.h5)의 model_config에서 batch_input_shape가 [None, None, H, W, C]로 저장된 경우
    [None, H, W, C]로 고쳐 동일 구조의 모델을 만들고 원본 가중치를 주입하여 dst_path로 저장.
    """
    with h5py.File(str(src_path), "r") as f:
        if "model_config" not in f.attrs:
            raise ValueError("model_config 가 없습니다. 이 파일은 패치 방식으로 고칠 수 없습니다.")
        cfg_raw = f.attrs["model_config"]

    cfg = json.loads(cfg_raw)

    # 첫 레이어(config 경로는 모델마다 약간 다를 수 있으므로 방어적으로 처리)
    layers = cfg.get("config", {}).get("layers", [])
    if not layers:
        raise ValueError("layers 정보가 없습니다.")

    first = layers[0].get("config", {})
    changed = False

    def _to4d(shape):
        # [None, None, H, W, C] -> [None, H, W, C]
        if isinstance(shape, list) and len(shape) == 5 and shape[0] is None and shape[1] is None:
            return [shape[0]] + shape[2:]
        return shape

    if "batch_input_shape" in first:
        new_shape = _to4d(first["batch_input_shape"])
        if new_shape != first["batch_input_shape"]:
            first["batch_input_shape"] = new_shape
            changed = True

    if "build_input_shape" in first:
        new_shape = _to4d(first["build_input_shape"])
        if new_shape != first["build_input_shape"]:
            first["build_input_shape"] = new_shape
            changed = True

    # 추가: Input 레이어의 batch_shape 처리 (5D -> 4D)
    if "batch_shape" in first:
        new_shape = _to4d(first["batch_shape"])
        if new_shape != first["batch_shape"]:
            first["batch_shape"] = new_shape
            changed = True

    if not changed:
        # 바꿀 게 없으면 그대로 복사 저장
        dst_path.write_bytes(src_path.read_bytes())
        return

    # 수정된 config로 빈 모델 생성
    fixed_json = json.dumps(cfg)
    model = tf.keras.models.model_from_json(fixed_json)

    # 가중치 로드(이름 기준). 구조가 완전히 같아야 함.
    model.load_weights(str(src_path), by_name=True, skip_mismatch=False)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dst_path))

def load_model_4d_or_patch(model_path: Path) -> tf.keras.Model:
    """
    .h5 로드. 5D 입력 문제로 실패하면 자동으로 패치하여 *_fixed.h5로 저장 후 다시 로드.
    """
    try:
        return tf.keras.models.load_model(str(model_path))
    except Exception:
        # 5D 입력 문제일 가능성이 높을 때만 패치 시도
        fixed_path = model_path.with_name(model_path.stem + "_fixed.h5")
        _fix_h5_input_to_4d(model_path, fixed_path)
        return tf.keras.models.load_model(str(fixed_path))


# =========================
# Streamlit App
# =========================
def main():
    """Simple Tool for Covid-19 Detection from Chest X-Ray"""
    html_templ = """
    <div style="background-color:blue;padding:10px;">
      <h1 style="color:yellow">Covid-19 Detection Tool</h1>
    </div>
    """
    st.markdown(html_templ, unsafe_allow_html=True)
    st.write("A simple proposal for Covid-19 Diagnosis powered by Deep Learning and Streamlit")

    # 사이드바 헤더 이미지 (최신 API: width='stretch')
    try:
        st.sidebar.image("covid19.jpg", width="stretch")
    except Exception:
        pass  # 이미지가 없어도 앱은 계속 동작

    image_file = st.sidebar.file_uploader(
        "Upload an X-Ray Image (jpg, png or jpeg)",
        type=["jpg", "png", "jpeg"],
    )

    if image_file is not None:
        our_image = Image.open(image_file)

        if st.sidebar.button("Image Preview"):
            st.sidebar.image(our_image, width="stretch")

        activities = ["Image Enhancement", "Diagnosis", "Disclaimer and Info"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        # ---------------- Image Enhancement ----------------
        if choice == "Image Enhancement":
            st.subheader("Image Enhancement")

            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Contrast", "Brightness"])

            if enhance_type == "Contrast":
                c_rate = st.slider("Contrast", 0.5, 5.0, 1.0)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, width="stretch")

            elif enhance_type == "Brightness":
                c_rate = st.slider("Brightness", 0.5, 5.0, 1.0)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, width="stretch")

            else:
                st.text("Original Image")
                st.image(our_image, width="stretch")

        # ---------------- Diagnosis ----------------
        elif choice == "Diagnosis":
            if st.sidebar.button("Diagnosis"):
                # 1) PIL -> NumPy, 명시 상수로 그레이 변환
                np_img = np.array(our_image.convert("RGB"))
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

                st.text("Chest X-Ray")
                st.image(gray, width="stretch")

                # 2) 전처리
                IMG_SIZE = (200, 200)
                img = cv2.equalizeHist(gray)                 # uint8, 단일채널
                img = cv2.resize(img, IMG_SIZE)              # (200, 200)
                img = img.astype("float32") / 255.0          # 정규화
                X_Ray = img.reshape(1, 200, 200, 1)          # (배치, H, W, C) = 4D

                # 3) 모델 로드 (5D -> 4D 자동 패치 지원)
                BASE_DIR = Path(__file__).resolve().parent
                MODEL_PATH = BASE_DIR / "models" / "Covid19_CNN_Classifier.h5"

                if not MODEL_PATH.exists():
                    st.error(f"모델 파일이 없습니다: {MODEL_PATH}")
                    st.stop()

                try:
                    model = load_model_4d_or_patch(MODEL_PATH)
                except Exception as e:
                    st.exception(e)
                    st.stop()

                # 디버그용: 입력 모양 확인
                # st.write(f"model.input_shape = {model.input_shape}")
                # st.write(f"X_Ray.shape = {X_Ray.shape}")

                # 4) 예측
                diagnosis_proba = model.predict(X_Ray)
                diagnosis = int(np.argmax(diagnosis_proba, axis=1)[0])

                # 5) 진행바(연출)
                my_bar = st.sidebar.progress(0)
                for p in range(100):
                    time.sleep(0.01)
                    my_bar.progress(p + 1)

                # 6) 결과 표시 (0: No-Covid, 1: Covid)
                if diagnosis == 0:
                    st.sidebar.success("DIAGNOSIS: NO COVID-19")
                else:
                    st.sidebar.error("DIAGNOSIS: COVID-19")

                st.caption(
                    "This Web App is a DEMO for Streamlit and AI; predictions have no clinical value."
                )

        # ---------------- Disclaimer and Info ----------------
        else:
            st.subheader("Disclaimer and Info")
            st.subheader("Disclaimer")
            st.write("**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**")
            st.write("**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**")
            st.subheader("Info")
            st.write("This Tool gets inspiration from the following works:")
            st.write("- Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning")
            st.write("- Fighting Corona Virus with Artificial Intelligence & Deep Learning")
            st.write("- Deep Learning per la Diagnosi del COVID-19")
            st.write("We used small datasets with augmentation; limitations apply.")

    # ---------------- About the Author ----------------
    if st.sidebar.button("About the Author"):
        st.sidebar.subheader("Covid-19 Detection Tool")
        st.sidebar.markdown("by [Author's Name](https://www.authorswebsite.com)")
        st.sidebar.markdown("[author@gmail.com](mailto:author@gmail.com)")
        st.sidebar.text("All Rights Reserved (2023)")


if __name__ == "__main__":
    main()