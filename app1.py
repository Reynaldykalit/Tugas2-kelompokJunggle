import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ===============================
# 1. KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Animals-10 Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# ===============================
# 2. CUSTOM CSS (UI THEME)
# ===============================
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #065f46;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-top: 20px;
    }
    .confidence {
        font-size: 22px;
        font-weight: bold;
        color: #047857;
    }
    .ood {
        color: #b91c1c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# 3. LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model_path = 'saved_model'
    if not os.path.exists(model_path):
        st.error(f"Folder '{model_path}' tidak ditemukan.")
        return None
    return tf.saved_model.load(model_path)

model = load_model()

# ===============================
# 4. DEFINISI KELAS
# ===============================
class_names = [
    'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
    'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
]

translate = {
    'cane': 'Anjing',
    'cavallo': 'Kuda',
    'elefante': 'Gajah',
    'farfalla': 'Kupu-kupu',
    'gallina': 'Ayam',
    'gatto': 'Kucing',
    'mucca': 'Sapi',
    'pecora': 'Domba',
    'ragno': 'Laba-laba',
    'scoiattolo': 'Tupai'
}

CONFIDENCE_THRESHOLD = 60.0

# ===============================
# 5. FUNGSI PREDIKSI
# ===============================
def import_and_predict(image_data, model):
    image_data = image_data.convert('RGB')
    image = ImageOps.fit(image_data, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    infer = model.signatures["serving_default"]
    output = infer(input_tensor)
    prediction = list(output.values())[0]

    return prediction.numpy()

# ===============================
# 6. SIDEBAR
# ===============================
st.sidebar.title("📌 Informasi Proyek")
st.sidebar.markdown("""
**Model** : MobileNetV2  
**Dataset** : Animals-10 (Kaggle)  
**Jumlah Kelas** : 10  
**Metode OOD** : Confidence Threshold  
""")

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini mendeteksi apakah gambar termasuk salah satu dari 10 kelas hewan.")

# ===============================
# 7. HEADER UTAMA
# ===============================
st.title("🐾 Klasifikasi Hewan Animals-10")
st.write(
    "Unggah gambar hewan, lalu sistem akan memprediksi jenis hewan "
    "berdasarkan model *Deep Learning MobileNetV2*."
)

# ===============================
# 8. UPLOAD GAMBAR
# ===============================
file = st.file_uploader(
    "📷 Upload Gambar Hewan (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

if file:
    image = Image.open(file)

    st.markdown("### 🖼️ Preview Gambar")
    st.image(image, width=300)

    if st.button("🔍 Prediksi Hewan"):
        with st.spinner("Model sedang menganalisis gambar..."):
            pred = import_and_predict(image, model)

        skor = pred[0]
        max_idx = np.argmax(skor)
        max_score = np.max(skor) * 100

        label = translate[class_names[max_idx]]

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if max_score < CONFIDENCE_THRESHOLD:
            st.markdown(
                "<h3 class='ood'>❌ Objek Tidak Dikenali</h3>",
                unsafe_allow_html=True
            )
            st.write(
                "Gambar berada **di luar domain dataset Animals-10**."
            )
        else:
            st.success(f"✅ Hewan Terdeteksi: **{label}**")
            st.markdown(
                f"<p class='confidence'>Confidence: {max_score:.2f}%</p>",
                unsafe_allow_html=True
            )

            st.markdown("#### 📊 Distribusi Probabilitas")
            chart_data = {
                translate[k]: float(v)
                for k, v in zip(class_names, skor)
            }
            st.bar_chart(chart_data)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("⬆️ Silakan upload gambar terlebih dahulu untuk melakukan prediksi.")
