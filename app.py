import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Klasifikasi Hewan Animals-10",
    page_icon="🐾",
    layout="centered"
)

# 2. LOAD MODEL MENGGUNAKAN tf.saved_model.load
@st.cache_resource
def load_model():
    model_path = 'saved_model'
    if not os.path.exists(model_path):
        st.error(f"Folder '{model_path}' tidak ditemukan.")
        return None
    
    try:
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# 3. DEFINISI KELAS (Urut Abjad)
class_names = [
    'cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
    'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
]

translate = {
    'cane': 'Anjing (Dog)',
    'cavallo': 'Kuda (Horse)',
    'elefante': 'Gajah (Elephant)',
    'farfalla': 'Kupu-kupu (Butterfly)',
    'gallina': 'Ayam (Chicken)',
    'gatto': 'Kucing (Cat)',
    'mucca': 'Sapi (Cow)',
    'pecora': 'Domba (Sheep)',
    'ragno': 'Laba-laba (Spider)',
    'scoiattolo': 'Tupai (Squirrel)'
}

# 4. FUNGSI PREDIKSI (DIPERBAIKI UNTUK RGBA -> RGB)
def import_and_predict(image_data, model):
    # --- PERBAIKAN DI SINI ---
    # Paksa ubah gambar ke format RGB (3 channel)
    # Ini membuang Alpha channel jika gambar adalah PNG transparan
    image_data = image_data.convert('RGB')
    
    # Resize ke 224x224
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Preprocessing
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # Ubah ke Tensor Float32
    input_tensor = tf.convert_to_tensor(img_reshape, dtype=tf.float32)
    
    # Panggil fungsi inference dari SavedModel
    inference_func = model.signatures["serving_default"]
    
    # Prediksi
    prediction_dict = inference_func(input_tensor)
    
    # Ambil hasil output pertama
    prediction_tensor = list(prediction_dict.values())[0]
    
    return prediction_tensor.numpy()

# 5. TAMPILAN DASHBOARD
st.title("🐾 Deteksi Hewan Ternak & Liar")
st.markdown("---")

st.sidebar.title("Info Proyek")
st.sidebar.info("Menggunakan MobileNetV2 (TF SavedModel).")

file = st.file_uploader("Upload gambar hewan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file is not None:
    # Buka gambar
    image = Image.open(file)
    st.image(image, caption='Gambar yang diupload', width=300)
    
    if st.button("🔍 Prediksi"):
        with st.spinner('Sedang menganalisis...'):
            if model is not None:
                try:
                    prediksi = import_and_predict(image, model)
                    
                    # Proses hasil
                    skor = prediksi[0]
                    kelas_teratas_index = np.argmax(skor)
                    label_asli = class_names[kelas_teratas_index]
                    label_indo = translate.get(label_asli, label_asli)
                    confidence = 100 * np.max(skor)
                    
                    st.success(f"Hewan Terdeteksi: **{label_indo}**")
                    st.metric("Tingkat Kepercayaan", f"{confidence:.2f}%")
                    
                    st.write("Statistik Probabilitas:")
                    chart_data = {translate.get(k, k): v for k, v in zip(class_names, skor)}
                    st.bar_chart(chart_data)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
            else:
                st.error("Model belum dimuat dengan benar.")
else:
    st.info("Silakan upload gambar.")