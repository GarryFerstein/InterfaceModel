# Импорт библиотек

import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor

# ----------------------------
# Конфигурация признаков
# ----------------------------
FEATURES = [
    'brand', 'processor', 'CPU', 'Ram', 'Ram_type', 'ROM', 'ROM_type',
    'GPU', 'display_size', 'resolution_width', 'resolution_height', 'OS'
]
CAT_FEATURES = ['brand', 'processor', 'CPU', 'Ram_type', 'ROM_type', 'GPU', 'OS']

# ----------------------------
# Загрузка предобученной модели
# ----------------------------
MODEL_PATH = "laptop_price_model.cbm"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Файл модели не найден: `{MODEL_PATH}`. Убедитесь, что он лежит в той же папке, что и `app.py`.")
        st.stop()
    model = CatBoostRegressor().load_model(MODEL_PATH)
    return model

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Прогноз цен на ноутбуки", layout="wide")
st.title("💻 Прогноз цен на ноутбуки")
st.markdown("Модель уже обучена. Просто укажите характеристики ноутбука или загрузите CSV.")

tab_infer_manual, tab_infer_csv = st.tabs(["Предсказать вручную", "Предсказать из CSV"])

# ============================
# Вкладка 1: Инференс вручную
# ============================
with tab_infer_manual:
    col1, col2, col3 = st.columns(3)
    with col1:
        brand = st.text_input("Бренд", "Lenovo")
        processor = st.text_input("Процессор", "Intel Core i5")
        CPU = st.text_input("Модель CPU", "i5-1135G7")
        Ram = st.number_input("Оперативная память (GB)", min_value=2, max_value=128, value=16)
        Ram_type = st.selectbox("Тип RAM", ["DDR4", "DDR5", "LPDDR4", "Unknown"])

    with col2:
        ROM = st.number_input("Память (GB)", min_value=64, max_value=8192, value=512)
        ROM_type = st.selectbox("Тип накопителя", ["SSD", "HDD", "NVMe", "Unknown"])
        GPU = st.text_input("Видеокарта", "Intel Iris Xe")
        OS = st.selectbox("ОС", ["Windows", "Linux", "macOS", "Unknown"])

    with col3:
        display_size = st.number_input("Диагональ (дюймы)", min_value=10.0, max_value=20.0, value=15.6)
        resolution_width = st.number_input("Ширина разрешения", min_value=800, max_value=3840, value=1920)
        resolution_height = st.number_input("Высота разрешения", min_value=600, max_value=2160, value=1080)

    if st.button("Предсказать цену"):
        input_data = pd.DataFrame([{
            'brand': brand,
            'processor': processor,
            'CPU': CPU,
            'Ram': Ram,
            'Ram_type': Ram_type,
            'ROM': float(ROM),
            'ROM_type': ROM_type,
            'GPU': GPU,
            'display_size': display_size,
            'resolution_width': resolution_width,
            'resolution_height': resolution_height,
            'OS': OS
        }])

        pred = model.predict(input_data)[0]
        st.success(f"💰 Предсказанная цена: **{pred:,.0f} руб.**")

# ============================
# Вкладка 2: Инференс из CSV
# ============================
with tab_infer_csv:
    st.markdown("""
    Загрузите CSV-файл с колонками:
    ```
    brand, processor, CPU, Ram, Ram_type, ROM, ROM_type, GPU, display_size, resolution_width, resolution_height, OS
    ```
    """)

    infer_file = st.file_uploader("Выберите CSV-файл", type="csv")
    if infer_file:
        df_infer = pd.read_csv(infer_file)

        missing_cols = set(FEATURES) - set(df_infer.columns)
        if missing_cols:
            st.error(f"❌ Отсутствуют колонки: {missing_cols}")
        else:
            df_infer = df_infer[FEATURES].fillna("Unknown")
            preds = model.predict(df_infer)
            df_infer['predicted_price'] = preds

            st.write("Результаты:")
            st.dataframe(df_infer)

            csv = df_infer.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name="predicted_prices.csv",
                mime="text/csv"
            )