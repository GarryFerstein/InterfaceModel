# Импорт библиотек

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ----------------------------
# Вспомогательные функции
# ----------------------------

def parse_rom(rom_str):
    if pd.isna(rom_str):
        return np.nan
    s = str(rom_str).strip().upper()
    if 'TB' in s:
        return float(s.replace('TB', '').strip()) * 1024
    elif 'GB' in s:
        return float(s.replace('GB', '').strip())
    else:
        try:
            return float(s)
        except:
            return np.nan

def preprocess_data(df):
    # Удаление лишнего столбца
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.dropna(subset=['price'])

    # RAM
    df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)

    # ROM
    df['ROM'] = df['ROM'].apply(parse_rom)

    # Display и разрешение
    df['display_size'] = pd.to_numeric(df['display_size'], errors='coerce')
    df['resolution_width'] = pd.to_numeric(df['resolution_width'], errors='coerce')
    df['resolution_height'] = pd.to_numeric(df['resolution_height'], errors='coerce')

    # Удаление строк с пропусками в ключевых признаках
    df = df.dropna(subset=['Ram', 'ROM', 'display_size', 'resolution_width', 'resolution_height', 'price'])

    return df

# ----------------------------
# Конфигурация признаков
# ----------------------------
FEATURES = [
    'brand', 'processor', 'CPU', 'Ram', 'Ram_type', 'ROM', 'ROM_type',
    'GPU', 'display_size', 'resolution_width', 'resolution_height', 'OS'
]
CAT_FEATURES = ['brand', 'processor', 'CPU', 'Ram_type', 'ROM_type', 'GPU', 'OS']
NUM_FEATURES = ['Ram', 'ROM', 'display_size', 'resolution_width', 'resolution_height']

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Прогноз цен на ноутбуки", layout="wide")
st.title("🧠 Прогноз цен на ноутбуки")

tab_train, tab_infer_manual, tab_infer_csv = st.tabs(["Обучение модели", "Инференс (вручную)", "Инференс (из CSV)"])

# ============================
# Вкладка 1: Обучение модели
# ============================
with tab_train:
    st.header("Обучение модели")
    uploaded_file = st.file_uploader("Загрузите обучающий датасет (CSV)", type="csv")
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.write("Первые 5 строк данных:")
        st.dataframe(df_raw.head())

        # Предобработка
        df = preprocess_data(df_raw.copy())
        st.success(f"После предобработки осталось {len(df)} записей.")

        # Гиперпараметры
        st.subheader("Гиперпараметры модели")
        col1, col2, col3 = st.columns(3)
        with col1:
            iterations = st.number_input("Iterations", min_value=100, max_value=5000, value=1000, step=100)
        with col2:
            lr = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        with col3:
            depth = st.slider("Depth", min_value=3, max_value=12, value=8)

        random_seed = st.number_input("Random Seed", value=42)

        if st.button("Обучить модель"):
            X = df[FEATURES].fillna("Unknown")
            y = df['price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

            cat_indices = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]

            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=lr,
                depth=depth,
                random_seed=random_seed,
                verbose=0,
                cat_features=cat_indices
            )

            with st.spinner("Обучение модели..."):
                model.fit(X_train, y_train)

            # Оценка
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            st.success("✅ Модель обучена!")
            st.metric("R² (тест)", f"{test_r2:.4f}")
            st.metric("MAE (тест)", f"{test_mae:.2f}")

            # Сохранение модели в сессию
            st.session_state['model'] = model
            st.session_state['features'] = FEATURES
            st.session_state['cat_features'] = CAT_FEATURES

            # График
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_test_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Реальная цена")
            ax.set_ylabel("Предсказанная цена")
            st.pyplot(fig)

# ============================
# Вкладка 2: Инференс вручную
# ============================
with tab_infer_manual:
    st.header("Предсказать цену (вручную)")

    if 'model' not in st.session_state:
        st.warning("Сначала обучите модель во вкладке 'Обучение модели'.")
    else:
        model = st.session_state['model']
        features = st.session_state['features']

        # Форма ввода
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
# Вкладка 3: Инференс из CSV
# ============================
with tab_infer_csv:
    st.header("Предсказать цены из CSV")

    if 'model' not in st.session_state:
        st.warning("Сначала обучите модель во вкладке 'Обучение модели'.")
    else:
        infer_file = st.file_uploader("Загрузите CSV для предсказания", type="csv")
        if infer_file:
            df_infer = pd.read_csv(infer_file)

            # Проверка наличия всех признаков
            missing = set(FEATURES) - set(df_infer.columns)
            if missing:
                st.error(f"В файле отсутствуют столбцы: {missing}")
            else:
                df_infer = df_infer[FEATURES].fillna("Unknown")
                preds = st.session_state['model'].predict(df_infer)
                df_infer['predicted_price'] = preds

                st.write("Результаты предсказания:")
                st.dataframe(df_infer)

                # Кнопка скачивания
                csv = df_infer.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать результаты",
                    data=csv,
                    file_name="predicted_prices.csv",
                    mime="text/csv"
                )