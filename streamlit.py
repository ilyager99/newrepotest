import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder

class ModelAPI:
    def __init__(self, host: str, port: int):
        self.base_url = f"{host}:{port}/api/v1/models"

    def fit_model(self, params: dict):
        """Отправка параметров для обучения модели."""
        response = requests.post(f"{self.base_url}/fit", json=params)
        return response.json()

    def get_model_info(self, model_id: str):
        """Получение информации об обученной модели."""
        response = requests.get(f"{self.base_url}/info/{model_id}")
        return response.json()

host = "http://****"  # Замените на рабочий хост
port = 8000          # Замените на рабочий порт
api_client = ModelAPI(host, port)

st.title("Модель по анализу данных")

if 'page' not in st.session_state:
    st.session_state.page = "🔄 Обучение модели"

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("🔄 Обучение модели"):
        st.session_state.page = "🔄 Обучение модели"
with col2:
    if st.button("ℹ️ Информация о модели"):
        st.session_state.page = "ℹ️ Информация о модели"
with col3:
    if st.button("🔮 Предсказания"):
        st.session_state.page = "🔮 Предсказания"

# Переменная для хранения загруженных данных
# Переменные для хранения данных и моделей
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'model' not in st.session_state:
    st.session_state.model = None  # Для хранения модели после обучения
if 'model_id' not in st.session_state:
    st.session_state.model_id = None  # Для хранения ID модели
if 'models' not in st.session_state:
    st.session_state.models = {}  # Для хранения модели и её ID после обучения

if st.session_state.page == "🔄 Обучение модели":
    st.header("Обучение модели")
@@ -138,7 +135,7 @@
            st.write(f"📉 Стандартное отклонение точности: {std_accuracy:.4f}")

            # Сохраняем модель и model_id в состоянии сессии
            st.session_state['model'] = model
            st.session_state.models[params["model_id"]] = model  # Сохраняем модель под её ID
            st.session_state['model_id'] = params["model_id"]

            if type_of_model == "🧠 CatBoost Classifier":
@@ -160,6 +157,9 @@
            account_ids = data['account_id'].unique()
            account_id_input = st.selectbox("Выберите Account ID для предсказания", account_ids)

            # Выбор модели для предсказания
            model_id_input = st.selectbox("Выберите ID модели", list(st.session_state.models.keys()))
            if st.button("🔮 Получить предсказание"):
                account_data = data[data['account_id'] == account_id_input]

@@ -175,40 +175,40 @@
                        X_predict[col] = le.fit_transform(X_predict[col].astype(str))

                    # Проверяем, была ли обучена модель
                    if 'model' in st.session_state and st.session_state.model is not None:
                        model = st.session_state['model']
                    if model_id_input in st.session_state.models:
                        model = st.session_state.models[model_id_input]
                    else:
                        st.error("Модель не была обучена. Сначала обучите модель.")
                        st.stop()

                    # Получение предсказания
                    if isinstance(model, CatBoostClassifier):
                        probability = model.predict_proba(X_predict)[:, 1]  # Вероятность победы для CatBoost
                    else:  # Ridge Classifier
                        probability = model.predict(X_predict)  # Для Ridge использовать предсказание

                    st.write(f"Вероятность победы для Account ID {account_id_input}: {probability[0]:.2f}")
        else:
            st.error("Данные не содержат столбца 'account_id'.")

elif st.session_state.page == "ℹ️ Информация о модели":
    st.header("Информация о модели")

    model_id = st.text_input("Введите ID модели для получения информации", value="model")

    if st.button("📖 Получить информацию о модели"):
        model_info = api_client.get_model_info(model_id)
        if model_info:
            st.write("📝 Информация о модели:")
            st.json(model_info)

            if "feature_importances" in model_info:
                st.write("📊 Важность признаков:")
                feature_importances = model_info["feature_importances"]
                feature_importances_df = pd.DataFrame({
                    "Feature": feature_importances.keys(),
                    "Importance": feature_importances.values()
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(feature_importances_df.set_index("Feature"))
        else:
            st.error("❌ Такой модельки нет, sorry :(")
