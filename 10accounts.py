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

    def get_fit_status(self):
        """Получение статуса асинхронной задачи обучения."""
        response = requests.get(f"{self.base_url}/fit/status")
        return response.json()

    def get_model_list(self):
        """Получение списка всех обученных моделей."""
        response = requests.get(f"{self.base_url}/list")
        return response.json()

    def activate_model(self, model_id: str):
        """Установка активной модели для прогноза."""
        response = requests.put(f"{self.base_url}/activate", json={"model_id": model_id})
        return response.json()

    def predict(self, data: dict):
        """Прогноз исхода на основе выбранных данных. Используется активированная модель."""
        response = requests.post(f"{self.base_url}/predict", json=data)
        return response.json()

    def predict_csv(self, csv_data):
        """Прогноз исхода на основе CSV-файла."""
        response = requests.post(f"{self.base_url}/predict_csv", files={"file": csv_data})
        return response.json()

    def get_model_info(self, model_id: str):
        """Получение информации об обученной модели."""
        response = requests.get(f"{self.base_url}/model_info", params={"model_id": model_id})
        return response.json()

    def get_account_ids(self):
        """Получение уникальных account_id из API."""
        response = requests.get(f"{self.base_url}/data/account_ids")
        if response.status_code == 200:
            return response.json()  # Предполагается, что API возвращает список account_id
        else:
            st.error("Не удалось получить Account IDs из API.")
            return []

# Инициализация API
host = "http://****"  # Замените на рабочий хост
port = 8000          # Замените на рабочий порт
api_client = ModelAPI(host, port)

# Заголовок приложения
st.title("Модель по анализу данных")

# Инициализация состояния сессии
if 'page' not in st.session_state:
    st.session_state.page = "🔄 Обучение модели"

# Создание вертикального меню с кнопками
st.sidebar.header("Меню быстрого доступа")
if st.sidebar.button("🔄 Обучение модели"):
    st.session_state.page = "🔄 Обучение модели"
if st.sidebar.button("ℹ️ Информация о модели"):
    st.session_state.page = "ℹ️ Информация о модели"
if st.sidebar.button("🔮 Предсказания"):
    st.session_state.page = "🔮 Предсказания"

# Переменные для хранения данных и моделей
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'model_id' not in st.session_state:
    st.session_state.model_id = None  # Для хранения ID модели
if 'models' not in st.session_state:
    st.session_state.models = []  # Для хранения обученных моделей

# Функции для получения гиперпараметров
def get_ridge_params(params):
    """Функция для ввода гиперпараметров Ridge Classifier."""
    params["alpha"] = st.number_input("Alpha", value=1.0, min_value=0.0)
    params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)

def get_catboost_params(params):
    """Функция для ввода гиперпараметров CatBoost Classifier."""
    params["learning_rate"] = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
    params["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
    params["iterations"] = st.number_input("Iterations", value=100, min_value=1)
    params["l2_leaf_reg"] = st.number_input("L2 Leaf Regularization", value=3, min_value=1, max_value=10)

if st.session_state.page == "🔄 Обучение модели":
    st.header("Обучение модели")

    type_of_model = st.selectbox("Выберите модель", ["⚖️ Ridge Classifier", "🧠 CatBoost Classifier"])
    params = {"type_of_model": type_of_model}

    st.subheader("Гиперпараметры модели")

    # Выбор гиперпараметров в зависимости от типа модели
    if type_of_model == "⚖️ Ridge Classifier":
        get_ridge_params(params)
    elif type_of_model == "🧠 CatBoost Classifier":
        get_catboost_params(params)

    params["model_id"] = st.text_input("Введите ID модели", value="model")

    if st.button("🚀 Обучить модель"):
        # Запуск обучения модели
        start_time = time.time()
        fit_response = api_client.fit_model(params)
        st.success("Обучение модели начато!")

        # Проверка статуса обучения
        while True:
            status_response = api_client.get_fit_status()
            st.write(f"Статус обучения: {status_response['status']}")
            if status_response['status'] in ["completed", "failed"]:
                break
            time.sleep(2)  # Задержка перед следующей проверкой

        end_time = time.time()
        st.write(f"⏳ Время обучения составило: {end_time - start_time:.2f} сек")

elif st.session_state.page == "🔮 Предсказания":
    st.header("Предсказания на основе обученной модели")

    # Получение account IDs
    account_ids_response = api_client.get_account_ids()
    if account_ids_response:
        account_ids = account_ids_response['account_ids']
    else:
        st.error("Не удалось загрузить account IDs.")
        st.stop()
    
    selected_account_ids = st.multiselect("Выберите Account IDs для предсказания", account_ids)

    if st.button("🔮 Получить предсказания"):
        predictions = []
        for account_id in selected_account_ids:
            data = {"account_id": account_id}  # Подготовка данных для запроса
            prediction_response = api_client.predict(data)
            predictions.append(prediction_response)

        for prediction in predictions:
            st.write(prediction)

    # Загрузка тестового датасета
    uploaded_test_file = st.file_uploader("📤 Загрузите тестовый датасет (CSV)", type=["csv"])
    if uploaded_test_file is not None and st.button("Отправить тестовый файл для предсказаний"):
        prediction_csv_response = api_client.predict_csv(uploaded_test_file)
        st.write("Предсказания из CSV файла:")
        st.json(prediction_csv_response)

elif st.session_state.page == "ℹ️ Информация о модели":
    st.header("Информация о модели")

    if st.button("Загрузить список обученных моделей"):
        model_list_response = api_client.get_model_list()
        if model_list_response:
            st.session_state.models = model_list_response['models']  # Сохраняем список моделей
            st.write("Обученные модели:")
            st.write(st.session_state.models)

    model_id_input = st.selectbox("Выберите ID модели", [model['id'] for model in st.session_state.models])

    if st.button("📖 Получить информацию о модели"):
        model_info = api_client.get_model_info(model_id_input)
        if model_info:
            st.write("📝 Информация о модели:")
            st.json(model_info)
        else:
            st.error("❌ Эта модель не существует.")
        
    if st.button("Активировать выбранную модель"):
        activate_response = api_client.activate_model(model_id_input)
        if activate_response.get("status") == "success":
            st.success("Модель успешно активирована!")
        else:
            st.error("Не удалось активировать модель.")
