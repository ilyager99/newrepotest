import streamlit as st
import requests
import time
import pandas as pd

# Класс для работы с API
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

# Инициализация API клиента
host = "http://localhost"  # Замените на ваш хост
port = 8000                # Замените на ваш порт
api_client = ModelAPI(host, port)

# Streamlit UI
st.title("Модель тренировки и анализа")

# Выбор типа модели
model_type = st.selectbox("Выберите модель", ["Linear Regression", "CatBoost"])

# Параметры для моделей
params = {"model_type": model_type}

if model_type == "Linear Regression":
    params["alpha"] = st.number_input("Alpha", value=0.01, min_value=0.0)
    params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)
    params["normalize"] = st.checkbox("Normalize", value=False)

elif model_type == "CatBoost":
    params["learning_rate"] = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
    params["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
    params["iterations"] = st.number_input("Iterations", value=100, min_value=1)

# Уникальный идентификатор модели
params["model_id"] = st.text_input("Введите Model ID", value="default_model")

# Загрузка тренировочного файла
uploaded_file = st.file_uploader("Загрузите тренировочные данные (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Пример данных:")
    st.write(data.head())

# Кнопка для отправки данных
if st.button("Обучить модель"):
    if uploaded_file is not None:
        params["train_data"] = data.to_dict(orient="list")
        start_time = time.time()  # Засекаем время
        response = api_client.fit_model(params)
        end_time = time.time()

        # Обработка ответа
        st.success("Модель обучена!")
        st.write(f"Время обучения: {end_time - start_time:.2f} секунд")
        st.write("Результаты кросс-валидации:")
        st.json(response.get("cross_validation_metrics", {}))

        # Важность признаков
        if "feature_importances" in response:
            st.write("Важность признаков:")
            feature_importances = response["feature_importances"]
            feature_importances_df = pd.DataFrame({
                "Feature": feature_importances.keys(),
                "Importance": feature_importances.values()
            }).sort_values(by="Importance", ascending=False)
            st.bar_chart(feature_importances_df.set_index("Feature"))

# Отображение информации об обученных моделях
st.header("Информация об обученной модели")
model_id = st.text_input("Введите Model ID для получения информации", value="default_model")

if st.button("Получить информацию о модели"):
    model_info = api_client.get_model_info(model_id)
    if model_info:
        st.write("Информация о модели:")
        st.json(model_info)

        # Важность признаков (если доступно)
        if "feature_importances" in model_info:
            st.write("Важность признаков:")
            feature_importances = model_info["feature_importances"]
            feature_importances_df = pd.DataFrame({
                "Feature": feature_importances.keys(),
                "Importance": feature_importances.values()
            }).sort_values(by="Importance", ascending=False)
            st.bar_chart(feature_importances_df.set_index("Feature"))
    else:
        st.error("Модель не найдена!")