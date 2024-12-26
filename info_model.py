import streamlit as st
import requests

# Функция для получения информации о модели
def model_info():
    st.subheader("Информация о модели")
    model_id = st.text_input("Введите ID модели для получения информации", value="model")

    # API клиент
    host = "http://****"  # Замените на рабочий хост
    port = 8000           # Замените на рабочий порт

    api_client = ModelAPI(host, port)

    if st.button("Получить информацию о модели"):
        model_info = api_client.get_model_info(model_id)
        if model_info:
            st.write("Информация о модели:")
            st.json(model_info)

            # важность признаков
            if "feature_importances" in model_info:
                st.write("Важность признаков:")
                feature_importances = model_info["feature_importances"]
                feature_importances_df = pd.DataFrame({
                    "Feature": feature_importances.keys(),
                    "Importance": feature_importances.values()
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(feature_importances_df.set_index("Feature"))
        else:
            st.error("Такой модельки нет, sorry :(")

# Классы API
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