import streamlit as st
import requests
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier

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

# API клиент
host = "http://****"  # Замените на рабочий хост
port = ****           # Замените на рабочий порт
api_client = ModelAPI(host, port)

# Основная функция приложения
st.title("Модель по анализу данных")

# ID модели
model_id = st.text_input("Введите ID модели", value="model")

# Навигация
st.header("Навигация")
col1, col2 = st.columns(2)
with col1:
    if st.button("Обучение модели"):
        train_model(model_id)
with col2:
    if st.button("Информация о модели"):
        model_info(model_id)

def train_model(model_id):
    st.subheader("Обучение модели")
    
    # Типы модели
    type_of_model = st.selectbox("Выберите модель", ["Ridge Classifier", "CatBoost Classifier"])

    # Параметры моделей
    params = {"type_of_model": type_of_model, "model_id": model_id}

    # Гиперпараметры для классификаторов
    st.write("### Гиперпараметры модели")
    
    if type_of_model == "Ridge Classifier":
        params["alpha"] = st.number_input("Alpha", value=0.01, min_value=0.0)
        params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)

    elif type_of_model == "CatBoost Classifier":
        params["learning_rate"] = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
        params["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
        params["iterations"] = st.number_input("Iterations", value=100, min_value=1)
        params["l2_leaf_reg"] = st.number_input("L2 Leaf Regularization", value=3, min_value=1, max_value=10)

    # загрузка файла
    uploaded_file = st.file_uploader("Загрузите данные (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Данные:")
        st.write(data.head())

        # Используем фиксированную целевую переменную
        y = data['radiant_win']
        X = data.drop(columns=['radiant_win'])

        # обучение модели
        if st.button("Обучить модель"):
            params["train_data"] = data.to_dict(orient="list")
            start_time = time.time()  # засекаем время обучения модели

            # проводим кросс-валидацию
            if type_of_model == "Ridge Classifier":
                model = RidgeClassifier(alpha=params["alpha"], fit_intercept=params["fit_intercept"])
            elif type_of_model == "CatBoost Classifier":
                model = CatBoostClassifier(
                    learning_rate=params["learning_rate"],
                    depth=params["depth"],
                    iterations=params["iterations"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    verbose=False)

            st.write("Кросс-валидация началась")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_results = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                fold_results.append(accuracy)

            mean_accuracy = sum(fold_results) / len(fold_results)
            end_time = time.time()

            # обработка результата
            st.success("Модель обучена!")
            st.write(f"Время обучения составило: {end_time - start_time:.2f} сек")
            st.write("Результаты кросс-валидации:")
            st.write(pd.DataFrame({"Fold": range(1, 6), "Accuracy": fold_results}))
            st.write(f"Средняя точность: {mean_accuracy:.4f}")

            # Важность признаков для CatBoost
            if type_of_model == "CatBoost Classifier":
                feature_importances = model.get_feature_importance()
                feature_importances_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": feature_importances
                }).sort_values(by="Importance", ascending=False)
                st.write("Важность признаков:")
                st.bar_chart(feature_importances_df.set_index("Feature"))

def model_info(model_id):
    st.subheader("Информация о модели")

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
