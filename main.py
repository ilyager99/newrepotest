import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier

# Классы API
class ModelAPI:
    def __init__(self, host: str, port: int):
        self.base_url = f"{host}:{port}/api/v1/models"

    def fit_model(self, params: dict):
        response = requests.post(f"{self.base_url}/fit", json=params)
        return response.json()

    def get_model_info(self, model_id: str):
        response = requests.get(f"{self.base_url}/info/{model_id}")
        return response.json()

# API клиент
host = "http://****"  # Замените на рабочий хост
port = 8000          # Замените на рабочий порт
api_client = ModelAPI(host, port)

# Заголовок приложения
st.title("Модель по анализу данных")

# Инициализация состояния для страницы
if 'page' not in st.session_state:
    st.session_state.page = "🔄 Обучение модели"

# Кнопки для навигации между страницами
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Обучение модели"):
        st.session_state.page = "🔄 Обучение модели"
with col2:
    if st.button("ℹ️ Информация о модели"):
        st.session_state.page = "ℹ️ Информация о модели"

# Страница "Обучение модели"
if st.session_state.page == "🔄 Обучение модели":
    st.header("Обучение модели")
    
    # Типы модели
    type_of_model = st.selectbox("Выберите модель", ["⚖️ Ridge Classifier", "🧠 CatBoost Classifier"])

    # Параметры моделей
    params = {"type_of_model": type_of_model}

    # Блок гиперпараметров
    st.subheader("Гиперпараметры модели")
    if type_of_model == "⚖️ Ridge Classifier":
        params["alpha"] = st.number_input("Alpha", value=1.0, min_value=0.0)
        params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)

    elif type_of_model == "🧠 CatBoost Classifier":
        params["learning_rate"] = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
        params["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
        params["iterations"] = st.number_input("Iterations", value=100, min_value=1)
        params["l2_leaf_reg"] = st.number_input("L2 Leaf Regularization", value=3, min_value=1, max_value=10)

    # ID модели
    params["model_id"] = st.text_input("Введите ID модели", value="model")

    # Загрузка файла
    uploaded_file = st.file_uploader("📤 Загрузите данные (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Данные:")
        st.write(data.head())

        # Определение целевой переменной
        target_column = "radiant_win"
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            st.subheader(f"Целевая переменная: {target_column}")
            st.write(y.value_counts())
        else:
            st.error(f"Целевая переменная '{target_column}' не найдена в данных.")
            st.stop()

        # Преобразуем категориальные признаки
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            # Проверяем наличие NaN и заменяем их на строку 'missing' или самую частую категорию
            X[col] = X[col].fillna('missing')

        # Обработка модели
if st.button("🚀 Обучить модель"):
    start_time = time.time()

    # Проверка на NaN и бесконечные значения
    if data.isnull().values.any() or np.isinf(data.values).any():
        st.error("❌ Данные содержат NaN или бесконечные значения. Пожалуйста, очистите данные.")
        st.stop()

    # Проводим кросс-валидацию локально
    if type_of_model == "⚖️ Ridge Classifier":
        # Преобразуем категориальные данные в числовой формат с использованием one-hot encoding
        X = pd.get_dummies(X, drop_first=True)  # One-hot encoding
        
        # Проверка на особенности в данных
        if X.shape[1] >= len(y):
            st.error("❌ Количество признаков больше или равно количеству наблюдений. Пожалуйста, уменьшите количество признаков.")
            st.stop()

        model = RidgeClassifier(alpha=params["alpha"], fit_intercept=params["fit_intercept"])
    elif type_of_model == "🧠 CatBoost Classifier":
        # CatBoost может работать с категориальными данными
        model = CatBoostClassifier(
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            iterations=params["iterations"],
            l2_leaf_reg=params["l2_leaf_reg"],
            cat_features=categorical_cols,  # Передаём категориальные признаки напрямую
            verbose=False
        )

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

    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)

    end_time = time.time()

    st.success("✅ Модель обучена!")
    st.write(f"⏳ Время обучения составило: {end_time - start_time:.2f} сек")
    st.write("📊 Результаты кросс-валидации:")
    st.write(pd.DataFrame({"Fold": range(1, 6), "Accuracy": fold_results}))
    st.write(f"🏆 Средняя точность: {mean_accuracy:.4f}")
    st.write(f"📉 Стандартное отклонение точности: {std_accuracy:.4f}")

    # Важность признаков для CatBoost
    if type_of_model == "🧠 CatBoost Classifier":
        feature_importances = model.get_feature_importance()
        feature_importances_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)
        st.write("📈 Важность признаков:")
        st.bar_chart(feature_importances_df.set_index("Feature"))
# Страница "Информация о модели"
elif st.session_state.page == "ℹ️ Информация о модели":
    st.header("Информация о модели")
    
    model_id = st.text_input("Введите ID модели для получения информации", value="model")
    if st.button("📖 Получить информацию о модели"):
        model_info = api_client.get_model_info(model_id)
        if model_info:
            st.write("📝 Информация о модели:")
            st.json(model_info)

            # Важность признаков
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
