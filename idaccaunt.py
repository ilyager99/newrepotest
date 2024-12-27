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

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Обучение модели"):
        st.session_state.page = "🔄 Обучение модели"
with col2:
    if st.button("ℹ️ Информация о модели"):
        st.session_state.page = "ℹ️ Информация о модели"

if st.session_state.page == "🔄 Обучение модели":
    st.header("Обучение модели")

    type_of_model = st.selectbox("Выберите модель", ["⚖️ Ridge Classifier", "🧠 CatBoost Classifier"])
    params = {"type_of_model": type_of_model}

    st.subheader("Гиперпараметры модели")
    if type_of_model == "⚖️ Ridge Classifier":
        params["alpha"] = st.number_input("Alpha", value=1.0, min_value=0.0)
        params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)

    elif type_of_model == "🧠 CatBoost Classifier":
        params["learning_rate"] = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
        params["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
        params["iterations"] = st.number_input("Iterations", value=100, min_value=1)
        params["l2_leaf_reg"] = st.number_input("L2 Leaf Regularization", value=3, min_value=1, max_value=10)

    params["model_id"] = st.text_input("Введите ID модели", value="model")
    uploaded_file = st.file_uploader("📤 Загрузите данные (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Данные:")
        st.write(data.head())

        target_column = "radiant_win"
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Обработка категориальных переменных
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            if type_of_model == "🧠 CatBoost Classifier":
                cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]  # Индексы категориальных признаков

            else:
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))  # Преобразуем категориальные признаки в числовые

            st.subheader(f"Целевая переменная: {target_column}")
            st.write(y.value_counts())

            # Если в данных есть столбец 'account_id', добавляем выбор
            if 'account_id' in data.columns:
                account_ids = data['account_id'].unique()
                selected_account_id = st.selectbox("Выберите Account ID", account_ids)

                # Фильтрация данных по выбранному account_id
                account_data = data[data['account_id'] == selected_account_id]

            else:
                st.warning("В данных отсутствует столбец 'account_id'.")
                account_data = None

        else:
            st.error(f"Целевая переменная '{target_column}' не найдена в данных.")
            st.stop()

        if st.button("🚀 Обучить модель"):
            params["train_data"] = data.to_dict(orient="list")
            start_time = time.time()

            if type_of_model == "⚖️ Ridge Classifier":
                model = RidgeClassifier(alpha=params["alpha"], fit_intercept=params["fit_intercept"])
            elif type_of_model == "🧠 CatBoost Classifier":
                model = CatBoostClassifier(
                    learning_rate=params["learning_rate"],
                    depth=params["depth"],
                    iterations=params["iterations"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    cat_features=cat_features_indices,  # Передаем индексы категориальных признаков
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

            mean_accuracy = np.mean(fold_results)
            std_accuracy = np.std(fold_results)

            end_time = time.time()

            st.success("✅ Модель обучена!")
            st.write(f"⏳ Время обучения составило: {end_time - start_time:.2f} сек")
            st.write("📊 Результаты кросс-валидации:")
            st.write(pd.DataFrame({"Fold": range(1, 6), "Accuracy": fold_results}))
            st.write(f"🏆 Средняя точность: {mean_accuracy:.4f}")
            st.write(f"📉 Стандартное отклонение точности: {std_accuracy:.4f}")

            if type_of_model == "🧠 CatBoost Classifier":
                feature_importances = model.get_feature_importance()
                feature_importances_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": feature_importances
                }).sort_values(by="Importance", ascending=False)
                st.write("📈 Важность признаков:")
                st.bar_chart(feature_importances_df.set_index("Feature"))

            # Новая функциональность для предсказания вероятности победы
            if account_data is not None and len(account_data) > 0:
                account_features = account_data.drop(columns=[target_column])  # Убираем целевую переменную
                if type_of_model == "⚖️ Ridge Classifier":
                    probability = model.decision_function(account_features)
                else:  # Для CatBoost используем predict_proba
                    probability = model.predict_proba(account_features)[:, 1]
                    
                st.write(f"Вероятность победы для Account ID {selected_account_id}: {probability[0]:.2f}")

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
