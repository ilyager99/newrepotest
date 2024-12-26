import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier


# Функция для обучения модели
def train_model():
    st.subheader("Обучение модели")

    # ID модели
    model_id = st.text_input("Введите ID модели", value="model")

    # типы модели
    type_of_model = st.selectbox("Выберите модель", ["Ridge Classifier", "CatBoost Classifier"])

    # параметры моделей
    params = {"type_of_model": type_of_model}

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
            start_time = time.time()

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