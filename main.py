import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier

# Остальная часть вашего кода остается без изменений...

# Страница "Обучение модели"
if st.session_state.page == "🔄 Обучение модели":
    st.header("Обучение модели")
    
    # Остальные части кода...

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

        # Проверка уникальных значений в целевой переменной
        st.write("Уникальные значения целевой переменной:")
        st.write(y.unique())
        st.write("Количество уникальных значений:", len(y.unique()))

        # Преобразуем категориальные признаки
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].fillna('missing')

        # Обработка модели
        if st.button("🚀 Обучить модель"):
            start_time = time.time()

            if type_of_model == "⚖️ Ridge Classifier":
                X = pd.get_dummies(X, drop_first=True)
                model = RidgeClassifier(alpha=params["alpha"], fit_intercept=params["fit_intercept"])
            elif type_of_model == "🧠 CatBoost Classifier":
                model = CatBoostClassifier(
                    learning_rate=params["learning_rate"],
                    depth=params["depth"],
                    iterations=params["iterations"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    cat_features=categorical_cols,
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

                # Отладка: выводим результаты каждого фолда
                st.write(f"Fold accuracy: {accuracy:.4f}")

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
