import streamlit as st
from train_model import train_model
from model_info import model_info

# Установка заголовка приложения
st.title("Модель по анализу данных")

# Выбор страницы
st.header("Навигация")
if st.button("Обучение модели"):
    train_model()
elif st.button("Информация о модели"):
    model_info()