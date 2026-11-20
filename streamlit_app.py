"""Вебінтерфейс на Streamlit. Запуск: streamlit run streamlit_app.py"""

import os
import re

import streamlit as st

from rewriter import MAX_CHARS, MODEL, STYLES, RewriteError, rewrite

EXAMPLES = [
    ("Записка викладачу",
     "Привіт! Слухай, я завтра не прийду на пару, бо щось захворів. Скинеш потім, що там задавали?",
     "Офіційно-діловий"),
    ("Пункт положення",
     "Відповідно до пункту 3.2 Положення, здобувачі освіти, які не з'явилися на підсумковий контроль "
     "з поважних причин, мають право на його перескладання у встановлені деканатом терміни.",
     "Простими словами"),
    ("Як вчиться нейромережа",
     "Нейронна мережа вчиться так: робить передбачення, дивиться, наскільки помилилася, "
     "і трохи змінює свої ваги, щоб наступного разу помилитися менше.",
     "Науковий"),
    ("English: meeting",
     "The meeting has been postponed due to unforeseen circumstances. "
     "Please confirm your availability for next Tuesday.",
     "Розмовний"),
]


def as_plain_text(text: str) -> str:
    """Екранує markdown, щоб відповідь показувалась як звичайний текст, зі збереженням рядків."""
    escaped = re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$])", r"\\\1", text)
    return escaped.replace("\n", "  \n")


def use_example(text: str, style: str) -> None:
    st.session_state.text = text
    st.session_state.style = style
    st.session_state.pop("answer", None)


st.set_page_config(page_title="Переписування тексту в іншому стилі", page_icon="✍️")

st.title("Переписування тексту в іншому стилі")
st.markdown(
    f"Вставте текст, оберіть стиль і натисніть «Переписати». Сервіс робить один запит до мовної моделі "
    f"`{MODEL}` через Groq API і показує лише переписаний текст. Зміст, факти, імена й числа "
    f"зберігаються, мова відповіді збігається з мовою оригіналу.\n\n"
    f"Потрібен API-ключ Groq (безкоштовно на [console.groq.com](https://console.groq.com/keys)). "
    f"Ключ використовується лише для вашого запиту і ніде не зберігається."
)

api_key = st.text_input("API-ключ Groq", type="password", placeholder="gsk_...")

st.caption("Приклади")
for column, (title, text, style) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    column.button(title, on_click=use_example, args=(text, style), width="stretch")

text = st.text_area("Текст", key="text", height=180, max_chars=MAX_CHARS,
                    placeholder="Вставте текст, який треба переписати")
style = st.radio("Стиль", list(STYLES), key="style", horizontal=True)

if st.button("Переписати", type="primary", width="stretch"):
    try:
        with st.spinner("Переписую..."):
            # GROQ_API_KEY із середовища лише для локальної перевірки; на сервері його немає
            st.session_state.answer = rewrite(text, style, api_key or os.environ.get("GROQ_API_KEY", ""))
    except RewriteError as e:
        st.session_state.pop("answer", None)
        st.error(str(e))

if "answer" in st.session_state:
    st.markdown("**Результат**")
    with st.container(border=True, key="result"):
        st.markdown(as_plain_text(st.session_state.answer))
