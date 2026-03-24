"""
AI Financial Assistant — MVP
Загружает финансовые данные компании, рассчитывает ключевые метрики
и отвечает на вопросы пользователя с помощью LLM.
"""

import os

import pandas as pd
import streamlit as st
import anthropic
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ─── Настройки страницы ───
st.set_page_config(page_title="AI Financial Assistant", page_icon="📊")
st.title("📊 AI Financial Assistant")
st.caption("Задавайте вопросы о финансовых показателях компании (2005–2024)")


# ─── Загрузка и обработка данных ───
@st.cache_data
def load_and_process_data(path: str = "financial_data.csv") -> pd.DataFrame:
    """Загружает CSV и рассчитывает ключевые финансовые метрики."""
    # Файл в UTF-16 с BOM — pandas обработает автоматически
    df = pd.read_csv(path, encoding="utf-16")

    # Рост выручки (год к году), %
    df["revenue_growth"] = df["revenue"].pct_change() * 100

    # Операционная маржа = (Выручка - COGS - Операционные расходы) / Выручка
    df["operating_margin"] = (
        (df["revenue"] - df["cogs"] - df["operating_expenses"]) / df["revenue"] * 100
    )

    # Чистая маржа = Чистая прибыль / Выручка
    df["net_margin"] = df["net_income"] / df["revenue"] * 100

    return df


df = load_and_process_data()


def build_data_context(df: pd.DataFrame) -> str:
    """Формирует текстовое представление данных для передачи в LLM."""
    lines = []
    lines.append("ФИНАНСОВЫЕ ДАННЫЕ КОМПАНИИ (2005–2024):")
    lines.append("=" * 60)

    for _, row in df.iterrows():
        year = int(row["year"])
        revenue = int(row["revenue"])
        cogs = int(row["cogs"])
        opex = int(row["operating_expenses"])
        net = int(row["net_income"])
        op_margin = round(row["operating_margin"], 2)
        net_margin = round(row["net_margin"], 2)

        growth_str = (
            f"{row['revenue_growth']:.1f}%" if pd.notna(row["revenue_growth"]) else "N/A"
        )

        lines.append(
            f"Год {year}: Выручка=${revenue:,} | COGS=${cogs:,} | "
            f"Опер. расходы=${opex:,} | Чистая прибыль=${net:,} | "
            f"Рост выручки={growth_str} | Опер. маржа={op_margin}% | "
            f"Чистая маржа={net_margin}%"
        )

    return "\n".join(lines)


SYSTEM_PROMPT = f"""Ты — профессиональный финансовый аналитик.
Твоя задача — отвечать на вопросы о финансовых показателях компании,
используя ТОЛЬКО предоставленные ниже данные.

СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО числа из предоставленных данных. Не придумывай значения.
2. Если вопрос выходит за рамки данных — честно скажи об этом.
3. Объясняй свои расчёты пошагово.
4. Давай структурированные ответы.
5. Отвечай на русском языке.
6. Будь лаконичен: давай точный ответ на вопрос, без избыточных деталей.

{build_data_context(df)}
"""


# ─── LLM-интеграция ───
def get_llm_response(question: str) -> str:
    """Отправляет вопрос пользователя в LLM вместе с финансовыми данными."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ API-ключ не найден. Создайте файл `.env` с переменной `ANTHROPIC_API_KEY`."

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return response.content[0].text


# ─── Интерфейс ───

# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Боковая панель: данные и графики ---
with st.sidebar:
    st.header("📋 Данные компании")

    # Интерактивный график выручки и прибыли
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["revenue"],
        name="Выручка", mode="lines+markers",
        line=dict(color="#2563eb", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["net_income"],
        name="Чистая прибыль", mode="lines+markers",
        line=dict(color="#16a34a", width=2),
    ))
    fig.update_layout(
        title="Выручка и чистая прибыль",
        xaxis_title="Год", yaxis_title="$",
        height=350, margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # График маржинальности
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["year"], y=df["revenue_growth"],
        name="Рост выручки, %",
        marker_color="#f59e0b",
    ))
    fig2.update_layout(
        title="Рост выручки (YoY)",
        xaxis_title="Год", yaxis_title="%",
        height=250, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Таблица (сворачиваемая)
    with st.expander("Таблица с метриками"):
        display_df = df.copy()
        display_df["revenue_growth"] = display_df["revenue_growth"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
        )
        display_df["operating_margin"] = display_df["operating_margin"].apply(
            lambda x: f"{x:.1f}%"
        )
        display_df["net_margin"] = display_df["net_margin"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Кнопки с примерами вопросов ---
EXAMPLE_QUESTIONS = [
    "В каком году был самый быстрый рост выручки?",
    "Как изменялась прибыльность компании со временем?",
    "Объясните динамику операционной маржи",
    "Сравните показатели 2005 и 2024 года",
]

st.markdown("**Примеры вопросов:**")
cols = st.columns(2)
for i, q in enumerate(EXAMPLE_QUESTIONS):
    if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
        st.session_state.pending_question = q

# --- История чата ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Ввод вопроса ---
user_input = st.chat_input("Задайте вопрос о финансовых показателях...")

# Обработка: ввод из чата или из кнопки-примера
question = user_input or st.session_state.pop("pending_question", None)

if question:
    # Показываем вопрос пользователя
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Получаем и показываем ответ
    with st.chat_message("assistant"):
        with st.spinner("Анализирую данные..."):
            answer = get_llm_response(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
