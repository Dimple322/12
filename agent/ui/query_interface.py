# agent/ui/query_interface.py (обновлён для нового build_sql_from_json и отображения mappings)
import sys
import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
import ollama
import re
import matplotlib.pyplot as plt

current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent.llm.ollama_client import AnalystLLM
from agent.utils.sql_builder import build_sql_from_json
from agent.utils.knowledge_base import load_knowledge_base, save_knowledge_base_entry
from agent.utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = project_root / "generated" / "digital_twin.db"
KB_PATH = project_root / "generated" / "knowledge_base.json"

def get_db_schema():
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    inspector = inspect(engine)
    tables = {}
    for table_name in inspector.get_table_names():
        if table_name != "docs":
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            tables[table_name] = columns
    return tables

def show_db_overview():
    st.subheader("Обзор базы данных")
    schema = get_db_schema()
    if not schema:
        st.error("Не найдено таблиц в базе данных.")
        return
    st.subheader("Схематичный вид")
    fig, ax = plt.subplots(figsize=(10, len(schema) * 0.5 + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(schema))
    ax.axis('off')
    y_pos = len(schema) - 0.5
    for table_name, columns in schema.items():
        ax.text(0.5, y_pos, f"Таблица: {table_name}", fontsize=12, fontweight='bold', verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        col_y_start = y_pos - 0.3
        for i, col in enumerate(columns):
            ax.text(2.5, col_y_start - i*0.2, f" - {col}", fontsize=10, verticalalignment='top')
        y_pos -= max(len(columns) * 0.2, 1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.subheader("Табличный вид")
    selected_table_for_view = st.selectbox("Выберите таблицу для просмотра данных:", list(schema.keys()), key="view_data_table")
    if selected_table_for_view:
        engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
        with engine.begin() as conn:
            limit = st.slider("Количество строк для отображения", min_value=1, max_value=1000, value=100, step=10, key="view_data_limit")
            query = text(f"SELECT * FROM \"{selected_table_for_view}\" LIMIT :limit")
            rows = conn.execute(query, {"limit": limit}).fetchall()
            if rows:
                df = pd.DataFrame([dict(row._mapping) for row in rows])
                st.dataframe(df, use_container_width=True)
            else:
                st.info(f"Таблица '{selected_table_for_view}' пуста.")

st.set_page_config(page_title="Цифровой двойник - Агент", page_icon="🤖", layout="wide")
st.title("🤖 Цифровой двойник - Агент-аналитик")

page = st.sidebar.selectbox("Выберите страницу:", ["Агент-аналитик", "Обзор базы данных"])

if page == "Агент-аналитик":
    schema = get_db_schema()
    if not schema:
        st.error("Не найдено таблиц в базе данных.")
        st.stop()

    table_names = list(schema.keys())
    selected_table = st.selectbox("Выберите таблицу для просмотра схемы (для генерации запроса):", table_names, key="schema_table")

    if selected_table:
        st.subheader(f"Схема таблицы: `{selected_table}`")
        cols_df = pd.DataFrame(schema[selected_table], columns=["Столбец"])
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.form(key='query_form'):
        user_question = st.text_input("Введите ваш вопрос:")
        submit_button = st.form_submit_button(label='Запросить')

    if submit_button and user_question:
        kb = load_knowledge_base()
        relevant_examples = []
        for entry in kb:
            if user_question.lower() in entry.get("question", "").lower() or entry.get("question", "").lower() in user_question.lower():
                relevant_examples.append(entry)

        schema_desc = f"Таблица: {selected_table}, колонки: {', '.join(schema[selected_table])}."
        examples_text = "\n".join([f"Вопрос: \"{entry['question']}\"\nОтвет: {entry['json_query']}" for entry in relevant_examples[:3]])
        prompt = f"""{examples_text}
        Схема данных: {schema_desc}.
        Ты - парсер вопросов пользователя. Твоя задача - проанализировать вопрос и вернуть JSON-описание SQL-запроса.
        Вопрос: '{user_question}'
        Формат JSON: {{"select": ["столбец1", "столбец2", ...], "distinct": true/false (опционально), "from": "имя_таблицы", "where": [{{"column": "имя_столбца", "operator": "оператор", "value": "значение"}}] (опционально)}}, "limit": N (опционально)}}.
        Если не можешь однозначно сопоставить — верни where.column как UNKNOWN_<name>.
        """

        messages = [
            {"role": "system", "content": "Ты - парсер вопросов пользователя. Всегда возвращай только JSON-описанием SQL-запроса."},
            {"role": "user", "content": prompt}
        ]

        try:
            LLM_MODEL = "digital_twin_analyst"
            resp_ollama = ollama.chat(model=LLM_MODEL, messages=messages)
            raw_response = resp_ollama["message"]["content"].strip()
            logger.debug("LLM raw: %s", raw_response)
            json_match = re.search(r'```json\s*(.*?)\s*```|^(.+)$', raw_response, re.DOTALL)
            if json_match:
                extracted_json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                if extracted_json_str:
                    raw_response = extracted_json_str.strip()
            try:
                generated_json_query = json.loads(raw_response)
            except json.JSONDecodeError as e:
                st.error(f"Ошибка парсинга JSON из ответа LLM: {e}")
                with st.expander("Raw LLM response"):
                    st.code(raw_response, language="text")
                generated_json_query = None
            st.subheader("Сгенерированный JSON:")
            st.json(generated_json_query)

            try:
                final_sql, params, mappings = build_sql_from_json(generated_json_query, DB_PATH)
                if mappings.get("unknowns"):
                    with st.expander("⚠️ Требуется уточнение (unknowns)"):
                        st.write("Поля, требующие уточнения:")
                        for unk in mappings["unknowns"]:
                            st.write(f"- {unk}")
                if mappings.get("replacements"):
                    with st.expander("ℹ️ Автоматические замены"):
                        for orig, mapped in mappings["replacements"].items():
                            st.write(f"`{orig}` → `{mapped}`")
                if final_sql:
                    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
                    with engine.begin() as conn:
                        rows = conn.execute(text(final_sql), params).fetchall()
                        data = [dict(r._mapping) for r in rows]
                        if data:
                            df = pd.DataFrame(data)
                            st.subheader("Результаты запроса:")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("Запрос не вернул результатов.")
                    st.subheader("Выполненный SQL:")
                    st.code(final_sql, language="sql")
                    st.write(f"Параметры: {params}")
                else:
                    st.warning("SQL не был сгенерирован из-за неизвестных полей.")
            except Exception as e_sql:
                st.error(f"Ошибка выполнения SQL: {e_sql}")
                st.code(str(e_sql), language="text")

        except json.JSONDecodeError as e:
            st.error(f"Ошибка парсинга JSON из ответа LLM: {e}")
            st.code(raw_response, language="text")
        except Exception as e_gen:
            st.error(f"Ошибка генерации или выполнения запроса: {e_gen}")
            st.code(str(e_gen), language="text")

    st.subheader("Исправить JSON (если результат неверен):")
    corrected_json_str = st.text_area("Введите правильный JSON:", value=json.dumps(generated_json_query, ensure_ascii=False, indent=2) if 'generated_json_query' in locals() else "", height=300)
    if st.button("Сохранить исправление"):
        try:
            corrected_json_obj = json.loads(corrected_json_str)
            save_knowledge_base_entry(user_question, corrected_json_str)
            try:
                final_sql, params, mappings = build_sql_from_json(corrected_json_obj, DB_PATH)
                if final_sql:
                    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
                    with engine.begin() as conn:
                        rows = conn.execute(text(final_sql), params).fetchall()
                        data = [dict(r._mapping) for r in rows]
                    st.success("Исправление сохранено и протестировано успешно!")
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Исправление сохранено, но SQL не сгенерирован (возможно, есть UNKNOWN-поля).")
            except Exception as e_test:
                st.warning(f"Исправление сохранено, но ошибка при тестовом выполнении: {e_test}")
        except json.JSONDecodeError:
            st.error("Введённый текст не является корректным JSON.")

elif page == "Обзор базы данных":
    show_db_overview()

if __name__ == "__main__":
    st.write("Запустите интерфейс: streamlit run agent/ui/query_interface.py")