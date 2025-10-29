# agent/ui/training_interface.py (исправлён undefined 'logic' и улучшен UX)
import sys
import streamlit as st
import json
import os
import re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from agent.utils.logger import get_logger

st.sidebar.header("DEBUG")
if "agent" in st.session_state:
    try:
        st.sidebar.write("Agent class:", type(st.session_state.agent).__name__)
        st.sidebar.write("History length:", len(st.session_state.agent.history))
        if st.session_state.agent.history:
            last = st.session_state.agent.history[-1]
            st.sidebar.write("Last message role:", last.role)
            st.sidebar.json(last.metadata or {})
    except Exception as e:
        st.sidebar.write("Debug read error:", e)

logger = get_logger(__name__)

current_file_path = Path(__file__).resolve().parent
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent.utils.knowledge_base import load_knowledge_base, save_knowledge_base_entry

DB_PATH = project_root / "generated" / "digital_twin.db"
KB_PATH = project_root / "generated" / "knowledge_base.json"
(project_root / "generated").mkdir(exist_ok=True)

try:
    from agent.thinking_agent import ThinkingAgent
    THINKING_AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"Не удалось импортировать agent.thinking_agent: {e}")
    THINKING_AGENT_AVAILABLE = False
    class ThinkingAgent:
        def __init__(self, *args, **kwargs):
            pass
        def think(self, question: str):
            return {"type": "final_answer", "content": f"Имитация ответа на вопрос: {question}", "details": {}}

def get_db_schema():
    if not DB_PATH.exists():
        st.warning("Файл базы данных не найден.")
        return {}
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
        inspector = inspect(engine)
        tables = {}
        for table_name in inspector.get_table_names():
            if table_name != "docs":
                columns = [col['name'] for col in inspector.get_columns(table_name)]
                tables[table_name] = columns
        return tables
    except Exception as e:
        st.error(f"Ошибка подключения к БД: {e}")
        return {}

def safe_dataframe_display(df: pd.DataFrame, use_container_width: bool = True, height: int | str | None = None):
    if df.empty:
        st.info("DataFrame пуст.")
        return
    df_safe = df.copy()
    for col in df_safe.columns:
        df_safe[col] = df_safe[col].infer_objects(copy=False).fillna("").astype(str)
    effective_height: int | str = height if height is not None else ('stretch' if use_container_width else 400)
    st.dataframe(df_safe, use_container_width=use_container_width, height=effective_height)

try:
    from agent.cli.query import ask_ru
    QUERY_MODULE_AVAILABLE = True
except ImportError as e:
    st.error(f"Не удалось импортировать agent.cli.query: {e}")
    st.info("Модуль агент-CLI недоступен — функциональность запроса временно ограничена.")
    QUERY_MODULE_AVAILABLE = False


    def ask_ru(question: str):
        """Fallback: возвращаем явную ошибку (не имитацию)."""
        return {
            "type": "error",
            "content": f"Модуль agent.cli.query недоступен: {e}"
        }

st.set_page_config(page_title="Цифровой двойник - Обучение", page_icon="🧠", layout="wide")
st.title("🧠 Цифровой двойник - Интерфейс Обучения")

if "thinking_agent" not in st.session_state:
    st.session_state.thinking_agent = ThinkingAgent(DB_PATH, KB_PATH)
agent = st.session_state.thinking_agent

tab1, tab2, tab3 = st.tabs(["Обзор БД", "Запрос к агенту", "База знаний"])

with tab1:
    st.header("Обзор структуры базы данных")
    schema = get_db_schema()
    if not schema:
        st.info("Нет доступных таблиц.")
    else:
        table_names = list(schema.keys())
        selected_table = st.selectbox("Выберите таблицу:", table_names, key="schema_table_overview")
        if selected_table:
            st.subheader(f"Схема таблицы: `{selected_table}`")
            cols_df = pd.DataFrame(schema[selected_table], columns=["Столбец"])
            safe_dataframe_display(cols_df, use_container_width=True, height=200)
            if st.checkbox("Показать образцы данных", key=f"show_sample_{selected_table}_overview"):
                try:
                    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
                    with engine.connect() as conn:
                        total_rows_result = conn.execute(text(f'SELECT COUNT(*) FROM "{selected_table}"'))
                        total_rows = total_rows_result.scalar()
                        st.write(f"**Общее количество строк в таблице:** {total_rows}")
                        if total_rows:
                            sample_result = conn.execute(text(f'SELECT * FROM "{selected_table}" LIMIT :limit'), {"limit": 10})
                            sample_rows = sample_result.fetchall()
                            sample_col_names = list(sample_result.keys()) if sample_rows else []
                            if sample_rows:
                                sample_df = pd.DataFrame(sample_rows, columns=sample_col_names)
                                st.write(f"**Первые {min(10, total_rows)} строк:**")
                                safe_dataframe_display(sample_df, use_container_width=True, height=400)
                except Exception as e:
                    st.error(f"Ошибка при загрузке образцов данных: {e}")

with tab2:
    st.header("Взаимодействие с агентом и обучение")
    schema = get_db_schema()
    if not schema:
        st.info("Нет доступных таблиц.")
        st.stop()
    table_names = list(schema.keys())
    context_table = st.selectbox("Выберите таблицу для контекста запроса:", table_names, key="context_table_query")
    user_question = st.text_input("Введите ваш вопрос к агенту:", placeholder="Например: Какие есть подобъекты?")
    if st.button("🧠 Отправить запрос агенту") and user_question:
        st.subheader("🧠 Работа агента")
        st.write(f"**Вопрос:** {user_question}")
        try:
            agent_response = ask_ru(user_question) if QUERY_MODULE_AVAILABLE else {"type":"final_answer","content":"Имитация"}
            if agent_response["type"] == "final_answer":
                st.session_state.agent_needs_clarification = False
                st.session_state.clarification_request = ""
                st.write(f"**🧠 Ответ агента:**")
                st.write(agent_response["content"])
                result_data = agent_response.get("result", [])
                if result_data:
                    result_df = pd.DataFrame(result_data)
                    st.write(f"**📊 Результат SQL-запроса (найдено строк: {len(result_df)}):**")
                    safe_dataframe_display(result_df, use_container_width=True, height=400)
                generated_json = agent_response.get("json_query", {})
                if generated_json:
                    st.write("**🧠 Сгенерированный JSON-запрос:**")
                    st.json(generated_json)
                generated_sql = agent_response.get("sql_query", "")
                if generated_sql:
                    st.write("**🔍 Построенный SQL-запрос:**")
                    st.code(generated_sql, language="sql")
            elif agent_response["type"] == "error":
                st.error(f"❌ Ошибка от агента: {agent_response['content']}")
            elif agent_response["type"] == "clarify":
                st.warning(agent_response.get("content"))
                mappings = agent_response.get("mappings", {})
                if mappings.get("unknowns"):
                    with st.expander("⚠️ Требуется уточнение"):
                        for unk in mappings["unknowns"]:
                            st.write(f"- {unk}")
        except Exception as e:
            st.error(f"❌ Ошибка при обращении к агенту: {e}")

    st.subheader("Исправить и сохранить как пример (для обучения)")
    initial_json_str = "{}"
    if 'agent_response' in locals() and isinstance(agent_response, dict) and agent_response.get("json_query"):
        try:
            initial_json_str = json.dumps(agent_response["json_query"], ensure_ascii=False, indent=2)
        except Exception:
            initial_json_str = "{}"
    else:
        initial_json_structure = {"select": ["*"], "from": context_table if 'context_table' in locals() else "table_name", "where": []}
        initial_json_str = json.dumps(initial_json_structure, ensure_ascii=False, indent=2)

    edited_json_str = st.text_area("Отредактируйте JSON, если агент ошибся:", value=initial_json_str, height=250, key="edit_json_query")
    if st.button("💾 Сохранить как пример"):
        try:
            json.loads(edited_json_str)
            save_knowledge_base_entry(user_question, edited_json_str)
        except json.JSONDecodeError:
            st.error("Введённый текст не является корректным JSON.")

    new_logic_str = st.text_area("Хотите изменить логику? (JSON)", value=json.dumps({}, ensure_ascii=False, indent=2), height=200, key="edit_logic")
    if st.button("💾 Сохранить новую логику"):
        try:
            logic_new = json.loads(new_logic_str)
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_path = project_root / "generated" / f"saved_logic_{ts}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(logic_new, ensure_ascii=False, indent=2), encoding='utf-8')
            st.success(f"Логика сохранена: {out_path.relative_to(project_root)}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tab3:
    st.header("База знаний агента")
    kb = load_knowledge_base()
    if not kb:
        st.info("База знаний пуста.")
    else:
        st.write(f"Всего примеров: {len(kb)}")
        num_to_show = st.number_input("Количество последних примеров для отображения:", min_value=1, max_value=len(kb), value=min(10, len(kb)), step=1)
        for i, entry in enumerate(list(reversed(kb))[:num_to_show]):
            with st.expander(f"**Вопрос {len(kb) - i}:** {entry['question']}", expanded=False):
                st.write(f"**Вопрос:** {entry['question']}")
                st.write(f"**JSON-запрос:**")
                st.json(entry['json_query'])
                st.write(f"**Сохранено:** {entry.get('timestamp', 'N/A')} пользователем {entry.get('user', 'N/A')}")