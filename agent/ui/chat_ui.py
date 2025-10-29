# agent/ui/chat_ui.py (обновлённый: уникальные ключи для кнопок предложений, отображение объяснений)
from pathlib import Path
import sys
import json
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text
import uuid

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

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.analytic_agent import AnalyticAgent
from agent.utils.knowledge_base import load_knowledge_base, save_knowledge_base_entry
from agent.utils.fuzzy_search import find_similar_values

DB_PATH = ROOT / "generated" / "digital_twin.db"
KB_PATH = ROOT / "generated" / "knowledge_base.json"

st.set_page_config(page_title="Цифровой двойник", layout="wide")
st.title("📊 Цифровой двойник — аналитик БД")

def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
        else:
            st.session_state["_rerun_requested"] = not st.session_state.get("_rerun_requested", False)
            st.stop()
    except Exception:
        st.session_state["_rerun_requested"] = not st.session_state.get("_rerun_requested", False)
        st.stop()

# Init analytic agent
if "analytic_agent" not in st.session_state:
    try:
        st.session_state.analytic_agent = AnalyticAgent(DB_PATH, KB_PATH)
    except Exception as e:
        st.error(f"Не удалось инициализировать AnalyticAgent: {e}")
        st.stop()
analytic: AnalyticAgent = st.session_state.analytic_agent

if "last_assistant_msg" not in st.session_state:
    st.session_state.last_assistant_msg = None

# Top controls
col_top_left, col_top_right = st.columns([1, 3])
with col_top_left:
    if st.button("🆕 Новый диалог / Сброс контекста"):
        st.session_state.analytic_agent = AnalyticAgent(DB_PATH, KB_PATH)
        st.session_state.last_assistant_msg = None
        safe_rerun()

tabs = st.tabs(["Чат-аналитик", "Обзор БД", "База знаний"])

with tabs[0]:
    st.header("💬 Чат с аналитиком")

    # show last assistant message
    last_msg = st.session_state.get("last_assistant_msg")
    if last_msg:
        try:
            if getattr(last_msg, "role", "") == "assistant":
                st.chat_message("assistant").write(last_msg.content)
                meta = getattr(last_msg, "metadata", {}) or {}
                if meta.get("sql") or meta.get("json_query"):
                    with st.expander("🔍 Детали последнего ответа"):
                        if meta.get("json_query"):
                            st.write("JSON-запрос:")
                            st.json(meta.get("json_query"))
                        if meta.get("sql"):
                            st.write("Выполненный SQL:")
                            st.code(meta.get("sql"), language="sql")
                        if meta.get("rows"):
                            df = pd.DataFrame(meta.get("rows"))
                            st.write(f"Результат (строк {len(df)}):")
                            st.dataframe(df.head(10), use_container_width=True)
        except Exception:
            st.write(str(last_msg))

    # render thinker history
    for msg in analytic.thinker.history:
        if msg.role == "user":
            st.chat_message("user").write(msg.content)
        elif msg.role == "assistant":
            st.chat_message("assistant").write(msg.content)
            meta = msg.metadata or {}
            # raw LLM
            if meta.get("raw_llm"):
                with st.expander("🧾 Raw LLM output", expanded=False):
                    st.code(meta.get("raw_llm"), language="text")
            # mappings / unknowns
            mappings = meta.get("mappings") or {}
            if mappings.get("unknowns"):
                with st.expander("🔧 Автоматические подстановки / неизвестные поля", expanded=True):
                    st.write("Неизвестные поля (нужны уточнения):")
                    for unk in mappings["unknowns"]:
                        st.write(f"- {unk}")
                        # explanation from thinker
                        expl = mappings.get("explanations", {}).get(unk)
                        if expl:
                            st.caption(expl)

                        # If thinker provided computation hint, show it
                        comp_hint = mappings.get("computation_hint")
                        if comp_hint:
                            st.info(
                                f"💡 Я предлагаю вычисление: {comp_hint.get('description')}. Если вы согласны — выберите колонку ниже, и я подставлю вычисление автоматически.")
                            # show suggested column if available
                            guessed = mappings.get("column_guesses", {}).get(unk)
                            if guessed:
                                key_map = f"mapcol_{unk}_{guessed}_{uuid.uuid4().hex[:6]}"
                                if st.button(f"Использовать колонку: {guessed} (и подставить вычисление)", key=key_map):
                                    reply = analytic.apply_clarification(f"{unk}={guessed}")
                                    st.session_state.last_assistant_msg = reply
                                    safe_rerun()

                        # suggestions: values from DB
                        suggestions = mappings.get("suggestions", {}).get(unk, [])
                        if suggestions:
                            st.write("Предложенные значения:")
                            for i, val in enumerate(suggestions):
                                unique_key = f"choose_{unk}_{i}_{uuid.uuid4().hex[:6]}"
                                if st.button(f"Выбрать: {val}", key=unique_key):
                                    reply = analytic.apply_clarification(f"{unk}={val}")
                                    st.session_state.last_assistant_msg = reply
                                    safe_rerun()
                        else:
                            st.write("Предложенных значений нет.")

                        # Manual column selection (show list of columns)
                        try:
                            engine_tmp = create_engine(f"sqlite:///{DB_PATH}", future=True)
                            inspector_tmp = inspect(engine_tmp)
                            first_table = inspector_tmp.get_table_names()[
                                0] if inspector_tmp.get_table_names() else None
                            if first_table:
                                cols = [c["name"] for c in inspector_tmp.get_columns(first_table)]
                                if cols:
                                    st.write("Или выберите колонку вручную:")
                                    for j, colname in enumerate(cols):
                                        key_col = f"choosecol_{unk}_{j}_{uuid.uuid4().hex[:6]}"
                                        if st.button(colname, key=key_col):
                                            reply = analytic.apply_clarification(f"{unk}={colname}")
                                            st.session_state.last_assistant_msg = reply
                                            safe_rerun()
                        except Exception:
                            pass

                        # text input fallback
                        clar_input_key = f"clar_{unk}_{uuid.uuid4().hex[:6]}"
                        clar_submit_key = f"clar_submit_{unk}_{uuid.uuid4().hex[:6]}"
                        clar_input = st.text_input(f"Введите значение или колонку для {unk} (пример: {unk}=Значение)",
                                                   key=clar_input_key)
                        if st.button(f"Уточнить {unk}", key=clar_submit_key):
                            if clar_input:
                                reply = analytic.apply_clarification(clar_input)
                                st.session_state.last_assistant_msg = reply
                                safe_rerun()

            # show SQL/result if present
            if meta.get("sql"):
                with st.expander("🔍 SQL и результат", expanded=False):
                    st.code(meta.get("sql", "--"), language="sql")
                    rows = meta.get("rows", [])
                    if rows:
                        st.dataframe(pd.DataFrame(rows).head(10), use_container_width=True)

    # user input
    if prompt := st.chat_input("Ваш вопрос или уточнение:"):
        st.chat_message("user").write(prompt)
        try:
            reply = analytic.process_user(prompt)
            st.session_state.last_assistant_msg = reply
        except Exception as e:
            st.error(f"Ошибка при запросе к аналитику: {e}")
            import traceback
            st.code(traceback.format_exc(), language="text")
        safe_rerun()

    # save to KB
    if st.button("💾 Сохранить последний запрос в базу знаний"):
        last = st.session_state.get("last_assistant_msg")
        if last and getattr(last, "metadata", None) and last.metadata.get("json_query"):
            save_knowledge_base_entry(last.content, json.dumps(last.metadata["json_query"], ensure_ascii=False))
            st.success("Сохранено!")
        else:
            st.warning("Нет данных для сохранения.")