#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс для продвинутой системы цифрового аналитика с Reasoning.
(Обновлён: показываем технические детали и доказательства безопасно, без падений)
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys
from sqlalchemy import create_engine, inspect, text
import re

from agent.utils.config import DB_PATH as CFG_DB_PATH
DB_PATH = CFG_DB_PATH

# Добавляем путь к модулю
sys.path.append(str(Path(__file__).parent))

from advanced_digital_twin_chroma import AdvancedDigitalTwin, QueryType, AnalysisResult

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Продвинутый цифровой аналитик",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния
if 'digital_twin' not in st.session_state:
    st.session_state.digital_twin = AdvancedDigitalTwin()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_session' not in st.session_state:
    st.session_state.current_session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# CSS стили для улучшенного интерфейса
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 2rem;
    }
    .reasoning-step {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .insight-card {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .recommendation-card {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .scenario-card {
        background-color: #e0e7ff;
        border: 1px solid #6366f1;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .confidence-high { color: #059669; }
    .confidence-medium { color: #d97706; }
    .confidence-low { color: #dc2626; }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<h1 class="main-header">🧠 Продвинутый цифровой аналитик</h1>', unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")

    backend_choice = st.selectbox(
        "Принудительный backend (auto = автоопределение):",
        options=["auto", "sql", "chroma", "hybrid", "schema"],
        index=0,
        help="Выберите, куда направлять запрос: auto/ sql / chroma / hybrid / schema"
    )
    st.session_state.backend_choice = backend_choice

    # Выбор типа запроса
    query_type = st.selectbox(
        "Тип запроса",
        options=[
            "Аналитика",
            "Сценарий 'что если'",
            "Предиктивная аналитика",
            "Валидация",
            "Объяснение"
        ],
        help="Выберите тип анализа для оптимизации результата"
    )

    # Сессия
    st.session_state.current_session = st.text_input(
        "ID сессии",
        value=st.session_state.current_session,
        help="Используйте одинаковый ID для сохранения контекста между запросами"
    )

    # Очистка истории
    if st.button("🗑️ Очистить историю", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

    # Экспорт результатов
    if st.button("📥 Экспорт JSON", type="secondary"):
        if st.session_state.chat_history:
            export_data = {
                "session_id": st.session_state.current_session,
                "timestamp": datetime.now().isoformat(),
                "queries": st.session_state.chat_history
            }
            st.download_button(
                label="💾 Скачать результаты",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"analytics_results_{st.session_state.current_session}.json",
                mime="application/json"
            )

def wizard_sql():
    import pandas as pd
    import sqlalchemy
    st.header('🔍 Мастер-пошаговый SQL Wizard')
    # state
    if 'wizard' not in st.session_state or st.session_state.get('reset_wizard'):
        st.session_state['wizard'] = {'step': 0, 'table': None, 'table_confirmed': False, 'reasonings': {}, 'column': None, 'column_confirmed': False, 'agg': None, 'agg_confirmed': False}
        st.session_state['reset_wizard'] = False

    wizard = st.session_state['wizard']
    step = wizard.get('step', 0)

    # 1. Выбор таблицы
    with st.expander('🗂️ Шаг 1: Выберите таблицу для анализа', expanded=(step == 0)):
        tables = []
        try:
            engine = sqlalchemy.create_engine(f'sqlite:///{DB_PATH}')
            inspector = sqlalchemy.inspect(engine)
            tables = inspector.get_table_names()
        except Exception as e:
            st.error(f'Ошибка получения списка таблиц: {e}')
        tables_rus = tables if tables else []
        choose_table = st.selectbox('Таблица:', tables_rus, index=0 if tables_rus else None, key='wizard_table_select')
        # reasoning по выбору таблицы
        st.info(f'LLM reasoning: Для данного анализа рекомендуется работать с таблицей "{choose_table}" — это ваша основная структура для аналитики.')
        preview = None
        if choose_table:
            try:
                with engine.begin() as conn:
                    preview = pd.read_sql(f'SELECT * FROM "{choose_table}" LIMIT 10', conn)
                st.write('Top 10 строк таблицы:')
                st.dataframe(preview)
            except Exception as e:
                st.warning(f'Ошибка предпросмотра: {e}')
        if st.button('Подтвердить выбор таблицы'):
            wizard['table'] = choose_table
            wizard['table_confirmed'] = True
            wizard['step'] = 1
            st.success(f'Шаг 1 подтвержден: выбрана таблица {choose_table}')
            st.rerun()

    # 2. Планируемый placeholder под выбор колонки/агрегации (будет реализовано в следующих батчах)
    with st.expander('📊 Шаг 2: Выберите колонку (будет после подтверждения таблицы)', expanded=(step == 1)):
        if not wizard.get('table_confirmed'):
            st.info('Сначала подтвердите таблицу.')
        else:
            st.info('В следующих шагах появится выбор колонок и предпросмотр значений.')
    # TODO: шаги 3..N — аналогично

    # reset wizard
    if st.button('❌ Сбросить мастер (начать заново)'):
        st.session_state['reset_wizard'] = True
        st.rerun()

# Основная область
main_col1, main_col2 = st.columns([2, 1])

if st.session_state.backend_choice == 'wizard_sql':
    with main_col1:
        wizard_sql()
else:
    # старый UX
with main_col1:
    # Область ввода запроса
    st.header("💬 Аналитический запрос")

    query_input = st.text_area(
        "Введите ваш аналитический запрос:",
        placeholder="Например: Сколько длился вид работ 'X' для каждого уникального объекта?",
        height=100,
        help="Опишите ваш аналитический запрос как можно подробнее"
    )

    # Кнопка отправки
        submit_button = st.button("🚀 Анализировать", type="primary")

# Обработка запроса
if submit_button and query_input:
    with st.spinner("🔄 Выполняется продвинутый анализ..."):
        try:
            # Определение типа запроса
            query_type_mapping = {
                "Аналитика": QueryType.ANALYTICS,
                "Сценарий 'что если'": QueryType.SCENARIO,
                "Предиктивная аналитика": QueryType.PREDICTION,
                "Валидация": QueryType.VALIDATION,
                "Объяснение": QueryType.EXPLANATION
            }

            # Выполнение анализа
            if backend_choice != "auto":
                q_for_agent = f"{backend_choice}: {query_input}"
            else:
                q_for_agent = query_input
            forced = st.session_state.get("backend_choice", "auto")
            q_for_agent = query_input
            if forced and forced != "auto":
                # используем префикс, ReasoningAgent._decide_route обрабатывает "schema:" и др.
                q_for_agent = f"{forced}: {query_input}"

            result = asyncio.run(st.session_state.digital_twin.process_query(
                query=q_for_agent,
                session_id=st.session_state.current_session,
                query_type=query_type_mapping[query_type]
            ))

            # Сохранение в историю (храним объект как есть — UI поддерживает и dataclass и dict)
            st.session_state.chat_history.append({
                "query": query_input,
                "type": query_type,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            st.error(f"❌ Ошибка при обработке запроса: {str(e)}")

# Отображение результатов
if st.session_state.chat_history:
    latest_result = st.session_state.chat_history[-1]["result"]


    # ---- PATCH для advanced_interface.py ----
    # support both dataclass and dict-like result
    def _get(obj, name):
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)


    answer_text = _get(latest_result, "answer")
    if not answer_text:
        # try to derive simple answer: single-row single-col in data
        data_preview = _get(latest_result, "data") or []
        if data_preview and isinstance(data_preview, list) and len(data_preview) == 1 and isinstance(data_preview[0],
                                                                                                     dict) and len(
                data_preview[0]) == 1:
            answer_text = str(list(data_preview[0].values())[0])

    if answer_text:
        st.subheader("✅ Краткий ответ")
        st.info(answer_text)
    else:
        st.subheader("✅ Краткий ответ")
        st.info("Краткий ответ не сгенерирован — см. данные и ход рассуждений.")


    # ---- end patch ----


    # helper getters: support both dataclass/object and dict
    def _get_field(obj, *names):
        if isinstance(obj, dict):
            for n in names:
                v = obj.get(n)
                if v is not None:
                    return v
            return None
        else:
            for n in names:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
            return None


    # Вкладки для отображения различных аспектов результата
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Результаты", "🧠 Ход рассуждений", "💡 Инсайты", "🎯 Сценарии", "📈 Визуализация", "🗄️ DB"
    ])

    with tab1:
        st.header("Результаты анализа")

        # Метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            data_len = len(_get_field(latest_result, "data") or [])
            st.metric("Записей найдено", data_len)

        with col2:
            confidence_score = _get_field(latest_result, "confidence_score") or 0.0
            confidence_color = "high" if confidence_score > 0.8 else "medium" if confidence_score > 0.6 else "low"
            st.metric("Уверенность", f"{confidence_score:.2f}")

        with col3:
            validation = _get_field(latest_result, "validation_results") or {}
            validation_status = "✅ Пройдена" if validation.get("is_valid") else "❌ Ошибки"
            st.metric("Валидация", validation_status)

        with col4:
            steps_len = len(_get_field(latest_result, "reasoning_steps") or [])
            st.metric("Шагов рассуждений", steps_len)

        # Данные
        data_obj = _get_field(latest_result, "data") or []
        if data_obj:
            st.subheader("📋 Данные")
            try:
                df = pd.DataFrame(data_obj)
                try:
                    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
                    if obj_cols:
                        df[obj_cols] = df[obj_cols].astype('string')
                except Exception:
                    pass
                st.dataframe(df, width='stretch')
            except Exception:
                # safety: convert to strings
                df = pd.DataFrame(
                    [{k: str(v) for k, v in (r.items() if isinstance(r, dict) else [])} for r in data_obj])
                st.dataframe(df, width='stretch')
        else:
            st.info("Данные отсутствуют.")

        # Блок уточнений (если нужны)
        validation = _get_field(latest_result, "validation_results") or {}
        if validation.get("needs_clarification"):
            st.warning("Требуются уточнения для завершения запроса.")
            unknowns = validation.get("unknowns", [])
            suggestions = validation.get("suggestions", {})
            guesses = validation.get("guesses", {})
            with st.expander("Показать детали неизвестных полей", expanded=True):
                st.write("Неизвестные элементы:", ", ".join(unknowns))
                if guesses:
                    st.write("Предполагаемые колонки:")
                    st.json(guesses)
                if suggestions:
                    st.write("Подсказки значений:")
                    st.json(suggestions)
            clar_text = st.text_input("Уточнение (пример: budget=cost или просто число)")
            if st.button("💾 Сохранить уточнение и выполнить снова", type="primary"):
                try:
                    res2 = asyncio.run(st.session_state.digital_twin.clarify(st.session_state.current_session, clar_text))
                    st.session_state.chat_history.append({
                        "query": f"[clarify] {clar_text}",
                        "type": "clarification",
                        "result": res2,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.success("Уточнение применено.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка применения уточнения: {e}")

        # --- Технические детали и доказательства (находится сразу после данных) ---
        with st.expander("🔍 Технические детали (SQL / Chroma / Evidence)", expanded=False):
            # SQL (support different field names)
            sql_text = _get_field(latest_result, "sql", "sql_query")
            chroma_q = _get_field(latest_result, "chroma_query", "generated_chroma_query")
            evidence = _get_field(latest_result, "evidence") or _get_field(latest_result, "evidence", "evidences") or []

            if sql_text:
                st.subheader("SQL-запрос")
                # show SQL with safe fallback
                try:
                    st.code(sql_text, language="sql")
                except Exception:
                    st.text(str(sql_text))

            if chroma_q:
                st.subheader("Chroma-запрос")
                try:
                    st.code(chroma_q, language="text")
                except Exception:
                    st.text(str(chroma_q))

            # If the result object contains a "mappings" or "raw_llm_traces", show them too
            mappings = _get_field(latest_result, "mappings") or {}
            raw_traces = _get_field(latest_result, "raw_llm_traces") or {}
            if mappings:
                with st.expander("🔧 Mappings / Unknowns (if any)", expanded=False):
                    st.json(mappings)
            if raw_traces:
                with st.expander("🧾 Raw LLM traces (planner / executor / explainer)", expanded=False):
                    for k, v in raw_traces.items():
                        if v:
                            st.markdown(f"**{k}**")
                            st.code(v, language="text")

            # Evidence: deterministic counts / supporting rows
            if evidence:
                st.subheader("Доказательства (Top‑N)")
                try:
                    # evidence expected: list of dicts with keys 'value' and 'count' (adapt if structure differs)
                    ev_rows = []
                    for ev in evidence:
                        # ev may have different shapes: {value:..., count:...} or {'row':..., 'count':...}
                        val = ev.get("value") if isinstance(ev, dict) else ev
                        cnt = ev.get("count") if isinstance(ev, dict) else None
                        # stringify value if it's dict
                        if isinstance(val, dict):
                            val_repr = ", ".join(f"{k}:{v}" for k, v in val.items())
                        else:
                            val_repr = str(val)
                        ev_rows.append({"value": val_repr, "count": int(cnt) if cnt is not None else None})
                    ev_df = pd.DataFrame(ev_rows)
                    st.dataframe(ev_df, use_container_width=True)
                    # show bar chart if counts available
                    if "count" in ev_df.columns and ev_df["count"].notnull().any():
                        fig = px.bar(ev_df, x="value", y="count", title="Top‑N counts")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write("Ошибка при отображении доказательств:", e)
            else:
                st.write("Нет детерминированных доказательств (агрегированных подсчётов).")

    with tab2:
        st.header("Ход рассуждений")

        steps = _get_field(latest_result, "reasoning_steps") or []
        for step in steps:
            # step can be dict or object
            desc = step.get("description") if isinstance(step, dict) else getattr(step, "description", "")
            step_number = step.get("step_number") if isinstance(step, dict) else getattr(step, "step_number", "")
            reasoning = step.get("reasoning") if isinstance(step, dict) else getattr(step, "reasoning", "")
            action = step.get("action") if isinstance(step, dict) else getattr(step, "action", "")
            expected = step.get("expected_outcome") if isinstance(step, dict) else getattr(step, "expected_outcome", "")
            actual = step.get("actual_outcome") if isinstance(step, dict) else getattr(step, "actual_outcome", None)
            confidence = step.get("confidence") if isinstance(step, dict) else getattr(step, "confidence", 0.0)
            validation_passed = step.get("validation_passed") if isinstance(step, dict) else getattr(step,
                                                                                                     "validation_passed",
                                                                                                     False)

            with st.container():
                st.markdown(f'<div class="reasoning-step">', unsafe_allow_html=True)
                st.markdown(f"**Шаг {step_number}: {desc}**")
                st.write(f"🤔 Обоснование: {reasoning}")
                st.write(f"⚡ Действие: {action}")
                st.write(f"🎯 Ожидаемый результат: {expected}")

                if actual:
                    st.write(f"✅ Фактический результат: {actual}")

                confidence_color = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
                st.markdown(f'<span class="confidence-{confidence_color}">Уверенность: {confidence:.2f}</span>',
                            unsafe_allow_html=True)

                if validation_passed:
                    st.success("✅ Валидация пройдена")

                st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.header("Инсайты и рекомендации")

        insights = _get_field(latest_result, "insights") or []
        recommendations = _get_field(latest_result, "recommendations") or []
        if insights:
            st.subheader("💡 Инсайты")
            for insight in insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        else:
            st.info("Инсайтов нет.")

        if recommendations:
            st.subheader("🎯 Рекомендации")
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)

    with tab4:
        st.header("Сценарии 'что если'")

        sa = _get_field(latest_result, "scenario_analysis")
        if sa:
            # Базовый сценарий
            if "baseline" in sa:
                baseline = sa["baseline"]
                st.subheader("📊 Базовый сценарий")
                st.write(f"**Описание:** {baseline.get('description', '')}")

                if baseline.get("metrics"):
                    st.write("**Метрики:**")
                    for metric, values in baseline["metrics"].items():
                        if isinstance(values, dict):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"{metric} (среднее)", f"{values.get('mean', 0):.2f}")
                            with col2:
                                st.metric(f"{metric} (медиана)", f"{values.get('median', 0):.2f}")
                            with col3:
                                st.metric(f"{metric} (мин)", f"{values.get('min', 0):.2f}")
                            with col4:
                                st.metric(f"{metric} (макс)", f"{values.get('max', 0):.2f}")

            # Альтернативные сценарии
            if "scenarios" in sa:
                st.subheader("🔄 Альтернативные сценарии")

                for scenario in sa.get("scenarios", []):
                    with st.expander(f"**{scenario.get('name', '')}**"):
                        st.write(f"**Описание:** {scenario.get('description', '')}")
                        if "parameters" in scenario:
                            st.write("**Параметры:**")
                            for param, value in scenario["parameters"].items():
                                st.write(f"• {param}: {value}")
                        if "expected_outcomes" in scenario:
                            st.write("**Ожидаемые результаты:**")
                            for outcome, description in scenario["expected_outcomes"].items():
                                st.write(f"• {outcome}: {description}")
                        if "risks" in scenario:
                            st.write("**Риски:**")
                            for risk in scenario["risks"]:
                                st.write(f"⚠️ {risk}")
                        confidence = scenario.get("confidence", 0.5)
                        st.progress(confidence)
                        st.write(f"Уровень уверенности: {confidence:.0%}")
        else:
            st.info("Для этого запроса сценарии не были сгенерированы")

    with tab5:
        st.header("Визуализация данных")

        if data_obj:
            try:
                df = pd.DataFrame(data_obj)
            except Exception:
                df = pd.DataFrame(
                    [{k: str(v) for k, v in (r.items() if isinstance(r, dict) else [])} for r in data_obj])

            # Автоматический выбор типа визуализации
            numeric_columns = df.select_dtypes(include=['number']).columns
            categorical_columns = df.select_dtypes(include=['object']).columns

            if len(numeric_columns) >= 2:
                # Scatter plot для числовых данных
                fig = px.scatter(
                    df,
                    x=numeric_columns[0],
                    y=numeric_columns[1],
                    title=f"Соотношение {numeric_columns[0]} и {numeric_columns[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)

            if len(categorical_columns) > 0:
                # Bar chart для категориальных данных
                for cat_col in categorical_columns[:2]:
                    value_counts = df[cat_col].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Распределение по {cat_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Гистограмма для числовых данных
            if len(numeric_columns) > 0:
                for num_col in numeric_columns[:2]:
                    fig = px.histogram(
                        df,
                        x=num_col,
                        title=f"Распределение {num_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для визуализации.")

    with tab6:
        st.header("🗄️ Обзор и загрузка базы данных (SQLite)")

        st.markdown(
            "Здесь вы можете просмотреть текущую SQLite базу (generated/digital_twin.db), "
            "посмотреть таблицы/схему и загрузить CSV (overwrite или append)."
        )

        # Проверяем наличие файла БД
        if not DB_PATH.exists():
            st.warning(f"Файл БД не найден: {DB_PATH}. Пока нет таблиц для просмотра.")
        else:
            # engine / inspector
            try:
                engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
                inspector = inspect(engine)
                tables = inspector.get_table_names()
            except Exception as e:
                st.error(f"Ошибка подключения к базе: {e}")
                tables = []

            st.subheader("Список таблиц")
            if tables:
                selected_table = st.selectbox("Выберите таблицу для просмотра:", ["-- выбрать --"] + tables, index=0)
                cols_viewer, rows_viewer = st.columns([1, 2])

                with cols_viewer:
                    if selected_table and selected_table != "-- выбрать --":
                        st.write("Схема таблицы:")
                        try:
                            cols = inspector.get_columns(selected_table)
                            schema_df = pd.DataFrame(
                                [{"name": c["name"], "type": str(c.get("type", ""))} for c in cols])
                            st.dataframe(schema_df, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.write("Ошибка чтения схемы:", e)

                        # quick actions
                        with st.expander("Дополнительно: Показать PRAGMA table_info", expanded=False):
                            try:
                                with engine.begin() as conn:
                                    rows = conn.execute(text(f'PRAGMA table_info("{selected_table}")')).fetchall()
                                    pr_df = pd.DataFrame([dict(r._mapping) for r in rows])
                                    st.dataframe(pr_df, use_container_width=True)
                            except Exception as e:
                                st.write("Ошибка PRAGMA:", e)

                with rows_viewer:
                    if selected_table and selected_table != "-- выбрать --":
                        limit = st.number_input("Показать строк (limit)", min_value=1, max_value=10000, value=100,
                                                step=10)
                        try:
                            with engine.begin() as conn:
                                q = text(f'SELECT * FROM "{selected_table}" LIMIT :limit')
                                rows = conn.execute(q, {"limit": limit}).fetchall()
                                if rows:
                                    df = pd.DataFrame([dict(r._mapping) for r in rows])
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.info("Таблица пуста.")
                        except Exception as e:
                            st.write("Ошибка при чтении данных:", e)

            else:
                st.info("Таблиц не найдено в базе данных.")

            st.markdown("---")
            st.subheader("Загрузить CSV в базу данных")

            with st.form("csv_loader_form"):
                uploaded = st.file_uploader("Выберите CSV файл (UTF-8)", type=["csv"], accept_multiple_files=False)
                table_name = st.text_input("Имя таблицы (куда загрузить):", value="my_table")
                mode = st.radio("Режим загрузки:", options=["overwrite (replace table)", "append (to existing)"],
                                index=0)
                normalize_cols = st.checkbox("Нормализовать имена колонок (recommended)", value=True)
                header_row = st.number_input("Номер строки с заголовком (0-based)", min_value=0, value=0)
                submit_csv = st.form_submit_button("Загрузить CSV")

                if submit_csv:
                    if not uploaded:
                        st.warning("Пожалуйста, выберите CSV файл.")
                    else:
                        # backup DB before overwrite
                        backup_needed = mode.startswith("overwrite")
                        if backup_needed and DB_PATH.exists():
                            bak = DB_PATH.with_suffix(
                                DB_PATH.suffix + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                            try:
                                import shutil

                                shutil.copy2(DB_PATH, bak)
                                st.info(f"Бэкап БД создан: {bak.name}")
                            except Exception as e:
                                st.warning(f"Не удалось создать бэкап: {e}")

                        try:
                            # read CSV (attempt encoding auto-detect)
                            import chardet, io

                            raw = uploaded.read()
                            enc = chardet.detect(raw).get("encoding") or "utf-8"
                            df = pd.read_csv(io.BytesIO(raw), encoding=enc, header=header_row)
                            st.write(
                                f"Файл прочитан (encoding={enc}). Размер: {len(df)} строк, {len(df.columns)} колонок.")
                        except Exception as e:
                            st.error(f"Ошибка чтения CSV: {e}")
                            df = None

                        if df is not None:
                            # Optional normalize column names using existing util if available
                            try:
                                from agent.utils.column_normalizer import normalize_dataframe_columns

                                if normalize_cols:
                                    df = normalize_dataframe_columns(df)
                                    st.write("Имена колонок нормализованы.")
                            except Exception:
                                # fallback: simple normalization
                                if normalize_cols:
                                    df.columns = [re.sub(r'[^\w]', '_', str(c)).strip('_') or f"col_{i}" for i, c in
                                                  enumerate(df.columns)]
                                    st.write("Имена колонок простыми правилами нормализованы (fallback).")

                            # Write to SQL (pandas.to_sql)
                            try:
                                if mode.startswith("overwrite"):
                                    if_exists = "replace"
                                else:
                                    if_exists = "append"
                                # Ensure DB dir exists
                                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                                engine_local = create_engine(f"sqlite:///{DB_PATH}", future=True)
                                # write
                                df.to_sql(table_name, con=engine_local, if_exists=if_exists, index=False)
                                st.success(f"CSV успешно загружен в таблицу '{table_name}' (mode={if_exists}).")
                                # refresh inspect / tables
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Ошибка при записи в БД: {e}")

# История запросов
with main_col2:
    st.header("📜 История запросов")

    for i, chat_item in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.container():
            st.markdown(f"**Запрос {len(st.session_state.chat_history) - i}:**")
            st.write(f"📝 {chat_item['query'][:50]}...")
            st.write(f"🎯 {chat_item['type']}")
            st.write(f"⏰ {chat_item['timestamp'][:19]}")

            if st.button(f"Восстановить запрос {len(st.session_state.chat_history) - i}", key=f"restore_{i}"):
                st.session_state.restored_query = chat_item['query']

            st.markdown("---")

# Восстановление запроса
if hasattr(st.session_state, 'restored_query'):
    st.info(f"Восстановленный запрос: {st.session_state.restored_query}")
    if st.button("Использовать этот запрос"):
        query_input = st.session_state.restored_query
        del st.session_state.restored_query

# Инструкции
with st.expander("📚 Инструкции по использованию"):
    st.markdown("""
    ### Как использовать продвинутого цифрового аналитика:

    1. **Выберите тип запроса** - это поможет системе выбрать оптимальную стратегию анализа
    2. **Сформулируйте запрос** - чем конкретнее запрос, тем точнее результаты
    3. **Используйте одинаковый ID сессии** - для сохранения контекста между запросами
    4. **Анализируйте результаты** - используйте все вкладки для полного понимания

    ### Примеры эффективных запросов:
    - "Сколько длился вид работ 'X' для каждого уникального объекта?"
    - "Что если бы ресурсов было на 40% больше, то объект бы построился на Y дней быстрее?"
    - "Какова средняя выработка в сутки по объектам?"
    - "Какие объекты показывают наилучшую эффективность?"

    ### Особенности системы:
    - ✅ Продвинутое Reasoning с пошаговым планированием
    - ✅ Контекстуальное понимание с учетом истории
    - ✅ Генерация сценариев "что если"
    - ✅ Автоматическая валидация результатов
    - ✅ Интерактивные визуализации
    - ✅ Экспорт результатов в JSON
    """)

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Продвинутый цифровой аналитик</strong></p>
    <p>Система с интеллектуальным Reasoning, контекстом и генерацией сценариев</p>
</div>
""", unsafe_allow_html=True)