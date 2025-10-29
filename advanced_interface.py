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

# Основная область
main_col1, main_col2 = st.columns([2, 1])

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
    submit_button = st.button("🚀 Анализировать", type="primary", use_container_width=True)

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
            result = asyncio.run(st.session_state.digital_twin.process_query(
                query=query_input,
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Результаты", "🧠 Ход рассуждений", "💡 Инсайты", "🎯 Сценарии", "📈 Визуализация"
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
                st.dataframe(df, use_container_width=True)
            except Exception:
                # safety: convert to strings
                df = pd.DataFrame(
                    [{k: str(v) for k, v in (r.items() if isinstance(r, dict) else [])} for r in data_obj])
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Данные отсутствуют.")

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