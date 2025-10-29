#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-интерфейс для продвинутой системы цифрового аналитика с Reasoning.
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
            
            # Сохранение в историю
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
    
    # Вкладки для отображения различных аспектов результата
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Результаты", "🧠 Ход рассуждений", "💡 Инсайты", "🎯 Сценарии", "📈 Визуализация"
    ])
    
    with tab1:
        st.header("Результаты анализа")
        
        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Записей найдено", len(latest_result.data))
        
        with col2:
            confidence_color = "high" if latest_result.confidence_score > 0.8 else "medium" if latest_result.confidence_score > 0.6 else "low"
            st.metric("Уверенность", f"{latest_result.confidence_score:.2f}")
        
        with col3:
            validation_status = "✅ Пройдена" if latest_result.validation_results.get("is_valid") else "❌ Ошибки"
            st.metric("Валидация", validation_status)
        
        with col4:
            st.metric("Шагов рассуждений", len(latest_result.reasoning_steps))
        
        # Данные
        if latest_result.data:
            st.subheader("📋 Данные")
            df = pd.DataFrame(latest_result.data)
            st.dataframe(df, use_container_width=True)
            
            # SQL запрос
            with st.expander("🔍 SQL запрос"):
                st.code(latest_result.sql_query, language="sql")
    
    with tab2:
        st.header("Ход рассуждений")
        
        for step in latest_result.reasoning_steps:
            with st.container():
                st.markdown(f'<div class="reasoning-step">', unsafe_allow_html=True)
                st.markdown(f"**Шаг {step.step_number}: {step.description}**")
                st.write(f"🤔 Обоснование: {step.reasoning}")
                st.write(f"⚡ Действие: {step.action}")
                st.write(f"🎯 Ожидаемый результат: {step.expected_outcome}")
                
                if step.actual_outcome:
                    st.write(f"✅ Фактический результат: {step.actual_outcome}")
                
                # Индикатор уверенности
                confidence_color = "high" if step.confidence > 0.8 else "medium" if step.confidence > 0.6 else "low"
                st.markdown(f'<span class="confidence-{confidence_color}">Уверенность: {step.confidence:.2f}</span>', 
                           unsafe_allow_html=True)
                
                if hasattr(step, 'validation_passed') and step.validation_passed:
                    st.success("✅ Валидация пройдена")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("Инсайты и рекомендации")
        
        # Инсайты
        if latest_result.insights:
            st.subheader("💡 Инсайты")
            for insight in latest_result.insights:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        # Рекомендации
        if latest_result.recommendations:
            st.subheader("🎯 Рекомендации")
            for rec in latest_result.recommendations:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
        
        # Валидация
        if latest_result.validation_results.get("checks"):
            st.subheader("🔍 Результаты валидации")
            for check in latest_result.validation_results["checks"]:
                status = "✅" if check["passed"] else "❌"
                st.write(f"{status} **{check['check']}**: {check['details']}")
    
    with tab4:
        st.header("Сценарии 'что если'")
        
        if latest_result.scenario_analysis:
            # Базовый сценарий
            if "baseline" in latest_result.scenario_analysis:
                baseline = latest_result.scenario_analysis["baseline"]
                st.subheader("📊 Базовый сценарий")
                st.write(f"**Описание:** {baseline['description']}")
                
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
            if "scenarios" in latest_result.scenario_analysis:
                st.subheader("🔄 Альтернативные сценарии")
                
                for scenario in latest_result.scenario_analysis["scenarios"]:
                    with st.expander(f"**{scenario['name']}**"):
                        st.write(f"**Описание:** {scenario['description']}")
                        
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
        
        if latest_result.data:
            df = pd.DataFrame(latest_result.data)
            
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

# История запросов
with main_col2:
    st.header("📜 История запросов")
    
    for i, chat_item in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.container():
            st.markdown(f"**Запрос {len(st.session_state.chat_history)-i}:**")
            st.write(f"📝 {chat_item['query'][:50]}...")
            st.write(f"🎯 {chat_item['type']}")
            st.write(f"⏰ {chat_item['timestamp'][:19]}")
            
            if st.button(f"Восстановить запрос {len(st.session_state.chat_history)-i}", key=f"restore_{i}"):
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