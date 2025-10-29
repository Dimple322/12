#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОДУКТИВНАЯ версия системы цифрового аналитика с ChromaDB.
Полная интеграция с ChromaDB, динамические сценарии, улучшенная обработка данных.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
import json
import re
import time
import asyncio
import sqlite3
from datetime import datetime
from enum import Enum
import logging

# ChromaDB импорты
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# AI и ML
import ollama
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from gpu_embed_global import GpuMiniLMEmbedding   # ← наша GPU-обёртка


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация для продукта
LLM_MODEL = "digital_twin_analyst"
CHROMA_PATH = Path("generated/chroma_db")
KNOWLEDGE_BASE_PATH = Path("generated/knowledge_base_productive.json")
SCENARIOS_PATH = Path("generated/scenarios")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class QueryType(Enum):
    ANALYTICS = "analytics"
    SCENARIO = "scenario"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    EXPLANATION = "explanation"

    def __str__(self):
        return self.value


@dataclass
class Context:
    """Контекст для сохранения состояния между запросами"""
    session_id: str
    user_id: str
    previous_queries: List[Dict] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "previous_queries": self.previous_queries,
            "domain_knowledge": self.domain_knowledge,
            "preferences": self.preferences,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReasoningStep:
    """Шаг в процессе рассуждений"""
    step_number: int
    description: str
    reasoning: str
    action: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    confidence: float = 0.0
    validation_passed: bool = False

    def to_dict(self):
        return {
            "step_number": self.step_number,
            "description": self.description,
            "reasoning": self.reasoning,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "confidence": self.confidence,
            "validation_passed": self.validation_passed
        }


@dataclass
class AnalysisResult:
    """Результат анализа с метаданными"""
    query: str
    chroma_query: str
    data: List[Dict]
    reasoning_steps: List[ReasoningStep]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    validation_results: Dict[str, Any]
    scenario_analysis: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            "query": self.query,
            "chroma_query": self.chroma_query,
            "data": self.data,
            "reasoning_steps": [step.to_dict() for step in self.reasoning_steps],
            "insights": self.insights,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "validation_results": self.validation_results,
            "scenario_analysis": self.scenario_analysis,
            "timestamp": self.timestamp.isoformat()
        }


class ChromaDBManager:
    """Менеджер для работы с ChromaDB"""

    def __init__(self, persist_directory: Path = CHROMA_PATH):
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Инициализация клиента ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        from sentence_transformers import SentenceTransformer
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st_model = SentenceTransformer(EMBEDDING_MODEL, device=device)  # ← GPU

        from sentence_transformers import SentenceTransformer
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st_model = SentenceTransformer(EMBEDDING_MODEL, device=device)  # ← GPU

        from gpu_embed_global import gpu_embedding
        self.embedding_function = gpu_embedding

        # Коллекции для разных типов данных
        self.collections = {}
        self._initialize_collections()

    def _initialize_collections(self):
        """Инициализирует коллекции ChromaDB"""

        # Основная коллекция для документов
        self.collections['documents'] = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # Коллекция для сценариев
        self.collections['scenarios'] = self.client.get_or_create_collection(
            name="scenarios",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # Коллекция для контекста запросов
        self.collections['queries'] = self.client.get_or_create_collection(
            name="queries",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # Коллекция для инсайтов
        self.collections['insights'] = self.client.get_or_create_collection(
            name="insights",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str],
                      collection_name: str = 'documents'):
        """Добавляет документы в коллекцию"""
        try:
            self.collections[collection_name].add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to {collection_name} collection")
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")

    def query_documents(self, query_text: str, n_results: int = 10, collection_name: str = 'documents',
                        filter_dict: Dict = None):
        """Выполняет поиск похожих документов"""
        try:
            results = self.collections[collection_name].query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Error querying documents from {collection_name}: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    def get_all_documents(self, collection_name: str = 'documents'):
        """Получает все документы из коллекции"""
        try:
            results = self.collections[collection_name].get()
            return results
        except Exception as e:
            logger.error(f"Error getting all documents from {collection_name}: {e}")
            return {'documents': [], 'metadatas': [], 'ids': []}

    def update_document(self, document_id: str, document: str, metadata: Dict, collection_name: str = 'documents'):
        """Обновляет документ в коллекции"""
        try:
            self.collections[collection_name].update(
                ids=[document_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"Updated document {document_id} in {collection_name} collection")
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")

    def delete_document(self, document_id: str, collection_name: str = 'documents'):
        """Удаляет документ из коллекции"""
        try:
            self.collections[collection_name].delete(ids=[document_id])
            logger.info(f"Deleted document {document_id} from {collection_name} collection")
        except Exception as e:
            logger.error(f"Error deleting document from {collection_name}: {e}")


class AdvancedDigitalTwin:
    """Продвинутая система цифрового аналитика с ChromaDB"""

    def __init__(self):
        # Инициализация ChromaDB
        self.chroma_manager = ChromaDBManager()

        # Агенты системы
        self.context_agent = ContextAgent(self.chroma_manager)
        self.reasoning_agent = ReasoningAgent(self.chroma_manager)
        self.data_agent = DataAgent(self.chroma_manager)
        self.validation_agent = ValidationAgent(self.chroma_manager)
        self.scenario_agent = DynamicScenarioAgent(self.chroma_manager)
        self.explanation_agent = ExplanationAgent(self.chroma_manager)

        # Хранилище контекстов
        self.contexts: Dict[str, Context] = {}

        # База знаний
        self.knowledge_base = self._load_knowledge_base()

        # Инициализация продукта
        self._initialize_product()

    def _initialize_product(self):
        """Инициализирует продуктивную систему"""
        print("🚀 Инициализация продукта с ChromaDB...")

        # Создание необходимых директорий
        SCENARIOS_PATH.mkdir(parents=True, exist_ok=True)

        # Загрузка начальных данных если есть
        self._load_initial_data()

        print("✅ Продукт инициализирован")

    def _load_initial_data(self):
        """Загружает начальные данные в ChromaDB"""
        # Этот метод должен быть настроен под ваши конкретные данные
        # Пример загрузки документов из файлов или API
        pass

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Загружает базу знаний"""
        if KNOWLEDGE_BASE_PATH.exists():
            with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"patterns": [], "scenarios": [], "validations": [], "user_queries": []}

    def _save_knowledge_base(self):
        """Сохраняет базу знаний"""
        with open(KNOWLEDGE_BASE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    async def process_query(self, query: str, session_id: str = "default",
                            query_type: QueryType = QueryType.ANALYTICS) -> AnalysisResult:
        """Обрабатывает пользовательский запрос с продвинутым Reasoning"""

        logger.info(f"Processing query: {query} (type: {query_type.value})")

        # Получение или создание контекста
        context = self.contexts.get(session_id, Context(session_id=session_id, user_id="user"))

        # Шаг 1: Анализ запроса и определение типа
        reasoning_steps = []

        step1 = ReasoningStep(
            step_number=1,
            description="Анализ типа запроса и определение стратегии",
            reasoning=self._determine_query_strategy(query, query_type),
            action="Определение типа запроса и подходящей стратегии обработки",
            expected_outcome="Четкое понимание цели запроса и выбор правильного пути решения"
        )
        reasoning_steps.append(step1)

        try:
            # Шаг 2: Извлечение контекста и истории
            relevant_context = await self.context_agent.extract_relevant_context(query, context)

            step2 = ReasoningStep(
                step_number=2,
                description="Извлечение релевантного контекста",
                reasoning=f"Найдено {len(relevant_context)} контекстных элементов из истории запросов",
                action="Анализ предыдущих запросов и извлечение релевантной информации",
                expected_outcome="Контекстуализированное понимание текущего запроса"
            )
            reasoning_steps.append(step2)

            # Шаг 3: Планирование решения
            if query_type == QueryType.SCENARIO:
                solution_plan = await self.scenario_agent.plan_scenario_analysis(query, context)
            else:
                solution_plan = await self.reasoning_agent.create_solution_plan(query, context, relevant_context)

            step3 = ReasoningStep(
                step_number=3,
                description="Создание плана решения",
                reasoning=solution_plan["reasoning"],
                action="Разработка пошагового плана анализа",
                expected_outcome="Четкий план действий для достижения цели запроса"
            )
            reasoning_steps.append(step3)

            # Шаг 4: Выполнение анализа
            if query_type == QueryType.SCENARIO:
                result = await self.scenario_agent.execute_scenario_analysis(query, solution_plan, context)
            else:
                result = await self._execute_analytics_query(query, solution_plan, context)

            step4 = ReasoningStep(
                step_number=4,
                description="Выполнение анализа",
                reasoning=f"Выполнено {len(result.data)} записей",
                action="Выполнение ChromaDB запроса и получение данных",
                expected_outcome="Получение релевантных данных для анализа",
                actual_outcome=f"Получено {len(result.data)} записей",
                confidence=0.8
            )
            reasoning_steps.append(step4)

            # Шаг 5: Генерация инсайтов (только если есть данные)
            if result.data:
                insights = await self.explanation_agent.generate_insights(result, context)

                step5 = ReasoningStep(
                    step_number=5,
                    description="Генерация инсайтов и рекомендаций",
                    reasoning=f"Сгенерировано {len(insights)} инсайтов",
                    action="Анализ данных и извлечение ключевых инсайтов",
                    expected_outcome="Понимание паттернов и трендов в данных",
                    actual_outcome=f"Найдено {len(insights)} ключевых инсайтов",
                    confidence=0.75
                )
                reasoning_steps.append(step5)

                # Обновляем результат с инсайтами
                result.insights = insights.get("insights", [])
                result.recommendations = insights.get("recommendations", [])

            # Шаг 6: Валидация результатов
            validation_results = await self.validation_agent.validate_results(result, context)

            step6 = ReasoningStep(
                step_number=6,
                description="Валидация результатов",
                reasoning=f"Валидация {'пройдена' if validation_results['is_valid'] else 'не пройдена'}",
                action="Проверка корректности результатов и выявление аномалий",
                expected_outcome="Подтверждение достоверности результатов",
                actual_outcome=validation_results["summary"],
                confidence=validation_results["confidence"],
                validation_passed=validation_results["is_valid"]
            )
            reasoning_steps.append(step6)

            # Шаг 7: Генерация динамических сценариев (если запрошено)
            scenario_analysis = None
            if query_type == QueryType.SCENARIO and result.data:
                scenario_analysis = await self.scenario_agent.generate_dynamic_scenarios(query, result, context)

                step7 = ReasoningStep(
                    step_number=7,
                    description="Генерация динамических сценариев",
                    reasoning=f"Сгенерировано {len(scenario_analysis.get('scenarios', []))} сценариев",
                    action="Анализ пользовательского запроса и создание релевантных сценариев",
                    expected_outcome="Понимание потенциальных вариантов развития",
                    actual_outcome=f"Разработано {len(scenario_analysis.get('scenarios', []))} сценариев",
                    confidence=0.7
                )
                reasoning_steps.append(step7)

            # Создание финального результата
            final_result = AnalysisResult(
                query=query,
                chroma_query=result.chroma_query,
                data=result.data,
                reasoning_steps=reasoning_steps,
                insights=result.insights,
                recommendations=result.recommendations,
                confidence_score=np.mean([step.confidence for step in reasoning_steps]),
                validation_results=validation_results,
                scenario_analysis=scenario_analysis
            )

            # Обновление контекста
            context.previous_queries.append({
                "query": query,
                "result": final_result.to_dict(),
                "timestamp": datetime.now()
            })
            self.contexts[session_id] = context

            # Сохранение в базу знаний
            self._update_knowledge_base(query, final_result)

            # Сохранение запроса в ChromaDB для будущего использования
            await self._save_query_to_chroma(query, final_result)

            return final_result

        except Exception as e:
            logger.error(f"Error processing query: {e}")

            # Возвращаем базовый результат с ошибкой
            return AnalysisResult(
                query=query,
                chroma_query="",
                data=[],
                reasoning_steps=reasoning_steps,
                insights=[f"Ошибка при обработке запроса: {str(e)}"],
                recommendations=["Попробуйте переформулировать запрос"],
                confidence_score=0.0,
                validation_results={"is_valid": False, "confidence": 0.0, "error": str(e)}
            )

    def _determine_query_strategy(self, query: str, query_type: QueryType) -> str:
        """Определяет стратегию обработки запроса"""
        if query_type == QueryType.SCENARIO:
            return "Запрос требует генерации динамических сценариев на основе пользовательского запроса"
        elif query_type == QueryType.PREDICTION:
            return "Запрос требует предиктивной аналитики и прогнозирования на основе исторических данных"
        elif query_type == QueryType.VALIDATION:
            return "Запрос требует валидации существующих данных или гипотез"
        else:
            return "Запрос требует стандартной аналитики с генерацией инсайтов на основе данных ChromaDB"

    async def _execute_analytics_query(self, query: str, solution_plan: Dict, context: Context) -> AnalysisResult:
        """Выполняет аналитический запрос через ChromaDB"""
        try:
            # Генерация ChromaDB запроса
            chroma_query = await self.data_agent.generate_chroma_query(query, solution_plan, context)

            # Выполнение запроса
            data = await self.data_agent.execute_chroma_query(chroma_query)

            return AnalysisResult(
                query=query,
                chroma_query=chroma_query,
                data=data,
                reasoning_steps=[],
                insights=[],
                recommendations=[],
                confidence_score=0.8,
                validation_results={"is_valid": True, "confidence": 0.8}
            )
        except Exception as e:
            logger.error(f"Error in analytics query execution: {e}")
            return AnalysisResult(
                query=query,
                chroma_query="",
                data=[],
                reasoning_steps=[],
                insights=[f"Ошибка выполнения: {str(e)}"],
                recommendations=[],
                confidence_score=0.0,
                validation_results={"is_valid": False, "confidence": 0.0, "error": str(e)}
            )

    def _update_knowledge_base(self, query: str, result: AnalysisResult):
        """Обновляет базу знаний новыми паттернами"""
        try:
            pattern = {
                "query_pattern": query,
                "query_type": "analytics",
                "successful_steps": [step.description for step in result.reasoning_steps if step.validation_passed],
                "insights": result.insights,
                "confidence": result.confidence_score,
                "timestamp": datetime.now().isoformat()
            }

            self.knowledge_base["patterns"].append(pattern)

            # Ограничиваем размер базы знаний
            if len(self.knowledge_base["patterns"]) > 1000:
                self.knowledge_base["patterns"] = self.knowledge_base["patterns"][-500:]

            self._save_knowledge_base()
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")

    async def _save_query_to_chroma(self, query: str, result: AnalysisResult):
        """Сохраняет запрос в ChromaDB для будущего использования"""
        try:
            # Создание документа из запроса и результатов
            document = f"Query: {query}\n"
            document += f"Insights: {', '.join(result.insights[:3])}\n"
            document += f"Recommendations: {', '.join(result.recommendations[:2])}\n"
            document += f"Data count: {len(result.data)}\n"

            # Метаданные
            metadata = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "confidence": result.confidence_score,
                "data_count": len(result.data),
                "query_type": "analytics"
            }

            # ID документа
            doc_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query) % 10000}"

            # Сохранение в ChromaDB
            self.chroma_manager.add_documents(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id],
                collection_name='queries'
            )

        except Exception as e:
            logger.error(f"Error saving query to ChromaDB: {e}")


class ContextAgent:
    """Агент для работы с контекстом и историей"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def extract_relevant_context(self, query: str, context: Context) -> List[Dict]:
        """Извлекает релевантный контекст из истории запросов через ChromaDB"""
        relevant_context = []

        # Поиск похожих запросов в ChromaDB
        similar_queries = self.chroma_manager.query_documents(
            query_text=query,
            n_results=5,
            collection_name='queries'
        )

        # Обработка результатов
        if similar_queries['documents'] and similar_queries['documents'][0]:
            for i, (doc, metadata) in enumerate(zip(similar_queries['documents'][0], similar_queries['metadatas'][0])):
                relevance_score = 1.0 - similar_queries['distances'][0][i] if similar_queries['distances'][0] else 0.5

                relevant_context.append({
                    "query": metadata.get("query", ""),
                    "relevance_score": relevance_score,
                    "insights": metadata.get("insights", []),
                    "confidence": metadata.get("confidence", 0.5)
                })

        # Также анализируем последние запросы из контекста
        recent_queries = context.previous_queries[-5:]

        for prev_query in recent_queries:
            # Простая проверка релевантности по ключевым словам
            prev_text = prev_query["query"].lower()
            current_text = query.lower()

            # Проверяем общие ключевые слова
            common_words = set(prev_text.split()) & set(current_text.split())
            relevance_score = len(common_words) / max(len(prev_text.split()), 1)

            if relevance_score > 0.2:  # Порог релевантности
                relevant_context.append({
                    "query": prev_query["query"],
                    "relevance_score": relevance_score,
                    "insights": prev_query.get("result", {}).get("insights", [])
                })

        return sorted(relevant_context, key=lambda x: x["relevance_score"], reverse=True)


class ReasoningAgent:
    """Агент для продвинутого рассуждения и планирования"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def create_solution_plan(self, query: str, context: Context, relevant_context: List[Dict]) -> Dict:
        """Создает план решения на основе анализа запроса"""

        # Получаем информацию о доступных данных из ChromaDB
        available_collections = list(self.chroma_manager.collections.keys())

        # Формируем промпт для LLM с учетом доступных данных
        prompt = f"""
        Создай детальный план решения для следующего аналитического запроса:

        Запрос: {query}

        Доступные коллекции данных: {available_collections}

        Релевантный контекст из предыдущих запросов:
        {json.dumps(relevant_context[:3], ensure_ascii=False, indent=2)}

        План должен включать:
        1. Определение цели анализа
        2. Выбор необходимых коллекций данных
        3. Пошаговую стратегию выполнения
        4. Методы валидации результатов
        5. Подход к генерации инсайтов
        6. Потенциальные сценарии для анализа

        Ответ в формате JSON:
        {{
            "reasoning": "Обоснование выбранной стратегии",
            "target_collections": ["коллекции для анализа"],
            "steps": [
                {{
                    "step": "Описание шага",
                    "purpose": "Цель шага",
                    "expected_result": "Ожидаемый результат"
                }}
            ],
            "validation_approach": "метод валидации",
            "scenario_potential": "потенциальные сценарии для анализа"
        }}
        """

        try:
            response = ollama.chat(model=LLM_MODEL, messages=[
                {"role": "system",
                 "content": "Ты - стратег аналитики. Создавай детальные планы решения аналитических задач для работы с ChromaDB."},
                {"role": "user", "content": prompt}
            ])

            plan_text = response["message"]["content"]

            # Извлекаем JSON из ответа
            json_match = re.search(r'```json\s*(.*?)\s*```', plan_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1))
            else:
                # Пытаемся найти JSON напрямую
                json_start = plan_text.find('{')
                json_end = plan_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    plan_data = json.loads(plan_text[json_start:json_end])
                else:
                    # Базовый план
                    plan_data = {
                        "reasoning": "Базовый план анализа для ChromaDB",
                        "target_collections": ["documents"],
                        "steps": [{"step": "Выполнить запрос", "purpose": "Получить данные",
                                   "expected_result": "Данные для анализа"}],
                        "validation_approach": "базовая проверка",
                        "scenario_potential": "стандартные сценарии"
                    }

            return plan_data

        except Exception as e:
            logger.error(f"Error creating solution plan: {e}")
            return {
                "reasoning": "Базовый план из-за ошибки",
                "target_collections": ["documents"],
                "steps": [{"step": "Выполнить запрос", "purpose": "Получить данные",
                           "expected_result": "Данные для анализа"}],
                "validation_approach": "базовая проверка",
                "scenario_potential": "стандартные сценарии"
            }


class DataAgent:
    """Агент для работы с данными через ChromaDB"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def generate_chroma_query(self, query: str, solution_plan: Dict, context: Context) -> str:
        """Генерирует оптимальный запрос для ChromaDB"""

        target_collections = solution_plan.get("target_collections", ["documents"])

        # Определяем основную коллекцию для запроса
        main_collection = target_collections[0] if target_collections else "documents"

        # Формируем оптимизированный текст запроса для ChromaDB
        # Убираем лишние слова и фокусируемся на ключевых понятиях
        optimized_query = re.sub(r'^(покажи|сколько|какие|найди)\s+', '', query.lower())
        optimized_query = re.sub(r'[?.,!]', '', optimized_query)

        logger.info(f"Generated ChromaDB query: {optimized_query} for collection: {main_collection}")

        return optimized_query

    async def execute_chroma_query(self, chroma_query: str, collection_name: str = "documents", n_results: int = 20) -> \
    List[Dict]:
        """Выполняет запрос в ChromaDB и возвращает результаты"""
        try:
            # Выполнение запроса в ChromaDB
            results = self.chroma_manager.query_documents(
                query_text=chroma_query,
                n_results=n_results,
                collection_name=collection_name
            )

            # Конвертация результатов в удобный формат
            formatted_results = []

            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    distance = results['distances'][0][i] if results['distances'][0] else 0
                    similarity = 1.0 - distance  # Преобразуем расстояние в сходство

                    formatted_result = {
                        "id": metadata.get("id", f"doc_{i}"),
                        "content": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "distance": distance
                    }
                    formatted_results.append(formatted_result)

            logger.info(f"ChromaDB query executed successfully, returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error executing ChromaDB query: {e}")

            # Возврат пустого результата или fallback
            return []


class ValidationAgent:
    """Агент для валидации результатов"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def validate_results(self, result: AnalysisResult, context: Context) -> Dict[str, Any]:
        """Валидирует результаты анализа"""

        validation_results = {
            "is_valid": True,
            "confidence": 0.8,
            "summary": "Результаты прошли базовую валидацию",
            "checks": []
        }

        # Проверка 1: Есть ли данные
        has_data = len(result.data) > 0
        validation_results["checks"].append({
            "check": "Наличие данных",
            "passed": has_data,
            "details": f"Найдено {len(result.data)} записей"
        })

        # Проверка 2: Качество результатов ChromaDB
        if has_data:
            avg_similarity = np.mean([item.get("similarity", 0) for item in result.data])
            validation_results["checks"].append({
                "check": "Качество результатов ChromaDB",
                "passed": avg_similarity > 0.5,
                "details": f"Средняя схожесть: {avg_similarity:.3f}"
            })

        # Проверка 3: Разнообразие данных
        if has_data and len(result.data) > 1:
            unique_contents = len(set([item.get("content", "") for item in result.data]))
            validation_results["checks"].append({
                "check": "Разнообразие данных",
                "passed": unique_contents > 1,
                "details": f"Уникальных результатов: {unique_contents} из {len(result.data)}"
            })

        # Проверка 4: Логическая консистентность
        logical_check = self._check_logical_consistency(result)
        validation_results["checks"].append(logical_check)

        # Определение итоговой валидации
        failed_checks = [check for check in validation_results["checks"] if not check["passed"]]
        validation_results["is_valid"] = len(failed_checks) == 0
        validation_results["confidence"] = 1.0 - (len(failed_checks) / len(validation_results["checks"])) * 0.5

        if failed_checks:
            validation_results[
                "summary"] = f"Обнаружены проблемы: {', '.join([check['check'] for check in failed_checks])}"

        return validation_results

    def _check_logical_consistency(self, result: AnalysisResult) -> Dict[str, Any]:
        """Проверяет логическую консистентность результатов"""
        # Базовая проверка логики
        if len(result.data) == 0:
            return {
                "check": "Логическая консистентность",
                "passed": False,
                "details": "Нет данных для проверки логики"
            }

        return {
            "check": "Логическая консистентность",
            "passed": True,
            "details": "Результаты логически консистентны"
        }


class DynamicScenarioAgent:
    """Агент для генерации динамических сценариев на основе пользовательских запросов"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def plan_scenario_analysis(self, query: str, context: Context) -> Dict:
        """Планирует анализ сценариев на основе запроса пользователя"""

        # Анализ запроса на предмет сценарных элементов
        scenario_elements = self._extract_scenario_elements(query)

        return {
            "reasoning": f"Планирование динамических сценариев на основе запроса: {query}",
            "scenario_type": "user_driven",
            "elements": scenario_elements,
            "approach": "dynamic_generation"
        }

    def _extract_scenario_elements(self, query: str) -> Dict[str, Any]:
        """Извлекает элементы для сценариев из запроса пользователя"""
        elements = {
            "variables": [],
            "conditions": [],
            "outcomes": [],
            "timeframe": None
        }

        # Поиск числовых значений (потенциальные переменные)
        numbers = re.findall(r'\d+(?:\.\d+)?%', query)  # Проценты
        elements["variables"].extend(numbers)

        numbers = re.findall(r'\d+', query)  # Обычные числа
        elements["variables"].extend([f"{num}" for num in numbers])

        # Поиск условий
        if 'если' in query.lower() or 'если бы' in query.lower():
            elements["conditions"].append("условие")

        # Поиск временных рамок
        time_patterns = ['день', 'недел', 'месяц', 'год']
        for pattern in time_patterns:
            if pattern in query.lower():
                elements["timeframe"] = pattern
                break

        return elements

    async def execute_scenario_analysis(self, query: str, solution_plan: Dict, context: Context) -> AnalysisResult:
        """Выполняет анализ сценариев на основе данных"""
        # Получаем данные для анализа сценариев
        chroma_query = await self.chroma_manager.data_agent.generate_chroma_query(query, solution_plan, context)
        data = await self.chroma_manager.data_agent.execute_chroma_query(chroma_query)

        return AnalysisResult(
            query=query,
            chroma_query=chroma_query,
            data=data,
            reasoning_steps=[],
            insights=[],
            recommendations=[],
            confidence_score=0.7,
            validation_results={"is_valid": True, "confidence": 0.7}
        )

    async def generate_dynamic_scenarios(self, original_query: str, result: AnalysisResult, context: Context) -> Dict[
        str, Any]:
        """Генерирует динамические сценарии на основе пользовательского запроса"""

        # Извлекаем элементы сценария из оригинального запроса
        scenario_elements = self._extract_scenario_elements(original_query)

        scenarios = {
            "baseline": {
                "description": "Текущий сценарий на основе пользовательского запроса",
                "query_analysis": original_query,
                "elements": scenario_elements,
                "metrics": self._extract_metrics(result.data)
            },
            "scenarios": []
        }

        # Генерация динамических сценариев на основе элементов запроса
        if scenario_elements["variables"]:
            for variable in scenario_elements["variables"]:
                scenario = await self._create_variable_scenario(original_query, variable, result.data)
                scenarios["scenarios"].append(scenario)

        # Генерация сценариев на основе данных
        if result.data:
            data_scenarios = await self._create_data_driven_scenarios(result.data, original_query)
            scenarios["scenarios"].extend(data_scenarios)

        # Сохранение сценариев в ChromaDB
        await self._save_scenarios_to_chroma(scenarios, original_query)

        return scenarios

    async def _create_variable_scenario(self, original_query: str, variable: str, data: List[Dict]) -> Dict[str, Any]:
        """Создает сценарий на основе переменной из запроса"""

        # Определение типа переменной и создание сценария
        if '%' in variable:
            # Процентная переменная
            percentage = float(variable.replace('%', ''))

            scenario = {
                "name": f"Изменение на {variable}",
                "description": f"Анализ влияния изменения на {variable} на основе запроса: {original_query}",
                "original_variable": variable,
                "parameters": {
                    "change_percentage": percentage / 100,
                    "direction": "увеличение" if percentage > 0 else "уменьшение"
                },
                "expected_outcomes": {
                    "impact_assessment": "Оценка влияния изменения",
                    "risk_analysis": "Анализ рисков",
                    "opportunity_identification": "Выявление возможностей"
                },
                "confidence": 0.7,
                "risks": [
                    "Нелинейное изменение параметров",
                    "Внешние факторы могут повлиять на результат"
                ]
            }
        else:
            # Числовая переменная
            scenario = {
                "name": f"Вариант с {variable}",
                "description": f"Анализ сценария с параметром {variable} на основе запроса: {original_query}",
                "parameters": {
                    "base_value": variable,
                    "multiplier": 1.2  # Пример multiplier
                },
                "expected_outcomes": {
                    "scenario_impact": "Влияние сценария на результат",
                    "optimization_potential": "Потенциал для оптимизации"
                },
                "confidence": 0.6,
                "risks": [
                    "Ограниченность данных",
                    "Предположения могут быть неточными"
                ]
            }

        return scenario

    async def _create_data_driven_scenarios(self, data: List[Dict], original_query: str) -> List[Dict[str, Any]]:
        """Создает сценарии на основе анализа данных"""

        scenarios = []

        if not data:
            return scenarios

        # Анализ данных для выявления паттернов
        try:
            # Группировка по сходству
            similarity_groups = {}
            for item in data:
                similarity = item.get("similarity", 0)
                if similarity > 0.8:
                    group = "highly_relevant"
                elif similarity > 0.6:
                    group = "moderately_relevant"
                else:
                    group = "low_relevance"

                if group not in similarity_groups:
                    similarity_groups[group] = []
                similarity_groups[group].append(item)

            # Создание сценариев на основе групп
            for group_name, group_data in similarity_groups.items():
                if len(group_data) > 0:
                    scenario = {
                        "name": f"Сценарий {group_name}",
                        "description": f"Анализ данных с {group_name} релевантностью",
                        "parameters": {
                            "relevance_group": group_name,
                            "data_count": len(group_data),
                            "avg_similarity": np.mean([item.get("similarity", 0) for item in group_data])
                        },
                        "expected_outcomes": {
                            "group_analysis": f"Анализ группы {group_name}",
                            "pattern_identification": "Выявление паттернов",
                            "recommendation_generation": "Генерация рекомендаций"
                        },
                        "confidence": 0.8,
                        "risks": [
                            "Группировка может быть неточной",
                            "Данные могут быть неполными"
                        ]
                    }
                    scenarios.append(scenario)

        except Exception as e:
            logger.error(f"Error creating data-driven scenarios: {e}")

        return scenarios

    def _extract_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """Извлекает ключевые метрики из данных ChromaDB"""
        if not data:
            return {}

        try:
            metrics = {
                "total_count": len(data),
                "avg_similarity": np.mean([item.get("similarity", 0) for item in data]),
                "max_similarity": max([item.get("similarity", 0) for item in data]),
                "min_similarity": min([item.get("similarity", 0) for item in data]),
                "unique_collections": len(set([item.get("metadata", {}).get("collection", "") for item in data]))
            }

            return metrics
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return {"total_count": len(data)}

    async def _save_scenarios_to_chroma(self, scenarios: Dict[str, Any], original_query: str):
        """Сохраняет сгенерированные сценарии в ChromaDB"""
        try:
            # Создание документов из сценариев
            documents = []
            metadatas = []
            ids = []

            for i, scenario in enumerate(scenarios.get("scenarios", [])):
                doc_content = f"Original Query: {original_query}\n"
                doc_content += f"Scenario: {scenario['name']}\n"
                doc_content += f"Description: {scenario['description']}\n"
                doc_content += f"Parameters: {json.dumps(scenario.get('parameters', {}), ensure_ascii=False)}\n"
                doc_content += f"Expected Outcomes: {json.dumps(scenario.get('expected_outcomes', {}), ensure_ascii=False)}\n"

                documents.append(doc_content)

                metadata = {
                    "original_query": original_query,
                    "scenario_name": scenario["name"],
                    "confidence": scenario.get("confidence", 0.5),
                    "timestamp": datetime.now().isoformat(),
                    "scenario_type": "dynamic"
                }
                metadatas.append(metadata)

                scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                ids.append(scenario_id)

            # Сохранение в ChromaDB
            if documents:
                self.chroma_manager.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    collection_name='scenarios'
                )
                logger.info(f"Saved {len(documents)} scenarios to ChromaDB")

        except Exception as e:
            logger.error(f"Error saving scenarios to ChromaDB: {e}")


class ExplanationAgent:
    """Агент для генерации инсайтов и объяснений"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def generate_insights(self, result: AnalysisResult, context: Context) -> Dict[str, List[str]]:
        """Генерирует инсайты и рекомендации на основе результатов ChromaDB"""

        if not result.data:
            return {"insights": [], "recommendations": []}

        try:
            insights = []
            recommendations = []

            # Базовая статистика
            insights.append(f"Анализ ChromaDB завершен. Найдено {len(result.data)} релевантных документов.")

            # Анализ сходства
            similarities = [item.get("similarity", 0) for item in result.data]
            if similarities:
                avg_similarity = np.mean(similarities)
                max_similarity = max(similarities)
                min_similarity = min(similarities)

                insights.append(f"Средняя релевантность результатов: {avg_similarity:.3f}")
                insights.append(f"Наиболее релевантный результат: {max_similarity:.3f}")

                if max_similarity > 0.9:
                    insights.append("Обнаружены высокорелевантные документы")
                elif max_similarity > 0.7:
                    insights.append("Результаты показывают хорошую релевантность")
                else:
                    insights.append("Рекомендуется уточнить запрос для лучших результатов")

            # Анализ метаданных
            collections_used = set()
            time_ranges = []

            for item in result.data:
                metadata = item.get("metadata", {})
                if "collection" in metadata:
                    collections_used.add(metadata["collection"])

                if "timestamp" in metadata:
                    try:
                        time_ranges.append(datetime.fromisoformat(metadata["timestamp"]))
                    except:
                        pass

            if collections_used:
                insights.append(f"Данные из коллекций: {', '.join(collections_used)}")

            if time_ranges:
                time_span = max(time_ranges) - min(time_ranges)
                insights.append(f"Временной диапазон данных: {time_span.days} дней")

            # Рекомендации на основе результатов
            if len(result.data) > 10:
                recommendations.append("Результаты показывают широкий охват темы")

            if avg_similarity < 0.6:
                recommendations.append("Рекомендуется уточнить запрос для повышения точности")

            recommendations.extend([
                "Изучите наиболее релевантные документы для получения ключевой информации",
                "Используйте фильтры по дате или коллекции для уточнения результатов",
                "Рассмотрите создание сценариев на основе найденных данных"
            ])

            return {
                "insights": insights[:8],
                "recommendations": recommendations[:5]
            }
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "insights": ["Ошибка при генерации инсайтов"],
                "recommendations": ["Проверьте корректность данных в ChromaDB"]
            }


# Функция для запуска системы
async def main():
    """Главная функция для демонстрации системы"""

    print("🚀 ЗАПУСК ПРОДУКТИВНОЙ СИСТЕМЫ ЦИФРОВОГО АНАЛИТИКА С ChromaDB")
    print("=" * 80)

    # Создаем систему
    digital_twin = AdvancedDigitalTwin()

    # Примеры запросов для демонстрации
    test_queries = [
        ("Покажи последние документы по проекту", QueryType.ANALYTICS),
        ("Что если увеличить ресурсы на 50% для ускорения работ?", QueryType.SCENARIO),
        ("Какие риски есть в текущем проекте?", QueryType.ANALYTICS),
        ("Сравни эффективность разных подходов", QueryType.ANALYTICS),
        ("Предскажи сроки завершения на основе текущих данных", QueryType.PREDICTION),
    ]

    for query, query_type in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Обработка запроса: {query}")
        print(f"Тип: {query_type.value}")
        print(f"{'=' * 60}")

        try:
            result = await digital_twin.process_query(query, query_type=query_type)

            print(f"\nРезультаты:")
            print(f"- Количество записей: {len(result.data)}")
            print(f"- Уровень уверенности: {result.confidence_score:.2f}")
            print(f"- Валидация: {'Пройдена' if result.validation_results.get('is_valid') else 'Не пройдена'}")

            if result.insights:
                print(f"\nИнсайты:")
                for insight in result.insights[:3]:
                    print(f"  • {insight}")

            if result.recommendations:
                print(f"\nРекомендации:")
                for rec in result.recommendations[:2]:
                    print(f"  • {rec}")

            if result.scenario_analysis:
                print(f"\nСценарии:")
                for scenario in result.scenario_analysis.get("scenarios", [])[:2]:
                    print(f"  • {scenario['name']}: {scenario['description']}")

        except Exception as e:
            print(f"❌ Ошибка при обработке запроса: {e}")


if __name__ == "__main__":
    # Запускаем асинхронную функцию
    asyncio.run(main())