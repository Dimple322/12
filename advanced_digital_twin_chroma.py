# advanced_digital_twin_chroma.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Продвинутая система цифрового аналитика с гибридной логикой:
- routing: sql / chroma / hybrid / schema
- capture raw LLM traces (planner / executor / explainer)
- hybrid execution: Chroma -> (optionally) SQL via ThinkingAgent -> SQLAgent
- AnalysisResult unified shape with serialization
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import re
import time
import asyncio
from datetime import datetime
import logging

# ChromaDB
import chromadb

# AI / LLM
import ollama
import numpy as np

# local agents & utils
from agent.sql_agent import SQLAgent
from agent.thinking_agent import ThinkingAgent
from agent.llm.ollama_client import AnalystLLM
from agent.utils.config import (
    DB_PATH as CFG_DB_PATH,
    CHROMA_PATH as CFG_CHROMA_PATH,
    KNOWLEDGE_BASE_PATH as CFG_KB_PATH,
    LLM_MODEL as CFG_LLM_MODEL,
    ensure_directories,
)
from gpu_embed_global import gpu_embedding

# For types already in file we keep minimal imports (plotting etc are used by UI)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Resolve configuration (prefer productive data)
ensure_directories()
LLM_MODEL = CFG_LLM_MODEL
CHROMA_PATH = CFG_CHROMA_PATH
KNOWLEDGE_BASE_PATH = CFG_KB_PATH
DB_PATH = CFG_DB_PATH  # sqlite DB used by SQLAgent / schema queries

class QueryType:
    ANALYTICS = "analytics"
    SCENARIO = "scenario"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    EXPLANATION = "explanation"

@dataclass
class ReasoningStep:
    step_number: float
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
    """Унифицированный результат — теперь включает answer, route, raw traces и evidence."""
    query: str
    chroma_query: Optional[str] = None
    data: List[Dict] = field(default_factory=list)
    reasoning_steps: List[Any] = field(default_factory=list)  # ReasoningStep или dict
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)
    scenario_analysis: Optional[Dict[str, Any]] = None

    # Новые/опциональные поля
    route: Optional[str] = None                      # "sql" | "chroma" | "hybrid" | "schema" | "error"
    answer: Optional[str] = None                     # Краткий человекочитаемый ответ
    raw_llm_traces: Dict[str, str] = field(default_factory=dict)  # planner/executor/explainer raw outputs
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # deterministic evidence (value, count, sample_rows)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "route": self.route,
            "answer": self.answer,
            "chroma_query": self.chroma_query,
            "data": self.data,
            "reasoning_steps": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.reasoning_steps],
            "insights": self.insights,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "validation_results": self.validation_results,
            "scenario_analysis": self.scenario_analysis,
            "raw_llm_traces": self.raw_llm_traces,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat()
        }


class ChromaDBManager:
    def __init__(self, persist_directory: Path = CHROMA_PATH):
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        # collections init
        self.collections = {}
        self._initialize_collections()

    def _initialize_collections(self):
        # use global embedding function
        try:
            self.collections['documents'] = self.client.get_or_create_collection(name="documents", embedding_function=gpu_embedding)
            self.collections['queries'] = self.client.get_or_create_collection(name="queries", embedding_function=gpu_embedding)
            self.collections['insights'] = self.client.get_or_create_collection(name="insights", embedding_function=gpu_embedding)
            self.collections['scenarios'] = self.client.get_or_create_collection(name="scenarios", embedding_function=gpu_embedding)
        except Exception as e:
            logger.warning("ChromaDB initialize error: %s", e)

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str], collection_name: str = 'documents'):
        try:
            self.collections[collection_name].add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error("Chroma add error: %s", e)

    def query_documents(self, query_text: str, n_results: int = 10, collection_name: str = 'documents', filter_dict: Dict = None):
        try:
            col = self.collections.get(collection_name)
            if col is None:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            results = col.query(query_texts=[query_text], n_results=n_results, where=filter_dict)
            return results
        except Exception as e:
            logger.error("Chroma query error: %s", e)
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

class Context:
    def __init__(self, session_id: str, user_id: str = "user"):
        self.session_id = session_id
        self.user_id = user_id
        self.previous_queries: List[Dict] = []
        self.domain_knowledge: Dict[str, Any] = {}
        self.preferences: Dict[str, Any] = {}

class ReasoningAgent:
    """Planner - returns structured plan, raw trace and routing decision"""
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def create_solution_plan(self, query: str, context: Context, relevant_context: List[Dict]) -> Dict[str, Any]:
        """
        Returns dict:
          {"plan": {...}, "raw": "<raw llm text>", "route": "sql|chroma|hybrid|schema"}
        For schema-related queries we return a deterministic plan and DO NOT call LLM
        (avoids LLM suggesting searching Chroma for schema info).
        """
        # decide route first
        route = self._decide_route(query)

        # If the query is about schema/column names — return deterministic plan (no LLM)
        if route == "schema":
            plan_data = {
                "reasoning": "Deterministic schema read: fetch table list and PRAGMA table_info for table 'data' if exists, otherwise first table.",
                "target_collections": [],
                "steps": [
                    {
                        "step": "Find SQLite table 'data' or fallback to first table",
                        "purpose": "Determine which table to inspect for column names",
                        "expected_result": "Table name to inspect"
                    },
                    {
                        "step": "Run PRAGMA table_info(<table>) to get columns",
                        "purpose": "Return list of columns",
                        "expected_result": "List of column names"
                    }
                ],
                "validation_approach": "Schema returned directly from SQLite PRAGMA is authoritative",
                "scenario_potential": "Applicable for all schema/column name queries"
            }
            raw = "[deterministic planner] schema read (no LLM)"
            return {"plan": plan_data, "raw": raw, "route": "schema"}

        # Otherwise fall back to LLM planner (existing behaviour)
        prompt = f"""
    Create a JSON plan for the analytic query.
    Query: {query}
    Detected route suggestion: {route}
    Relevant context (examples): {json.dumps(relevant_context[:3], ensure_ascii=False)}
    Available collections: {list(self.chroma_manager.collections.keys())}

    Return both: a short textual reasoning and a JSON plan block. Example JSON schema:
    {{"reasoning":"...","target_collections":["documents"],"steps":[{{"step":"...","purpose":"...","expected_result":"..."}}],"validation_approach":"...","scenario_potential":"..."}}
    """
        try:
            resp = ollama.chat(model=LLM_MODEL, messages=[
                {"role": "system",
                 "content": "You are an analytics planner. Provide a short reasoning and a JSON plan."},
                {"role": "user", "content": prompt}
            ])
            raw = resp["message"]["content"]
        except Exception as e:
            logger.error("LLM planner error: %s", e)
            raw = f"[LLM planner error]: {e}"
        plan_data = None
        try:
            m = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
            if m:
                plan_data = json.loads(m.group(1))
            else:
                # try to extract first {...}
                start = raw.find('{')
                end = raw.rfind('}') + 1
                if start != -1 and end > start:
                    plan_data = json.loads(raw[start:end])
        except Exception as e:
            logger.debug("Planner JSON parse failed: %s", e)
            plan_data = {
                "reasoning": f"Fallback plan (couldn't parse JSON): see raw planner trace",
                "target_collections": ["documents"],
                "steps": [
                    {"step": "semantic_search", "purpose": "get relevant docs", "expected_result": "relevant docs"}],
                "validation_approach": "basic",
                "scenario_potential": "default"
            }
        # allow override of route from parsed plan
        if isinstance(plan_data, dict) and plan_data.get("route"):
            route = plan_data.get("route")
        return {"plan": plan_data, "raw": raw, "route": route}

        # Замените метод ReasoningAgent._decide_route на следующий (вставьте вместо старого)
    def _decide_route(self, query: str) -> str:
            q = (query or "").lower().strip()
            # explicit force tokens (user can prefix query with "schema:" etc.)
            if q.startswith("schema:") or q.startswith("[schema]") or q.startswith("sql:"):
                return "schema"
            if q.startswith("chroma:") or q.startswith("[chroma]"):
                return "chroma"
            if q.startswith("hybrid:") or q.startswith("[hybrid]"):
                return "hybrid"

            # schema / column questions (широкий набор русских синонимов)
            if re.search(
                    r'(колонк|столбц|наименован|наименование|назван|названий|название|имена\s+столбц|какие\s+столбцы|имена столбцов|какое\s+наименование|как\W+называют\W+колонки|список\s+столбцов)',
                    q):
                return "schema"

            # SQL-like / aggregation
            if re.search(
                    r'\b(max|min|count|sum|avg|group by|having|distinct|order by|сколько|сумма|средн|максимальн|минимальн|самая поздн)\b',
                    q):
                return "sql"

            # textual / document search terms
            if re.search(r'\b(документ|статья|отчет|отчёт|где говорится|опишите|найди|описание|файл|док)\b', q):
                return "chroma"

            # fallback hybrid
            return "hybrid"

class DataAgent:
    """Агент для работы с данными через ChromaDB и (опционально) SQL via ThinkingAgent -> SQLAgent."""

    def __init__(self, chroma_manager):
        self.chroma_manager = chroma_manager

    async def generate_chroma_query(self, query: str, solution_plan: Dict, context: Context) -> str:
        # текущая реализация сохраняем
        optimized_query = re.sub(r'^(покажи|сколько|какие|найди)\s+', '', query.lower())
        optimized_query = re.sub(r'[?.,!]', '', optimized_query)
        plan_steps = solution_plan.get("steps") if isinstance(solution_plan, dict) else None
        if plan_steps:
            optimized_query = optimized_query + " | plan: " + "; ".join([s.get("step","") for s in plan_steps[:3]])
        return optimized_query

    async def execute_chroma_query(self, chroma_query: str, collection_name: str = "documents", n_results: int = 20):
        results = self.chroma_manager.query_documents(query_text=chroma_query, n_results=n_results,
                                                      collection_name=collection_name)
        formatted = []
        if results and results.get('documents') and results['documents'][0]:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                distance = results['distances'][0][i] if results['distances'][0] else 0
                flat = {
                    "id": meta.get("id", f"doc_{i}"),
                    "content": doc,
                    "distance": distance,
                    "similarity": 1.0 - distance if distance is not None else None,
                    # все поля из meta на верхний уровень
                    **{k: v for k, v in meta.items()}
                }
                # оставляем metadata для обратной совместимости
                flat["metadata"] = meta
                formatted.append(flat)
        return formatted

    async def execute_sql_flow(self, query_for_thinker: str, thinking_agent, sql_agent, context: Context) -> Dict[str, Any]:
        """
        Вызов ThinkingAgent.think для получения json_query и последующее исполнение через SQLAgent.
        Возвращает dict: {"sql": str, "rows": list, "mappings": dict, "raw_planner": str}
        НЕ выполняет SQL, если mappings содержит unknowns (SQLAgent сам вернёт mappings в этом случае).
        """
        # Prefer non-blocking call if available
        thinking_msg = await getattr(thinking_agent, "athink", thinking_agent.think)(query_for_thinker)
        meta = getattr(thinking_msg, "metadata", {}) or {}
        raw_planner = meta.get("raw_llm", "")
        json_q = meta.get("json_query")
        mappings = meta.get("mappings", {}) or {}

        # Safety: если json_q содержит UNKNOWN_ плейсхолдеры, но mappings пуст — добавим их
        try:
            import re as _re
            if json_q and not mappings.get("unknowns"):
                txt = json.dumps(json_q, ensure_ascii=False)
                found = list(dict.fromkeys(_re.findall(r'UNKNOWN_[A-Za-z0-9_]+', txt)))
                if found:
                    mappings = {**mappings, "unknowns": found}
        except Exception:
            pass

        # Если ThinkingAgent не сгенерировал json_query — возвращаем mappings/raw и позволяем верхнему слою запросить уточнение
        if not json_q:
            return {"sql": "", "rows": [], "mappings": mappings or {"error": "No json_query generated"}, "raw_planner": raw_planner}

        # Передаём json_query в SQLAgent.execute — он вернёт (sql, rows, mappings)
        sql, rows, exec_mappings = sql_agent.execute(json_q)

        # объединяем mappings (те, что Thinking дал + те, что вернул SQL builder/executor)
        combined_mappings = {}
        combined_mappings.update(mappings or {})
        combined_mappings.update(exec_mappings or {})

        return {"sql": sql, "rows": rows, "mappings": combined_mappings, "raw_planner": raw_planner}

class ExplanationAgent:
    """Агент для генерации инсайтов и объяснений"""

    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def generate_insights(self, result: AnalysisResult, context: Context) -> Dict[str, Any]:
        if not result.data:
            return {"insights": [], "recommendations": [], "raw_explainer": "", "evidence": []}
        try:
            rows = result.data
            # Найти поле для агрегации — сначала ищем "Подобъект" (разных вариантов), иначе первую строковую колонку
            target_cols = [k for k in rows[0].keys() if re.search(r'подобъект', k, re.I)]
            col = target_cols[0] if target_cols else None
            if col:
                from collections import Counter
                values = [r.get(col) for r in rows if r.get(col)]
                counts = Counter(values)
                top5 = counts.most_common(5)
                evidence = [{"value": v, "count": c, "sample_rows": [r for r in rows if r.get(col) == v][:3]} for v, c
                            in top5]
                insights = []
                if evidence:
                    top3_summary = ", ".join([f"{e['value']} ({e['count']})" for e in evidence[:3]])
                    insights.append(f"Наиболее частые значения по колонке '{col}': {top3_summary}")
                recommendations = []
                raw_explainer = f"Deterministic aggregation over column '{col}'"
                return {"insights": insights, "recommendations": recommendations, "raw_explainer": raw_explainer,
                        "evidence": evidence}

            # fallback - no deterministic evidence
            return {"insights": [f"Найдено {len(rows)} релевантных документов."], "recommendations": [], "raw_explainer": "", "evidence": []}

        except Exception as e:
            logger.error(f"Error generating insights deterministically: {e}")
            return {"insights": ["Ошибка при генерации инсайтов"], "recommendations": ["Проверьте данные"], "raw_explainer": str(e), "evidence": []}

class ValidationAgent:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def validate_results(self, result: AnalysisResult, context: Context) -> Dict[str, Any]:
        checks = []
        has_data = bool(result.data)
        checks.append({"check":"Has data","passed":has_data,"details":f"{len(result.data)} items"})
        avg_sim = None
        if has_data:
            sims = [d.get("similarity", 0) for d in result.data if isinstance(d.get("similarity", None), (int,float))]
            avg_sim = float(np.mean(sims)) if sims else 0.0
            checks.append({"check":"Avg similarity","passed":avg_sim>0.45,"details":f"{avg_sim:.3f}"})
        passed = all(c["passed"] for c in checks)
        confidence = 0.9 if passed else 0.5
        summary = "Validation passed" if passed else "Issues found"
        return {"is_valid": passed, "confidence": confidence, "summary": summary, "checks": checks}

class DynamicScenarioAgent:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def plan_scenario_analysis(self, query: str, context: Context):
        # keep previous logic
        return {"reasoning": f"Plan for scenario: {query}", "scenario_type":"user_driven", "elements":{}}

    async def execute_scenario_analysis(self, query, solution_plan, context):
        # placeholder: return empty AnalysisResult-like dict
        return AnalysisResult(query=query, route="scenario", answer=None, chroma_query=None, data=[], reasoning_steps=[], insights=[], recommendations=[], confidence_score=0.5, validation_results={"is_valid": False})

    async def generate_dynamic_scenarios(self, original_query: str, result: AnalysisResult, context: Context):
        # simple stub; real logic in production
        return {"baseline": {"description": original_query, "metrics": {}}, "scenarios": []}

class AdvancedDigitalTwin:
    def __init__(self):
        self.chroma_manager = ChromaDBManager()
        self.context_agent = ContextAgent(self.chroma_manager)
        self.reasoning_agent = ReasoningAgent(self.chroma_manager)
        self.data_agent = DataAgent(self.chroma_manager)
        self.validation_agent = ValidationAgent(self.chroma_manager)
        self.scenario_agent = DynamicScenarioAgent(self.chroma_manager)
        self.explanation_agent = ExplanationAgent(self.chroma_manager)
        # initialize SQL and Thinking agents (used for sql routing)
        self.sql_agent = SQLAgent(DB_PATH)
        self.thinking_agent = ThinkingAgent(DB_PATH, None)
        self.contexts: Dict[str, Context] = {}
        self.knowledge_base = self._load_knowledge_base()

    def _is_date_question(self, text: str) -> bool:
        q = (text or "").lower()
        return any(tok in q for tok in ["самая ран", "ранняя дата", "минимальная дата", "min date", "earliest", "дата"])

    def _suggest_date_columns(self) -> Dict[str, Any]:
        try:
            from sqlalchemy import create_engine, inspect
            engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
            insp = inspect(engine)
            tables = insp.get_table_names()
            candidates = []
            # Prefer table named 'data' if exists
            ordered = [t for t in tables if t == 'data'] + [t for t in tables if t != 'data']
            for t in ordered[:5]:
                cols = insp.get_columns(t)
                for c in cols:
                    name = c.get("name", "")
                    typ = str(c.get("type", "")).lower()
                    if any(k in name.lower() for k in ["date", "дата", "start", "начал", "end", "оконча", "finish"]) or "date" in typ:
                        candidates.append({"table": t, "column": name})
            return {"date_candidates": candidates}
        except Exception:
            return {"date_candidates": []}

    def _load_knowledge_base(self):
        try:
            if KNOWLEDGE_BASE_PATH.exists():
                return json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding='utf-8'))
        except Exception:
            logger.debug("KB load failed")
        return {"patterns": []}

    async def process_query(self, query: str, session_id: str = "default", query_type: str = QueryType.ANALYTICS) -> AnalysisResult:
        context = self.contexts.get(session_id, Context(session_id))
        reasoning_steps: List[ReasoningStep] = []
        raw_traces = {}

        # step1: strategy
        strategy_text = f"Query type: {query_type}"
        reasoning_steps.append(ReasoningStep(1.0, "Determine strategy", strategy_text, "Decide route (sql/chroma/hybrid/schema)", "Choose route", confidence=0.8))

        try:
            relevant_context = await self.context_agent.extract_relevant_context(query, context)
            reasoning_steps.append(ReasoningStep(2.0, "Extract context", f"Found {len(relevant_context)} items", "Collect relevant history", "Relevant context collected", confidence=0.8))

            sol = await self.reasoning_agent.create_solution_plan(query, context, relevant_context)
            plan = sol.get("plan", {})
            raw_planner = sol.get("raw", "")
            route = sol.get("route", "hybrid")
            raw_traces["planner"] = raw_planner or ""
            reasoning_steps.append(ReasoningStep(3.0, "Create solution plan", plan.get("reasoning", "") if isinstance(plan, dict) else str(plan), "LLM planning", "Plan created", confidence=0.8))

            # Execute according to route
            if route == "schema":
                # deterministic schema read: prefer table named 'data', otherwise first table
                cols = []
                tbl = None
                try:
                    import sqlite3
                    if DB_PATH.exists():
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        # get all tables
                        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        rows = [r[0] for r in cur.fetchall()]
                        if "data" in [r.lower() for r in rows]:
                            # find actual name with case preserved
                            for r in rows:
                                if r.lower() == "data":
                                    tbl = r
                                    break
                        elif rows:
                            tbl = rows[0]
                        if tbl:
                            cur.execute(f'PRAGMA table_info("{tbl}")')
                            cols = [r[1] for r in cur.fetchall()]
                        conn.close()
                    answer_text = f"Columns of table '{tbl}':\n" + "\n".join(
                        cols) if cols else "No schema found or table is empty."
                except Exception as e:
                    answer_text = f"Schema read error: {e}"
                final = AnalysisResult(query=query, route="schema", answer=answer_text, chroma_query=None, data=[],
                                       reasoning_steps=reasoning_steps, insights=[], recommendations=[],
                                       confidence_score=0.95, validation_results={"is_valid": True})
                final.raw_llm_traces = raw_traces
                # save context
                context.previous_queries.append(
                    {"query": query, "result": final.to_dict(), "timestamp": datetime.now().isoformat()})
                self.contexts[session_id] = context
                return final

            elif route == "sql":
                # attempt to create json_query via ThinkingAgent and execute via SQLAgent
                # use full query as prompt for ThinkingAgent
                # pass session preferences to thinker
                self.thinking_agent.set_session_preferences(self.contexts.get(session_id, Context(session_id)).preferences)
                sql_exec = await self.data_agent.execute_sql_flow(query, self.thinking_agent, self.sql_agent, context)
                raw_traces["executor_planner"] = sql_exec.get("raw_planner", "")
                if sql_exec.get("mappings", {}).get("unknowns"):
                    # need clarification — build user-friendly prompt with options
                    m = sql_exec["mappings"]
                    unknowns = m.get("unknowns", [])
                    suggestions = m.get("suggestions", {})
                    guesses = m.get("column_guesses", {})
                    question = "Требуются уточнения по: " + ", ".join(unknowns)
                    final = AnalysisResult(query=query, route="sql", answer=question, chroma_query=None, data=[], reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.0, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": unknowns, "suggestions": suggestions, "guesses": guesses})
                    final.raw_llm_traces = raw_traces
                    context.previous_queries.append({"query": query, "result": final.to_dict(), "timestamp": datetime.now().isoformat()})
                    self.contexts[session_id] = context
                    return final
                # success
                rows = sql_exec.get("rows", [])
                sql_text = sql_exec.get("sql", "")
                build_maps = sql_exec.get("mappings", {}) or {}
                if build_maps.get("error") and "Таблица" in build_maps.get("error", ""):
                    # ask to choose a table
                    try:
                        from sqlalchemy import create_engine, inspect
                        engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
                        insp = inspect(engine)
                        tbls = insp.get_table_names()
                    except Exception:
                        tbls = []
                    question = "Выберите таблицу для запроса"
                    final = AnalysisResult(query=query, route="sql", answer=question, chroma_query=None, data=[], reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.0, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": ["table"], "suggestions": {"table": tbls}, "guesses": {}})
                    final.raw_llm_traces = {"planner": sql_exec.get("raw_planner", ""), "sql": sql_text}
                    context.previous_queries.append({"query": query, "result": final.to_dict(), "timestamp": datetime.now().isoformat()})
                    self.contexts[session_id] = context
                    return final
                # If empty rows for date-like question — ask clarification for date column
                if (not rows) and self._is_date_question(query):
                    sugg = self._suggest_date_columns()
                    final = AnalysisResult(query=query, route="sql", answer="Требуется указать колонку даты (например, date/end_date)", chroma_query=None, data=[], reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.0, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": ["UNKNOWN_date"], "suggestions": {"UNKNOWN_date": [f"{x['table']}.{x['column']}" for x in sugg.get('date_candidates', [])]}, "guesses": {}})
                    final.raw_llm_traces = {"planner": sql_exec.get("raw_planner", ""), "sql": sql_text}
                    context.previous_queries.append({"query": query, "result": final.to_dict(), "timestamp": datetime.now().isoformat()})
                    self.contexts[session_id] = context
                    return final

                result = AnalysisResult(query=query, route="sql", chroma_query=None, data=rows,
                                        reasoning_steps=reasoning_steps, insights=[], recommendations=[],
                                        confidence_score=0.8, validation_results={"is_valid": True})
                # сохранить trace планировщика/исполнителя
                result.raw_llm_traces = {"planner": sql_exec.get("raw_planner", ""), "sql": sql_text}
                # Если rows содержит ровно 1 строку и 1 колонку — делаем короткий ответ
                if rows and len(rows) == 1 and isinstance(rows[0], dict) and len(rows[0]) == 1:
                    result.answer = str(list(rows[0].values())[0])
                else:
                    # Если есть агрегатные поля (cnt / count / max / min) — используем их
                    if rows and isinstance(rows[0], dict):
                        # ищем count-like или max-like полей
                        cnt_key = next((k for k in rows[0].keys() if 'cnt' in k.lower() or 'count' in k.lower()), None)
                        max_key = next(
                            (k for k in rows[0].keys() if k.lower().startswith('max_') or 'max' in k.lower()), None)
                        if cnt_key:
                            # строим evidence и answer
                            sorted_rows = sorted(rows, key=lambda r: (r.get(cnt_key) or 0), reverse=True)
                            top = sorted_rows[0]
                            val = {k: v for k, v in top.items() if k != cnt_key}
                            cnt = int(top.get(cnt_key) or 0)
                            result.evidence = [{"value": val, "count": cnt, "row": top}]
                            # short answer
                            if isinstance(val, dict):
                                first_val = list(val.values())[0] if val else None
                                if first_val is not None:
                                    result.answer = f"{first_val} ({cnt} вхождений)"
                            else:
                                result.answer = f"{val} ({cnt} вхождений)"
                        elif max_key:
                            # аналогично для max_
                            top = rows[0]
                            result.answer = str(top.get(max_key))
                # If still no answer, try to derive from insights (explanation agent)
                # generate explanation FIRST so explanation can operate on result.data (if needed)
                expl = await self.explanation_agent.generate_insights(result, context)
                result.insights = expl.get("insights", [])
                result.recommendations = expl.get("recommendations", [])
                result.raw_llm_traces["explainer"] = expl.get("raw_explainer", "")
                result.evidence = expl.get("evidence", [])
                if not result.answer and result.insights:
                    result.answer = result.insights[0]
                # If still nothing, provide friendly fallback
                if not result.answer:
                    if rows and isinstance(rows, list):
                        result.answer = f"Найдено {len(rows)} записей. Смотрите детали и доказательства."
                    else:
                        result.answer = "Нет данных."
                # persist
                context.previous_queries.append({"query": query, "result": result.to_dict(), "timestamp": datetime.now().isoformat()})
                self.contexts[session_id] = context
                await self._save_query_to_chroma(query, result)
                return result

            elif route == "chroma":
                chroma_q = await self.data_agent.generate_chroma_query(query, plan, context)
                raw_traces["generated_chroma_query"] = chroma_q
                data = await self.data_agent.execute_chroma_query(chroma_q)
                res = AnalysisResult(query=query, route="chroma", answer=None, chroma_query=chroma_q, data=data, reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.8, validation_results={"is_valid": True})
                res.raw_llm_traces = raw_traces
                expl = await self.explanation_agent.generate_insights(res, context)
                res.insights = expl.get("insights", [])
                res.recommendations = expl.get("recommendations", [])
                res.raw_llm_traces["explainer"] = expl.get("raw_explainer", "")
                # assign evidence into res (was buggy: previously assigned to result)
                res.evidence = expl.get("evidence", [])
                # concise answer from insights or fallback
                if res.insights:
                    res.answer = res.insights[0]
                elif data:
                    res.answer = f"Найдено {len(data)} документов. Смотрите детали."
                else:
                    res.answer = "Нет релевантных документов."
                context.previous_queries.append({"query": query, "result": res.to_dict(), "timestamp": datetime.now().isoformat()})
                self.contexts[session_id] = context
                await self._save_query_to_chroma(query, res)
                return res

            else:  # hybrid
                # Step A: semantic search
                chroma_q = await self.data_agent.generate_chroma_query(query, plan, context)
                raw_traces["generated_chroma_query"] = chroma_q
                chroma_data = await self.data_agent.execute_chroma_query(chroma_q)
                # Step B: try to synthesize SQL using top doc snippets as context
                top_snips = " ".join([d.get("content","")[:800] for d in chroma_data[:5]])
                enriched_prompt = f"Context snippets: {top_snips}\nUser question: {query}\nCreate JSON SQL-description as before."
                self.thinking_agent.set_session_preferences(self.contexts.get(session_id, Context(session_id)).preferences)
                sql_exec = await self.data_agent.execute_sql_flow(enriched_prompt, self.thinking_agent, self.sql_agent, context)
                raw_traces["hybrid_planner_raw"] = sql_exec.get("raw_planner","")
                # If SQL produced usable rows, merge
                if sql_exec.get("rows"):
                    rows = sql_exec.get("rows")
                    # if chroma has no docs, return SQL rows as primary data
                    merged = chroma_data if chroma_data else rows
                    final = AnalysisResult(query=query, route="hybrid", answer=None, chroma_query=chroma_q, data=merged, reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.75, validation_results={"is_valid": True})
                    final.raw_llm_traces = raw_traces
                    expl = await self.explanation_agent.generate_insights(final, context)
                    final.insights = expl.get("insights", [])
                    final.recommendations = expl.get("recommendations", [])
                    final.raw_llm_traces["explainer"] = expl.get("raw_explainer","")
                    final.evidence = expl.get("evidence", [])
                    final.answer = final.insights[0] if final.insights else (
                        f"Найдено {len(chroma_data)} документов; SQL вернул {len(rows)} строк." if chroma_data else f"SQL вернул {len(rows)} строк."
                    )
                    context.previous_queries.append({"query": query, "result": final.to_dict(), "timestamp": datetime.now().isoformat()})
                    self.contexts[session_id] = context
                    await self._save_query_to_chroma(query, final)
                    return final
                else:
                    # If need clarification — return prompt to user; else fallback chroma-only
                    if sql_exec.get("mappings", {}).get("unknowns"):
                        m = sql_exec["mappings"]
                        unknowns = m.get("unknowns", [])
                        suggestions = m.get("suggestions", {})
                        guesses = m.get("column_guesses", {})
                        question = "Требуются уточнения по: " + ", ".join(unknowns)
                        final = AnalysisResult(query=query, route="hybrid", answer=question, chroma_query=chroma_q, data=chroma_data, reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.5, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": unknowns, "suggestions": suggestions, "guesses": guesses})
                    elif self._is_date_question(query):
                        # Ask for date column if date-like question and no SQL rows
                        sugg = self._suggest_date_columns()
                        final = AnalysisResult(query=query, route="hybrid", answer="Требуется указать колонку даты (например, date/end_date)", chroma_query=chroma_q, data=chroma_data, reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.5, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": ["UNKNOWN_date"], "suggestions": {"UNKNOWN_date": [f"{x['table']}.{x['column']}" for x in sugg.get('date_candidates', [])]}, "guesses": {}})
                    else:
                    final = AnalysisResult(query=query, route="hybrid", answer=None, chroma_query=chroma_q, data=chroma_data, reasoning_steps=reasoning_steps, insights=[], recommendations=[], confidence_score=0.6, validation_results={"is_valid": True})
                    final.raw_llm_traces = raw_traces
                    expl = await self.explanation_agent.generate_insights(final, context)
                    final.insights = expl.get("insights", [])
                    final.recommendations = expl.get("recommendations", [])
                    final.raw_llm_traces["explainer"] = expl.get("raw_explainer","")
                    final.evidence = expl.get("evidence", [])
                    if final.validation_results.get("needs_clarification"):
                        # keep clarification question as answer
                        pass
                    else:
                    final.answer = final.insights[0] if final.insights else f"Найдено {len(chroma_data)} документов."
                    context.previous_queries.append({"query": query, "result": final.to_dict(), "timestamp": datetime.now().isoformat()})
                    self.contexts[session_id] = context
                    await self._save_query_to_chroma(query, final)
                    return final

        except Exception as e:
            logger.exception("process_query error")
            return AnalysisResult(query=query, route="error", answer=None, chroma_query=None, data=[], reasoning_steps=reasoning_steps, insights=[f"Error: {e}"], recommendations=[], confidence_score=0.0, validation_results={"is_valid": False, "error": str(e)})

    async def _save_query_to_chroma(self, query: str, result: AnalysisResult):
        try:
            doc = f"Query: {query}\nInsights: {', '.join(result.insights[:3])}\nData count: {len(result.data)}"
            meta = {"query": query, "timestamp": datetime.now().isoformat(), "confidence": result.confidence_score, "data_count": len(result.data)}
            doc_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(query))%10000}"
            self.chroma_manager.add_documents(documents=[doc], metadatas=[meta], ids=[doc_id], collection_name='queries')
        except Exception as e:
            logger.debug("save query to chroma failed: %s", e)

    async def _save_clarification_to_chroma(self, session_id: str, clarification: str):
        try:
            doc = f"Clarification: {clarification}\nSession: {session_id}"
            meta = {"type": "clarification", "session_id": session_id, "timestamp": datetime.now().isoformat()}
            doc_id = f"c_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(session_id+clarification))%10000}"
            self.chroma_manager.add_documents(documents=[doc], metadatas=[meta], ids=[doc_id], collection_name='queries')
        except Exception as e:
            logger.debug("save clarification to chroma failed: %s", e)

    async def clarify(self, session_id: str, clarification: str) -> AnalysisResult:
        """Применяет уточнение пользователя к последнему json_query и выполняет SQL маршрут."""
        context = self.contexts.get(session_id) or Context(session_id)
        # применяем уточнение в мыслителе
        msg = self.thinking_agent.apply_user_clarification(clarification)
        meta = getattr(msg, "metadata", {}) or {}
        json_q = meta.get("json_query")
        mappings = meta.get("mappings", {}) or {}
        # Сохраняем простые предпочтения (если строка вида KEY=ColumnName и колонка существует)
        # Позже можно расширить на хранение в Chroma
        if "=" in clarification:
            k, v = map(str.strip, clarification.split("=", 1))
            if v:
                context.preferences[k] = v
        self.contexts[session_id] = context
        # persist clarification into Chroma for future retrieval
        await self._save_clarification_to_chroma(session_id, clarification)
        # Если unknowns ещё остались — вернём просьбу уточнить дальше
        if mappings.get("unknowns"):
            return AnalysisResult(query="[clarify]", route="sql", answer="Требуются дополнительные уточнения: " + ", ".join(mappings.get("unknowns", [])), chroma_query=None, data=[], reasoning_steps=[], insights=[], recommendations=[], confidence_score=0.0, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": mappings.get("unknowns", [])})
        # Иначе выполним SQL
        sql, rows, exec_mappings = self.sql_agent.execute(json_q or {})
        if exec_mappings.get("unknowns"):
            return AnalysisResult(query="[clarify]", route="sql", answer="Требуются дополнительные уточнения: " + ", ".join(exec_mappings.get("unknowns", [])), chroma_query=None, data=[], reasoning_steps=[], insights=[], recommendations=[], confidence_score=0.0, validation_results={"is_valid": False, "needs_clarification": True, "unknowns": exec_mappings.get("unknowns", [])})
        res = AnalysisResult(query="[clarify]", route="sql", answer=None, chroma_query=None, data=rows, reasoning_steps=[], insights=[], recommendations=[], confidence_score=0.8, validation_results={"is_valid": True})
        if rows and len(rows) == 1 and isinstance(rows[0], dict) and len(rows[0]) == 1:
            res.answer = str(list(rows[0].values())[0])
        return res

class ContextAgent:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager

    async def extract_relevant_context(self, query: str, context: Context):
        relevant = []
        try:
            res = self.chroma_manager.query_documents(query_text=query, n_results=5, collection_name='queries')
            if res and res.get('documents') and res['documents'][0]:
                for i, (doc, meta) in enumerate(zip(res['documents'][0], res['metadatas'][0])):
                    score = 1.0 - res['distances'][0][i] if res['distances'][0] else 0.5
                    relevant.append({"query": meta.get("query",""), "relevance_score": score, "insights": meta.get("insights", [])})
        except Exception as e:
            logger.debug("context extract error: %s", e)
        # also include recent context
        for prev in context.previous_queries[-5:]:
            if any(tok in prev.get("query","").lower() for tok in query.lower().split()[:3]):
                relevant.append({"query": prev.get("query",""), "relevance_score": 0.3})
        return sorted(relevant, key=lambda x: x["relevance_score"], reverse=True)