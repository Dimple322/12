from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import json
import re
import difflib

from sqlalchemy import create_engine, inspect, text
import ollama

from agent.utils.knowledge_base import load_knowledge_base, save_knowledge_base_entry
from agent.utils.sql_builder import build_sql_from_json
from agent.sql_agent import SQLAgent
from agent.utils.logger import get_logger

logger = get_logger(__name__)
LLM_MODEL = "digital_twin_analyst"

@dataclass
class Message:
    role: str
    content: str
    metadata: Optional[dict] = None

class ChatAgent:
    def __init__(self, db_path: Path, kb_path: Path):
        self.db_path = db_path
        self.kb_path = kb_path
        self.engine  = create_engine(f"sqlite:///{db_path}", future=True)
        self.history: List[Message] = []
        self.kb      = load_knowledge_base()
        self._last_candidates: Dict[str, List[tuple[str, float]]] = {}

    def send_message(self, user_text: str) -> Message:
        self.history.append(Message("user", user_text))
        # 1) KB lookup
        kb_json = self._kb_lookup(user_text)
        if kb_json:
            return self._execute_json(kb_json, source="KB")
        # 2) entity extraction & fuzzy search
        entities = self._extract_entities(user_text)
        candidates = self._search_candidates(entities)
        ambiguous = {k: v for k, v in candidates.items() if len(v) > 1}
        if ambiguous:
            return self._ask_clarification(ambiguous, entities)
        chosen = {k: v[0][0] for k, v in candidates.items() if v}
        json_query = self._build_json_from_chosen(chosen)
        return self._execute_json(json_query, source="LLM", chosen=chosen, entities=entities)

    def apply_user_choice(self, table_col: str, chosen_value: str) -> Message:
        if not self._last_candidates or table_col not in self._last_candidates:
            return Message("system", "Нет кандидатов для выбора.")
        chosen = {table_col: chosen_value}
        for k, v in self._last_candidates.items():
            if k != table_col and v:
                chosen.setdefault(k, v[0][0])
        json_query = self._build_json_from_chosen(chosen)
        return self._execute_json(json_query, source="USER", chosen=chosen)

    def _execute_json(self, json_query: dict, source: str, **meta_kwargs) -> Message:
        """
        Используем SQLAgent.execute (sql, rows, mappings).
        Если mappings содержит unknowns — возвращаем сообщение с просьбой уточнить.
        """
        try:
            sql_agent = SQLAgent(self.db_path)
            sql, rows, mappings = sql_agent.execute(json_query)
        except Exception as e:
            err = f"Ошибка при подготовке/выполнении запроса: {e}"
            meta = {"error": str(e), "json_query": json_query, "source": source}
            msg = Message("system", err, metadata=meta)
            self.history.append(msg)
            return msg

        if mappings and mappings.get("unknowns"):
            content_lines = ["Мне нужно уточнить следующие поля:"]
            for unk in mappings["unknowns"]:
                content_lines.append(f" - {unk}")
            content_lines.append("Пожалуйста, уточните (введите значение или выберите из списка).")
            meta = {"json_query": json_query, "mappings": mappings, "source": source, **meta_kwargs}
            msg = Message("assistant", "\n".join(content_lines), metadata=meta)
            self.history.append(msg)
            return msg

        if mappings and mappings.get("error"):
            meta = {"error": mappings.get("error"), "json_query": json_query, "source": source}
            msg = Message("system", f"Ошибка: {mappings.get('error')}", metadata=meta)
            self.history.append(msg)
            return msg

        answer = self._format_answer(rows)
        meta = {"json_query": json_query, "sql": sql, "rows": rows, "source": source, **meta_kwargs}
        msg = Message("assistant", answer, metadata=meta)
        self.history.append(msg)
        return msg

    # остальные методы без изменений
    def _extract_entities(self, text: str) -> List[str]:
        objs = re.findall(r'["«»](.+?)["«»]', text)
        if not objs:
            objs = re.findall(r"(?:объект|участок|станция|шифр)[а-я]*\s+([а-я0-9\s\.]+)", text, re.I)
        return [o.strip() for o in objs]

    def _search_candidates(self, entities):
        res = {}
        if not entities:
            return res
        inspector = inspect(self.engine)
        for table in inspector.get_table_names():
            for col in (c["name"] for c in inspector.get_columns(table)):
                if not self._is_text_column(table, col):
                    continue
                for ent in entities:
                    matches = self._fuzzy_column_values(table, col, ent)
                    if matches:
                        key = f"{table}.{col}"
                        res.setdefault(key, []).extend(matches)
        for k, v in res.items():
            seen = set()
            uniq = []
            for val, sc in sorted(v, key=lambda x: x[1], reverse=True):
                if val not in seen:
                    seen.add(val)
                    uniq.append((val, sc))
                    if len(uniq) == 3:
                        break
            res[k] = uniq
        self._last_candidates = res
        return res

    def _fuzzy_column_values(self, table, col, user_text, limit=3):
        with self.engine.begin() as conn:
            rows = conn.execute(text(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL')).fetchall()
            unique = [r[0] for r in rows if r[0]]
        import difflib
        matches = difflib.get_close_matches(user_text, unique, n=limit, cutoff=0.5)
        return [(m, difflib.SequenceMatcher(None, user_text, m).ratio()) for m in matches]

    def _is_text_column(self, table, col):
        with self.engine.begin() as conn:
            for row in conn.execute(text(f'PRAGMA table_info("{table}")')).fetchall():
                if row[1] == col and "TEXT" in str(row[2]).upper():
                    return True
        return False

    def _ask_clarification(self, ambiguous, entities):
        lines = ["Я нашёл несколько похожих значений:"]
        for table_col, vals in ambiguous.items():
            table, col = table_col.split(".", 1)
            lines.append(f"Колонка «{col}»:")
            for v, sc in vals:
                lines.append(f"  – {v}  (score={sc:.2f})")
        lines.append("Выберите нужный вариант (нажмите кнопку ниже).")
        content = "\n".join(lines)
        meta = {"ambiguous": ambiguous, "entities": entities}
        msg = Message("assistant", content, metadata=meta)
        self.history.append(msg)
        return msg

    def _build_json_from_chosen(self, chosen):
        table = list(self._get_schema().keys())[0]
        where = []
        for table_col, val in chosen.items():
            _, col = table_col.split(".", 1)
            where.append({"column": col, "operator": "=", "value": val})
        return {"select": ["*"], "from": table, "where": where}

    def _kb_lookup(self, question: str) -> Optional[dict]:
        for ex in self.kb:
            if question.lower() in ex["question"].lower() or ex["question"].lower() in question.lower():
                jq = ex["json_query"]
                return json.loads(jq) if isinstance(jq, str) else jq
        return None

    def _get_schema(self):
        inspector = inspect(self.engine)
        return {t: [c["name"] for c in inspector.get_columns(t)] for t in inspector.get_table_names()}

    def _run_sql(self, sql: str) -> List[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql)).fetchall()
            return [dict(r._mapping) for r in rows]

    def _format_answer(self, rows: List[dict]) -> str:
        if not rows:
            return "Ничего не найдено."
        if len(rows) == 1 and len(rows[0]) == 1:
            val = list(rows[0].values())[0]
            return f"Ответ: {val}"
        return f"Найдено записей: {len(rows)}. Первые 3: {rows[:3]}"