from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import re
import difflib

import ollama
from sqlalchemy import create_engine, inspect, text

from agent.utils.knowledge_base import load_knowledge_base
from agent.utils.logger import get_logger

logger = get_logger(__name__)
LLM_MODEL = "digital_twin_analyst"

@dataclass
class Message:
    role: str
    content: str
    metadata: Optional[dict] = None

class ThinkingAgent:
    """
    Агент-мыслитель: думает вслух, формирует/нормализует JSON-запрос, управляет уточнениями.
    НЕ выполняет SQL: возвращает json_query + mappings; AnalyticAgent запускает исполнение.
    """
    def __init__(self, db_path: Path, kb_path: Optional[Path] = None):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        self.history: List[Message] = []
        self.kb = load_knowledge_base()
        self._last_json: Optional[dict] = None
        self._last_mappings: Optional[dict] = None
        self._last_raw: Optional[str] = None

    def think(self, user_text: str) -> Message:
        """Думает вслух и возвращает assistant Message с metadata: raw_llm, json_query (если есть), mappings (если unknowns)."""
        logger.info("ThinkingAgent.think: %s", user_text)
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        table_ctx = tables[0] if tables else None
        cols = [c["name"] for c in inspector.get_columns(table_ctx)] if table_ctx else []
        schema_text = f"Таблица: {table_ctx}. Колонки: {', '.join(cols)}."

        examples = ""
        try:
            examples_list = list(reversed(self.kb))[:3]
            examples = "\n".join([f"Q: {e['question']}\nJSON: {e['json_query']}" for e in examples_list])
        except Exception:
            examples = ""

        prompt = f"""
Ты — аналитик. Сначала подумай вслух, затем ВЕРНИТЕ ТОЛЬКО JSON-ОПИСАНИЕ SQL-ЗАПРОСА в строгом формате.
Схема: {schema_text}
Примеры:
{examples}

Требуемый формат (строго):
{{"select":[...],"from":"table","where":[{{"column":"...","operator":"...","value":"..."}}],"group_by":[...],"order_by":[{{"column":"...","direction":"asc|desc"}}],"limit":N}}
Если не можешь сопоставить колонку — используй "UNKNOWN_<name>".
ДАЙ СВОИ РАЗМЫШЛЕНИЯ (thinking) и В КОНЦЕ ТОЛЬКО JSON (без пояснений).
Вопрос: {user_text}
"""
        try:
            resp = ollama.chat(model=LLM_MODEL, messages=[
                {"role": "system", "content": "Ты — парсер вопросов. После короткого reasoning в конце верни JSON."},
                {"role": "user", "content": prompt}
            ])
            raw = resp["message"]["content"]
        except Exception as e:
            logger.exception("LLM error")
            msg = Message("assistant", f"[LLM ERROR] {e}", metadata={"error": str(e)})
            self.history.append(msg)
            return msg

        logger.debug("ThinkingAgent raw LLM: %s", raw)
        self._last_raw = raw

        parsed = self._extract_task_json(raw)
        mappings: Dict[str, Any] = {}
        if parsed is None:
            msg = Message("assistant", "Я не смог сформировать однозначный JSON‑запрос. Уточните, пожалуйста детали.", metadata={"raw_llm": raw})
            self.history.append(msg)
            return msg

        normalized = self._normalize_query(parsed)
        unknowns = self._find_unknowns_in_json(normalized)

        # Compute suggestions/explanations for unknowns
        suggestions, col_guesses, explanations = self._compute_suggestions_for_unknowns(unknowns, table_ctx)

        # detect higher-level computation intent (e.g. diff = MAX - MIN) in raw LLM text
        comp_hint = self._detect_computation_intent(raw, unknowns)
        if comp_hint:
            # attach computation hint details (type, description)
            mappings = {
                "unknowns": unknowns,
                "suggestions": suggestions,
                "column_guesses": col_guesses,
                "explanations": explanations,
                "computation_hint": comp_hint
            }
        else:
            mappings = {
                "unknowns": unknowns,
                "suggestions": suggestions,
                "column_guesses": col_guesses,
                "explanations": explanations
            }

        self._last_json = normalized
        self._last_mappings = mappings

        content = "Я сформировал JSON-запрос." + ((" Требуются уточнения: " + ", ".join(unknowns)) if unknowns else "")
        meta = {"raw_llm": raw, "json_query": normalized, "mappings": mappings}
        msg = Message("assistant", content, metadata=meta)
        self.history.append(msg)
        return msg

    def apply_user_clarification(self, clarification: str, json_query: Optional[dict] = None) -> Message:
        """
        Применяет уточнение к последнему json (или к переданному) и возвращает assistant Message.
        Поддерживает:
         - UNKNOWN_X=Value  (Value может быть: значение из колонки -> treated as value)
         - UNKNOWN_X=ColumnName  (if RHS matches a column name -> treat as mapping to column)
         - просто "Value"  (assign to first unknown as value)
        Если в last_mappings есть computation_hint, и пользователь маппит unknown -> column,
        автоматом подставим вычисляемое выражение (например MAX/MIN diff).
        """
        logger.info("ThinkingAgent.apply_user_clarification: %s", clarification)
        self.history.append(Message("user", clarification))

        j = json_query or self._last_json
        if not j:
            return Message("system", "Не найден JSON-запрос для уточнения.")

        last_mappings = self._last_mappings or {}
        unknowns = last_mappings.get("unknowns", [])

        clar = clarification.strip()
        inspector = inspect(self.engine)
        table_ctx = inspector.get_table_names()[0] if inspector.get_table_names() else None
        table_cols = [c["name"] for c in inspector.get_columns(table_ctx)] if table_ctx else []

        # parse "KEY=VALUE" or plain value
        if "=" in clar:
            k, v = map(str.strip, clar.split("=", 1))
            target = None
            for unk in unknowns:
                if k.lower() in unk.lower() or unk.lower().endswith(k.lower()) or unk == k:
                    target = unk
                    break
            if target is None and unknowns:
                target = unknowns[0]

            # determine if v matches a column name
            matched_col = None
            for col in table_cols:
                if col.lower() == v.lower():
                    matched_col = col
                    break

            if matched_col:
                # If computation hint present and suggests computation, build expression and replace placeholder
                comp_hint = last_mappings.get("computation_hint")
                if comp_hint:
                    expr = self._build_computation_expression(comp_hint, matched_col)
                    # Replace placeholder in select if present, otherwise append
                    new_select = []
                    replaced_any = False
                    for expr_item in j.get("select", []):
                        if isinstance(expr_item, str) and target in expr_item:
                            new_select.append(expr.replace(target, expr))  # replace placeholder with expr (expr has quoted column)
                            replaced_any = True
                        else:
                            new_select.append(expr_item)
                    if not replaced_any:
                        # Replace placeholder tokens anywhere in select
                        sel = []
                        for s in j.get("select", []):
                            if isinstance(s, str) and target in s:
                                sel.append(expr)
                                replaced_any = True
                            else:
                                sel.append(s)
                        j["select"] = sel if sel else j.get("select", []) + [expr]
                    else:
                        j["select"] = new_select
                    # Also, if where has target placeholder as column, replace with matched_col
                    for cond in j.get("where", []):
                        if cond.get("column") == target:
                            cond["column"] = matched_col
                    # Clear unknowns (we constructed computation)
                    j.setdefault("where", j.get("where", []))
                else:
                    # No computation hint -> treat as column mapping (replace placeholder occurrences)
                    for cond in j.get("where", []):
                        if cond.get("column") == target:
                            cond["column"] = matched_col
                    new_select = []
                    for expr_item in j.get("select", []):
                        if isinstance(expr_item, str) and target in expr_item:
                            new_select.append(expr_item.replace(target, matched_col))
                        else:
                            new_select.append(expr_item)
                    j["select"] = new_select
            else:
                # Treat RHS as value
                replaced = False
                for cond in j.setdefault("where", []):
                    if isinstance(cond.get("column"), str) and cond["column"] == target:
                        cond["value"] = v
                        replaced = True
                        break
                if not replaced:
                    guessed_col = target.replace("UNKNOWN_", "")
                    j.setdefault("where", []).append({"column": guessed_col, "operator": "=", "value": v})
        else:
            # plain value -> assign to first unknown in where or append
            if unknowns:
                first_unknown = unknowns[0]
                replaced = False
                for cond in j.setdefault("where", []):
                    if cond.get("column") == first_unknown:
                        cond["value"] = clar
                        replaced = True
                        break
                if not replaced:
                    guessed_col = first_unknown.replace("UNKNOWN_", "")
                    j.setdefault("where", []).append({"column": guessed_col, "operator": "=", "value": clar})
            else:
                # treat as new question
                return self.think(clar)

        # After applying, recompute unknowns and suggestions
        unknowns = self._find_unknowns_in_json(j)
        suggestions, col_guesses, explanations = self._compute_suggestions_for_unknowns(unknowns, table_ctx)
        mappings = {
            "unknowns": unknowns,
            "suggestions": suggestions,
            "column_guesses": col_guesses,
            "explanations": explanations
        } if unknowns else {}

        self._last_json = j
        self._last_mappings = mappings

        content = "Уточнение принято." + ((" Всё ещё требуются уточнения: " + ", ".join(unknowns)) if unknowns else " Готов передать на выполнение.")
        msg = Message("assistant", content, metadata={"json_query": j, "mappings": mappings})
        self.history.append(msg)
        return msg

    # === helpers ===
    def _extract_task_json(self, think_text: str) -> Optional[dict]:
        if not think_text:
            return None
        m = re.search(r'```json\s*(.*?)\s*```', think_text, re.DOTALL | re.IGNORECASE)
        candidate = None
        if m:
            candidate = m.group(1).strip()
        else:
            start = think_text.find('{')
            if start == -1:
                return None
            depth = 0
            end = -1
            for i in range(start, len(think_text)):
                if think_text[i] == '{': depth += 1
                elif think_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                candidate = think_text[start:end+1]
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except Exception:
            logger.debug("Failed parse candidate JSON: %s", candidate)
            return None

    def _normalize_query(self, q: dict) -> dict:
        if "query" in q and isinstance(q["query"], dict):
            q = q["query"]
        new = {}
        for k, v in q.items():
            new[k.strip().lower().replace(" ", "_")] = v
        sel = new.get("select", ["*"])
        if isinstance(sel, str):
            sel = [sel]
        new["select"] = sel
        frm = new.get("from")
        if isinstance(frm, list) and frm:
            new["from"] = frm[0]
        new.setdefault("where", [])
        return new

    def _find_unknowns_in_json(self, q: dict) -> List[str]:
        unknowns = []
        if not isinstance(q, dict):
            return unknowns
        for expr in q.get("select", []) or []:
            if isinstance(expr, str):
                unknowns += re.findall(r'UNKNOWN_[A-Za-z0-9_]+', expr)
        for cond in q.get("where", []) or []:
            col = cond.get("column")
            if isinstance(col, str) and col.startswith("UNKNOWN_"):
                unknowns.append(col)
        return list(dict.fromkeys(unknowns))

    def _compute_suggestions_for_unknowns(self, unknowns: List[str], table: Optional[str]) -> (Dict[str, List[str]], Dict[str, Optional[str]], Dict[str, str]):
        suggestions: Dict[str, List[str]] = {}
        col_guesses: Dict[str, Optional[str]] = {}
        explanations: Dict[str, str] = {}

        inspector = inspect(self.engine)
        table_ctx = table or (inspector.get_table_names()[0] if inspector.get_table_names() else None)

        real_cols = {}
        if table_ctx:
            for c in inspector.get_columns(table_ctx):
                real_cols[c["name"].lower()] = c["name"]

        for unk in unknowns:
            guessed = unk.replace("UNKNOWN_", "")
            mapped = None
            if real_cols:
                if guessed.lower() in real_cols:
                    mapped = real_cols[guessed.lower()]
                else:
                    possible = difflib.get_close_matches(guessed, list(real_cols.values()), n=1, cutoff=0.6)
                    if possible:
                        mapped = possible[0]
                    else:
                        possible2 = difflib.get_close_matches(guessed.lower(), list(real_cols.keys()), n=1, cutoff=0.6)
                        if possible2:
                            mapped = real_cols[possible2[0]]
            col_guesses[unk] = mapped

            if mapped:
                if re.search(r'дат|date|day|дата', mapped, re.I):
                    explanations[unk] = f"Вероятно столбец даты: '{mapped}'. Можно указать конкретную дату или использовать агрегаты (MAX/MIN) для вычислений."
                else:
                    explanations[unk] = f"Возможно колонка: '{mapped}'. Можно выбрать значение из предложенных или ввести своё."
            else:
                explanations[unk] = f"Не удалось сопоставить с именем колонки автоматически."

            vals = []
            try:
                if mapped and table_ctx:
                    with self.engine.begin() as conn:
                        q = text(f'SELECT DISTINCT "{mapped}" FROM "{table_ctx}" WHERE "{mapped}" IS NOT NULL LIMIT 20')
                        rows = conn.execute(q).fetchall()
                        vals = [r[0] for r in rows if r[0] is not None]
                else:
                    if table_ctx:
                        for c in inspector.get_columns(table_ctx):
                            col_name = c["name"]
                            try:
                                with self.engine.begin() as conn:
                                    q = text(f'SELECT DISTINCT "{col_name}" FROM "{table_ctx}" WHERE "{col_name}" IS NOT NULL LIMIT 50')
                                    rows = conn.execute(q).fetchall()
                                    for r in rows:
                                        v = r[0]
                                        if v and isinstance(v, str) and guessed.lower() in v.lower():
                                            vals.append(v)
                                            if len(vals) >= 10:
                                                break
                            except Exception:
                                continue
            except Exception:
                vals = []

            seen = set()
            uniq_vals = []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    uniq_vals.append(v)
                if len(uniq_vals) >= 10:
                    break

            suggestions[unk] = uniq_vals

        return suggestions, col_guesses, explanations

    def _detect_computation_intent(self, raw_text: str, unknowns: List[str]) -> Optional[dict]:
        """
        На основе raw LLM reasoning пытаемся понять, хотел ли агент выполнить вычисление:
        возвращаем hint: {'type':'maxmin','description':..., 'template':...} или None.
        """
        if not raw_text:
            return None
        rt = raw_text.lower()
        # Detect difference / range intent
        if any(w in rt for w in ["разниц", "разница", "difference", "diff", "max and min", "max/min", "max и min", "max и min", "min and max", "julianday"]):
            # suggest max-min diff
            return {"type": "maxmin", "description": "Вычислить разницу между MAX и MIN (в днях через julianday) или просто MAX-MIN", "unit": "days"}
        # detect max only intent
        if any(w in rt for w in ["самая поздняя", "максимальная дата", "max(", "наибольшая дата", "максимум"]):
            return {"type": "max", "description": "Вычислить MAX(column)", "unit": None}
        # detect min only
        if any(w in rt for w in ["минимальная дата", "min(", "наименьшая дата"]):
            return {"type": "min", "description": "Вычислить MIN(column)", "unit": None}
        # detect count
        if any(w in rt for w in ["сколько", "count(", "количество"]):
            return {"type": "count", "description": "Вычислить COUNT(*) или COUNT(column)", "unit": None}
        return None

    def _build_computation_expression(self, comp_hint: dict, column_name: str) -> str:
        """
        Строим SQL-expression на основе hint и имени колонки.
        Всегда экранируем имя колонки двойными кавычками.
        """
        col = column_name
        c = comp_hint.get("type")
        if c == "maxmin":
            # вычисляем разницу в днях через julianday
            return f'julianday(MAX("{col}")) - julianday(MIN("{col}")) AS Diff'
        if c == "max":
            return f'MAX("{col}") AS max_{col}'
        if c == "min":
            return f'MIN("{col}") AS min_{col}'
        if c == "count":
            return f'COUNT("{col}") AS cnt_{col}'
        # default - return column
        return f'"{col}"'

    def _format_kb_examples(self) -> str:
        try:
            return "\n".join([f"Q: {ex['question']}\nA: {ex['json_query']}" for ex in self.kb[-5:]])
        except Exception:
            return ""