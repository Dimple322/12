import sys
import json
import re
from pathlib import Path
from sqlalchemy import create_engine, text
from agent.llm.ollama_client import AnalystLLM
import ollama
from agent.utils.logger import get_logger

logger = get_logger(__name__)

current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DB_PATH = project_root / "generated" / "digital_twin.db"
LLM = AnalystLLM()

from agent.utils.sql_builder import build_sql_from_json
from agent.utils.knowledge_base import load_knowledge_base

def _extract_json_from_llm_raw(raw: str) -> str:
    m = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: try to find first {...}
    start = raw.find('{')
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            return raw[start:end+1].strip()
    return raw.strip()

def ask_ru(question: str):
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

    with engine.begin() as conn:
        tables = [r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()]
    if not tables:
        return {"type":"error","content":"В БД нет таблиц."}

    table_name = tables[0]
    with engine.begin() as conn:
        cols = [r[1] for r in conn.execute(text(f"PRAGMA table_info({table_name});")).fetchall()]

    kb = load_knowledge_base()
    examples_text = "\n".join([f"Вопрос: {e['question']}\nJSON: {e['json_query']}" for e in list(reversed(kb))[:3]])
    schema_desc = f"Таблица: {table_name}, колонки: {', '.join(cols)}."
    prompt = f"""{examples_text}
Схема данных: {schema_desc}
Пожалуйста, верни ТОЛЬКО JSON в формате:
{{"select":[...],"from":"{table_name}","where":[{{"column":"...","operator":"...","value":"..."}}], "distinct":bool, "limit":N}}
Если не можешь сопоставить — используй 'UNKNOWN_<name>' в поле column.
Вопрос: {question}
"""

    try:
        messages = [{"role":"system","content":"Ты — парсер вопросов. Всегда возвращай только JSON."},
                    {"role":"user","content":prompt}]
        resp = ollama.chat(model=LLM.model, messages=messages)
        raw = resp["message"]["content"].strip()
    except Exception as e:
        logger.exception("LLM call failed")
        return {"type":"error","content":f"Ошибка вызова LLM: {e}"}

    json_str = _extract_json_from_llm_raw(raw)
    try:
        generated_json = json.loads(json_str)
    except Exception as e:
        logger.error("JSON parse error: %s; raw: %s", e, raw)
        return {"type":"error","content":f"Ошибка парсинга JSON: {e}", "raw": raw}

    try:
        sql, params, mappings = build_sql_from_json(generated_json, DB_PATH)
    except Exception as e:
        return {"type":"error","content":f"Ошибка построения SQL: {e}", "json_query": generated_json}

    if mappings.get("unknowns"):
        return {"type":"clarify","content":"Требуется уточнение: " + ", ".join(mappings["unknowns"]), "mappings": mappings, "json_query": generated_json}

    try:
        with engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            data = [dict(r._mapping) for r in rows]
    except Exception as e:
        logger.exception("SQL execution error")
        return {"type":"error","content":f"Ошибка выполнения SQL: {e}", "sql": sql}

    final_text = ""
    if data and len(data) == 1 and len(data[0]) == 1:
        final_text = f"Ответ: {list(data[0].values())[0]}"
    elif data:
        final_text = f"Найдено {len(data)} строк."
    else:
        final_text = "Нет данных."

    return {"type":"final_answer","content":final_text,"result":data,"json_query":generated_json,"sql_query":sql,"mappings":mappings}