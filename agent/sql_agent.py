from pathlib import Path
from typing import List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from agent.utils.logger import get_logger
import re

logger = get_logger(__name__)

class SQLAgent:
    def __init__(self, db_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        self.db_path = db_path

    def execute(self, task_json: dict) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Унифицированный контракт: возвращаем (sql, rows, mappings).
        mappings может содержать 'unknowns' или 'error'.
        Поддерживаем разные форматы возврата build_sql_from_json:
          - (sql, params, mappings)
          - {"sql": sql, "params": params, "mappings": mappings}
          - sql (str)
        ВАЖНО: если mappings содержит unknowns -> не выполнять SQL, вернуть mappings для уточнения.
        """
        from agent.utils.sql_builder import build_sql_from_json
        mappings: Dict[str, Any] = {}
        sql = ""
        params: Dict[str, Any] = {}

        try:
            res = build_sql_from_json(task_json, self.db_path)
            # Поддерживаем несколько форматов
            if isinstance(res, tuple):
                sql, params, build_mappings = res
                params = params or {}
                mappings = build_mappings or {}
            elif isinstance(res, dict):
                sql = res.get("sql", "") or ""
                params = res.get("params", {}) or {}
                mappings = res.get("mappings", {}) or {}
            else:
                sql = str(res)
                params = {}
                mappings = {}
        except Exception as e:
            msg = str(e)
            logger.warning("SQLBuilder error: %s", msg)
            # Попытка извлечь UNKNOWN_ placeholders прямо из json_query
            unknowns = []

            def collect(u):
                if isinstance(u, dict):
                    for v in u.values(): collect(v)
                elif isinstance(u, list):
                    for x in u: collect(x)
                elif isinstance(u, str):
                    unknowns.extend(re.findall(r'UNKNOWN_[A-Za-z0-9_]+', u))
            collect(task_json)
            mappings = {"error": msg, "unknowns": list(dict.fromkeys(unknowns))}
            return "", [], mappings

        # Если билдер вернул unknowns — не выполнять SQL, вернуть mappings для clarification
        if mappings.get("unknowns"):
            logger.info("SQLAgent: build returned unknowns, skipping execution: %s", mappings.get("unknowns"))
            return "", [], mappings

        # Если SQL пустой — возврат с mappings/error
        if not sql:
            if not mappings:
                mappings = {"error": "Empty SQL generated"}
            return "", [], mappings

        # Исполнение SQL с параметрами (parametrized)
        try:
            with self.engine.begin() as conn:
                if params:
                    rows = conn.execute(text(sql), params).fetchall()
                else:
                    rows = conn.execute(text(sql)).fetchall()
                data = [dict(r._mapping) for r in rows]
        except Exception as e:
            logger.exception("SQL execution failed: %s", e)
            mappings = {"error": str(e)}
            return sql, [], mappings

        return sql, data, mappings