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
        """
        from agent.utils.sql_builder import build_sql_from_json
        mappings: Dict[str, Any] = {}
        try:
            sql = build_sql_from_json(task_json, self.db_path)
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

        # Выполнение SQL
        try:
            with self.engine.begin() as conn:
                rows = conn.execute(text(sql)).fetchall()
                data = [dict(r._mapping) for r in rows]
        except Exception as e:
            logger.exception("SQL execution failed: %s", e)
            mappings = {"error": str(e)}
            return sql, [], mappings

        return sql, data, mappings