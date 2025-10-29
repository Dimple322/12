from pathlib import Path
from typing import Optional
from agent.thinking_agent import ThinkingAgent, Message as ThinkMessage
from agent.sql_agent import SQLAgent

class AnalyticAgent:
    """
    Координатор: использует ThinkingAgent для reasoning/clarify и SQLAgent для execution.
    API:
      - process_user(question: str) -> Message (assistant message)
      - apply_clarification(clar: str) -> Message
    """
    def __init__(self, db_path: Path, kb_path: Optional[Path] = None):
        self.thinker = ThinkingAgent(db_path, kb_path)
        self.exec = SQLAgent(db_path)

    def process_user(self, question: str) -> ThinkMessage:
        thinking_msg = self.thinker.think(question)
        meta = thinking_msg.metadata or {}
        json_q = meta.get("json_query")
        mappings = meta.get("mappings") or {}
        if json_q and not mappings.get("unknowns"):
            sql, rows, exec_mappings = self.exec.execute(json_q)
            if exec_mappings and exec_mappings.get("unknowns"):
                msg = ThinkMessage("assistant", "Нужны уточнения перед выполнением запроса.", metadata={"json_query": json_q, "mappings": exec_mappings})
                self.thinker.history.append(msg)
                return msg
            if exec_mappings and exec_mappings.get("error"):
                msg = ThinkMessage("system", f"Ошибка выполнения SQL: {exec_mappings.get('error')}", metadata={"json_query": json_q, "mappings": exec_mappings})
                self.thinker.history.append(msg)
                return msg
            content = self._format_answer(rows)
            msg = ThinkMessage("assistant", content, metadata={"json_query": json_q, "sql": sql, "rows": rows})
            self.thinker.history.append(msg)
            return msg
        return thinking_msg

    def apply_clarification(self, clar: str) -> ThinkMessage:
        thinking_reply = self.thinker.apply_user_clarification(clar)
        meta = thinking_reply.metadata or {}
        json_q = meta.get("json_query")
        mappings = meta.get("mappings") or {}
        if json_q and not mappings.get("unknowns"):
            sql, rows, exec_mappings = self.exec.execute(json_q)
            if exec_mappings and exec_mappings.get("unknowns"):
                msg = ThinkMessage("assistant", "После уточнения всё ещё нужны уточнения.", metadata={"json_query": json_q, "mappings": exec_mappings})
                self.thinker.history.append(msg)
                return msg
            if exec_mappings and exec_mappings.get("error"):
                msg = ThinkMessage("system", f"Ошибка выполнения SQL: {exec_mappings.get('error')}", metadata={"json_query": json_q, "mappings": exec_mappings})
                self.thinker.history.append(msg)
                return msg
            content = self._format_answer(rows)
            msg = ThinkMessage("assistant", content, metadata={"json_query": json_q, "sql": sql, "rows": rows})
            self.thinker.history.append(msg)
            return msg
        return thinking_reply

    def _format_answer(self, rows):
        if not rows:
            return "Ничего не найдено."
        if len(rows) == 1 and len(rows[0]) == 1:
            return f"Ответ: {list(rows[0].values())[0]}"
        return f"Найдено записей: {len(rows)}. Первые 3: {rows[:3]}"