from typing import List, Tuple
import pandas as pd
from sqlalchemy import create_engine, text

def find_similar_values(
    db_path: str,
    table: str,
    column: str,
    user_text: str,
    limit: int = 5,
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Возвращает [(значение, скор)], отсортированные по похожести.
    Используем простую метрику: ratio из difflib.
    """
    import difflib

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as conn:
        rows = conn.execute(
            text(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL')
        ).fetchall()
        unique = [r[0] for r in rows if r[0]]

    matches = difflib.get_close_matches(
        user_text, unique, n=limit, cutoff=threshold
    )
    scored = [(m, difflib.SequenceMatcher(None, user_text, m).ratio()) for m in matches]
    return sorted(scored, key=lambda x: x[1], reverse=True)