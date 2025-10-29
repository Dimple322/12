from pathlib import Path
from typing import Dict, Any, List
import re
import difflib
from sqlalchemy import create_engine, inspect

SQL_KEYWORDS = {
    'SELECT','FROM','WHERE','GROUP','BY','ORDER','HAVING','JOIN','ON','AND','OR','AS','IN','NOT','NULL',
    'IS','INSERT','INTO','VALUES','UPDATE','SET','DELETE','LIMIT','OFFSET','UNION','ALL','DISTINCT',
    'SUM','AVG','COUNT','MIN','MAX'
}

def _extract_candidate_identifiers(expr: str) -> List[str]:
    """
    Извлекает токены-предположения (возможные имена колонок / идентификаторы)
    из выражения select/where. Не находит имена функций вызова (перед '()').
    """
    expr_no_str = re.sub(r"'[^']*'|\"[^\"]*\"", " ", expr)  # убрать строковые литералы
    # взять слова, которые не сопровождаются '(' (т.е. не функции)
    tokens = re.findall(r'(?<!\w)([A-Za-zА-Яа-я0-9_]+)(?!\s*\()', expr_no_str)
    filtered = []
    seen = set()
    for t in tokens:
        if not t:
            continue
        if t.upper() in SQL_KEYWORDS:
            continue
        low = t.lower()
        if low not in seen:
            seen.add(low)
            filtered.append(t)
    return filtered

def _fuzzy_map_column(col_name: str, real_cols: Dict[str, str]) -> str | None:
    """
    Попытка мэппинга имени (возможно не точного) на реальные имена колонок.
    real_cols: mapping lowercase -> actual_name
    """
    if not col_name:
        return None
    low = col_name.lower()
    if low in real_cols:
        return real_cols[low]
    # 1) пытаться найти точный похожий по названию (case-sensitive values)
    matches = difflib.get_close_matches(col_name, list(real_cols.values()), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    # 2) попытаться сопоставить по lowercase keys
    matches2 = difflib.get_close_matches(low, list(real_cols.keys()), n=1, cutoff=0.6)
    if matches2:
        return real_cols[matches2[0]]
    return None

def build_sql_from_json(query_json: Dict[str, Any], db_path: str | Path) -> str:
    """
    Преобразует JSON-описание запроса в SQL-строку.
    Поддерживает select выражения с агрегатами/функциями (e.g. MAX(Дата) AS last_date),
    а также нормализует и проверяет имена колонок в выражениях.
    """
    db_path = Path(db_path)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    inspector = inspect(engine)

    try:
        table = query_json["from"]
        if table not in inspector.get_table_names():
            raise ValueError(f"Таблица '{table}' не существует.")

        # real_cols: lowercase -> actual column name
        real_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(table)}

        # SELECT
        select_items = query_json.get("select", ["*"])
        if not isinstance(select_items, list):
            raise ValueError("'select' должен быть списком выражений.")

        cleaned_select: List[str] = []
        for expr in select_items:
            if not isinstance(expr, str):
                raise ValueError(f"Элемент в select должен быть строкой: {expr}")
            expr_work = expr.strip()
            # Если выражение == "*" — ок
            if expr_work == "*":
                cleaned_select.append(expr_work)
                continue

            # Найдём кандидатов (возможные идентификаторы) в выражении и попытаемся замэппить
            candidates = _extract_candidate_identifiers(expr_work)
            for cand in candidates:
                # пропускаем числа, ключевые слова, функции (MAX и т.п. уже исключены)
                mapped = _fuzzy_map_column(cand, real_cols)
                if mapped:
                    # Заменяем в выражении все вхождения токена cand на "mapped"
                    # Используем границы слов, чтобы не порезать другие слова
                    expr_work = re.sub(rf'(?<!\w){re.escape(cand)}(?!\w)', f'"{mapped}"', expr_work)
                else:
                    # Если не нашли мэппинга — возможно LLM вернул UNKNOWN_... — оставляем как есть
                    # но пометим это как потенциальную проблему: SQLBuilder не сможет выполнить запрос с несуществующими колонками.
                    # Здесь мы предпочитаем вернуть ошибку, чтобы вызывающий компонент мог запросить уточнение.
                    if re.match(r'^UNKNOWN_', cand, re.I):
                        # пусть вызывающий обработает UNKNOWN_...
                        continue
                    # иначе — колонка не найдена -> выбрасываем
                    raise ValueError(f'Столбец "{cand}" (в выражении "{expr}") не найден в таблице {table}.')

            cleaned_select.append(expr_work)

        select_clause = ", ".join(cleaned_select)

        # WHERE
        where_parts = []
        if "where" in query_json and query_json["where"]:
            for cond in query_json["where"]:
                col = cond.get("column")
                op = cond.get("operator", "=")
                val = cond.get("value")
                if isinstance(col, str) and col.startswith("UNKNOWN_"):
                    # Unknown placeholder — вызывающий должен инициировать уточнение; сигнализируем об ошибке
                    raise ValueError(f'Неизвестная колонка/placeholder в where: "{col}"')
                # постараемся найти колонку (функции в where не поддерживаем)
                mapped = _fuzzy_map_column(col, real_cols) if isinstance(col, str) else None
                if not mapped:
                    raise ValueError(f'Столбец "{col}" не найден в таблице {table}.')
                # Экранируем значение
                val_str = str(val).replace("'", "''") if val is not None else ""
                where_parts.append(f'"{mapped}" {op} \'{val_str}\'')
        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # GROUP BY (если есть) — принимаем список
        group_clause = ""
        gb = query_json.get("group_by") or query_json.get("groupby") or []
        if gb:
            if not isinstance(gb, list):
                raise ValueError("'group_by' должен быть списком")
            gb_cols = []
            for c in gb:
                mapped = _fuzzy_map_column(str(c), real_cols)
                if not mapped:
                    raise ValueError(f'Столбец для group_by "{c}" не найден в таблице {table}.')
                gb_cols.append(f'"{mapped}"')
            group_clause = " GROUP BY " + ", ".join(gb_cols)

        # ORDER BY
        order_clause = ""
        ob = query_json.get("order_by") or query_json.get("orderby") or []
        if ob:
            parts = []
            for item in ob:
                if isinstance(item, dict):
                    col = item.get("column")
                    dirc = (item.get("direction") or "asc").upper()
                elif isinstance(item, str):
                    toks = item.split()
                    col = toks[0]
                    dirc = toks[1].upper() if len(toks) > 1 else "ASC"
                else:
                    continue
                mapped = _fuzzy_map_column(str(col), real_cols)
                if not mapped:
                    raise ValueError(f'Столбец для order_by "{col}" не найден в таблице {table}.')
                parts.append(f'"{mapped}" {dirc}')
            if parts:
                order_clause = " ORDER BY " + ", ".join(parts)

        # LIMIT
        limit_clause = ""
        if "limit" in query_json and query_json["limit"] is not None:
            try:
                limit_clause = f" LIMIT {int(query_json['limit'])}"
            except Exception:
                raise ValueError("'limit' должен быть целым числом.")

        sql_parts = [f"SELECT {select_clause}", f'FROM "{table}"', where_clause, group_clause, order_clause, limit_clause]
        final_sql = " ".join([p for p in sql_parts if p]).strip()
        return final_sql

    except Exception as e:
        raise ValueError(f"Ошибка построения SQL из JSON: {e}") from e