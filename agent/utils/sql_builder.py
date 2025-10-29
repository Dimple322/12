from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
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

def _fuzzy_map_column(col_name: str, real_cols: Dict[str, str]) -> "Optional[str]":
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

def build_sql_from_json(query_json: Dict[str, Any], db_path: Union[str, Path]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Преобразует JSON-описание запроса в (sql, params, mappings).

    Контракт:
      - sql: строка SQL для исполнения (если пустая — значит требуются уточнения)
      - params: словарь параметров для parametrized execution (ключи без ':' как в SQLAlchemy)
      - mappings: дополнительная информация, например:
          {"unknowns": [...], "replacements": {...}, "error": "..."} либо {} если ничего

    Поведение при нераспознанных колонках:
    - Если LLM вернул UNKNOWN_... — добавляем их в mappings['unknowns'] и возвращаем sql = "".
    - Если встретилась неизвестная колонка без UNKNOWN_ префикса — также считаем это unknown и возвращаем sql = "" с объяснением.
    """
    db_path = Path(db_path)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    inspector = inspect(engine)

    mappings: Dict[str, Any] = {}
    params: Dict[str, Any] = {}

    try:
        table = query_json.get("from")
        if not table:
            return "", {}, {"error": "'from' не указан в JSON"}

        if table not in inspector.get_table_names():
            return "", {}, {"error": f"Таблица '{table}' не существует."}

        # real_cols: lowercase -> actual column name
        real_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(table)}

        # SELECT
        select_items = query_json.get("select", ["*"])
        if not isinstance(select_items, list):
            return "", {}, {"error": "'select' должен быть списком выражений."}

        cleaned_select: List[str] = []
        replacements: Dict[str, str] = {}
        unknowns: List[str] = []

        for expr in select_items:
            if not isinstance(expr, str):
                return "", {}, {"error": f"Элемент в select должен быть строкой: {expr}"}
            expr_work = expr.strip()
            # Если выражение == "*" — ок
            if expr_work == "*":
                cleaned_select.append(expr_work)
                continue

            # Найдём кандидатов (возможные идентификаторы) в выражении и попытаемся замэппить
            candidates = _extract_candidate_identifiers(expr_work)
            for cand in candidates:
                mapped = _fuzzy_map_column(cand, real_cols)
                if mapped:
                    expr_work = re.sub(rf'(?<!\w){re.escape(cand)}(?!\w)', f'"{mapped}"', expr_work)
                    replacements[cand] = mapped
                else:
                    if re.match(r'^UNKNOWN_[A-Za-z0-9_]+', cand, re.I):
                        if cand not in unknowns:
                            unknowns.append(cand)
                        # оставляем placeholder в выражении, но запрос не будет выполнен
                        continue
                    # неизвестная колонка -> помечаем как unknown для уточнения
                    if cand not in unknowns:
                        unknowns.append(cand)

            cleaned_select.append(expr_work)

        # Если есть unknowns в select — вернём их для уточнения
        if unknowns:
            return "", {}, {"unknowns": unknowns, "replacements": replacements} if replacements else {"unknowns": unknowns}

        select_clause = ", ".join(cleaned_select)

        # WHERE - строим parametrized условия
        where_parts = []
        param_counter = 0
        if "where" in query_json and query_json["where"]:
            if not isinstance(query_json["where"], list):
                return "", {}, {"error": "'where' должен быть списком условий."}
            for cond in query_json["where"]:
                if not isinstance(cond, dict):
                    continue
                col = cond.get("column")
                op = cond.get("operator", "=").strip()
                val = cond.get("value")
                if isinstance(col, str) and re.match(r'^UNKNOWN_[A-Za-z0-9_]+', col, re.I):
                    if col not in unknowns:
                        unknowns.append(col)
                    continue
                mapped = _fuzzy_map_column(str(col), real_cols) if isinstance(col, str) else None
                if not mapped:
                    if col and col not in unknowns:
                        unknowns.append(col)
                    continue
                pname = f"p{param_counter}"
                param_counter += 1
                if op.upper() == "IN" and isinstance(val, (list, tuple)):
                    placeholders = []
                    for i, v in enumerate(val):
                        pn = f"{pname}_{i}"
                        params[pn] = v
                        placeholders.append(f":{pn}")
                    where_parts.append(f'"{mapped}" IN ({", ".join(placeholders)})')
                elif val is None:
                    if op in ("=", "=="):
                        where_parts.append(f'"{mapped}" IS NULL')
                    elif op in ("!=", "<>"):
                        where_parts.append(f'"{mapped}" IS NOT NULL')
                    else:
                        where_parts.append(f'"{mapped}" {op} NULL')
                else:
                    params[pname] = val
                    where_parts.append(f'"{mapped}" {op} :{pname}')

        # Если есть unknowns в where — возвращаем для уточнения
        if unknowns:
            out = {"unknowns": unknowns}
            if replacements:
                out["replacements"] = replacements
            return "", {}, out

        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # GROUP BY
        group_clause = ""
        gb = query_json.get("group_by") or query_json.get("groupby") or []
        if gb:
            if not isinstance(gb, list):
                return "", {}, {"error": "'group_by' должен быть списком."}
            gb_cols = []
            for c in gb:
                mapped = _fuzzy_map_column(str(c), real_cols)
                if not mapped:
                    return "", {}, {"error": f'Столбец для group_by "{c}" не найден', "unknowns": [c]}
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
                    return "", {}, {"error": f'Столбец для order_by "{col}" не найден', "unknowns": [col]}
                parts.append(f'"{mapped}" {dirc}')
            if parts:
                order_clause = " ORDER BY " + ", ".join(parts)

        # LIMIT
        limit_clause = ""
        if "limit" in query_json and query_json["limit"] is not None:
            try:
                limit_clause = f" LIMIT {int(query_json['limit'])}"
            except Exception:
                return "", {}, {"error": "'limit' должен быть целым числом."}

        sql_parts = [f"SELECT {select_clause}", f'FROM "{table}"', where_clause, group_clause, order_clause, limit_clause]
        final_sql = " ".join([p for p in sql_parts if p]).strip()

        # build mappings only if non-empty
        out_mappings = {}
        if replacements:
            out_mappings["replacements"] = replacements

        return final_sql, params, out_mappings

    except Exception as e:
        return "", {}, {"error": f"Ошибка построения SQL из JSON: {e}"}