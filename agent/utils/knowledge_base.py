from pathlib import Path
import json
import time
import sys

# Путь к файлу базы знаний (относительно репо)
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent.parent
KB_PATH = project_root / "generated" / "knowledge_base.json"

def _backup_corrupt_file(path: Path):
    try:
        ts = time.strftime("%Y%m%dT%H%M%S")
        bak = path.with_suffix(path.suffix + f".corrupt.{ts}")
        path.replace(bak)
        print(f"[KB] ⚠️ Файл базы знаний был некорректен и перемещён в: {bak}")
    except Exception as e:
        print(f"[KB] Не удалось создать резервную копию повреждённого KB: {e}")

def load_knowledge_base():
    """
    Загружает базу знаний. Если файл повреждён — делает резервную копию и возвращает [].
    Нормализует записи: если json_query хранится как строка — пытается распарсить в dict.
    """
    if not KB_PATH.exists():
        return []

    try:
        with KB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[KB ERROR] Ошибка чтения файла базы знаний: {e}")
        # делаем резервную копию и возвращаем пустую базу
        try:
            _backup_corrupt_file(KB_PATH)
        except Exception:
            pass
        return []
    except Exception as e:
        print(f"[KB ERROR] Не удалось открыть файл базы знаний: {e}")
        return []

    if not isinstance(data, list):
        print("[KB WARNING] Файл базы знаний не содержит список записей — приводим к пустому списку.")
        return []

    # Нормализация: привести json_query к dict если он сохранён как строка
    normalized = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        j = entry.copy()
        jq = j.get("json_query")
        if isinstance(jq, str):
            try:
                j["json_query"] = json.loads(jq)
            except Exception:
                # оставим строку (позже код должен поддерживать оба варианта)
                j["json_query_raw"] = jq
                j["json_query"] = None
        normalized.append(j)
    return normalized

def save_knowledge_base_entry(question: str, correct_json_query):
    """
    Сохраняет запись в KB.
    correct_json_query может быть dict или строкой; мы сохраняем как dict.
    """
    kb = load_knowledge_base()
    # нормализуем input
    if isinstance(correct_json_query, str):
        try:
            jq_obj = json.loads(correct_json_query)
        except Exception:
            # если не парсится — сохраняем как сырой текст в поле json_query_raw
            jq_obj = None
            raw = correct_json_query
        else:
            raw = None
    else:
        jq_obj = correct_json_query
        raw = None

    # Проверка дубликатов (по question + json_query)
    for entry in kb:
        if entry.get("question") == question:
            # если json_query совпадает — считаем дубликатом
            existing = entry.get("json_query")
            if existing == jq_obj:
                print("[KB] Запись уже существует — пропускаем.")
                return

    new_entry = {
        "question": question,
        "json_query": jq_obj,
        "json_query_raw": raw,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user": "unknown"
    }
    kb.append(new_entry)

    # Пишем файл atomically
    try:
        KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = KB_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)
            f.flush()
        tmp.replace(KB_PATH)
        print("[KB] ✅ Запись добавлена в базу знаний.")
    except Exception as e:
        print(f"[KB ERROR] Не удалось сохранить запись в KB: {e}")