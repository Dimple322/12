# agent/main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный файл локального агента-аналитика.
Следит за папкой data/incoming, парсит файлы, строит БД, модели, векторный индекс,
заполняет таблицы данными и отвечает на вопросы.
"""
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------- импорты агентов ----------
from agent.llm.ollama_client import AnalystLLM
from agent.parser.universal_parser import extract_text
# from agent.db.schema_builder import build_db, get_engine, DB_PATH # <-- Убираем DB_PATH отсюда
from agent.db.schema_builder import build_db # <-- Импортируем только функцию
from agent.codegen.sqlalchemy_gen import generate_model_py
from agent.db.loader import load_excel_to_db
from agent.vector.indexer import index_documents


# ---------- пути ----------
ROOT      = Path(__file__).resolve().parent.parent
INBOX     = ROOT / "data" / "incoming"
PROCESSED = ROOT / "data" / "processed"
GENERATED = ROOT / "generated"
# --- НОВОЕ: Определяем DB_PATH здесь ---
DB_PATH   = GENERATED / "digital_twin.db"
# --- КОНЕЦ НОВОГО ---

llm = AnalystLLM()


class NewFileHandler(FileSystemEventHandler):
    """Обрабатывает только появление нового файла."""

    def on_created(self, event):
        if event.is_directory:
            return
        self.handle_file(Path(event.src_path))

    # --------- основной пайплайн ---------
    def handle_file(self, file: Path):
        if file.suffix.lower() not in {".txt", ".md", ".pdf", ".xlsx", ".xls", ".pptx"}:
            return
        if not file.exists():
            print(f"[SKIP] File already processed: {file.name}")
            return

        print(f"📥  Обнаружен файл: {file.name}")
        try:
            # 1. Извлечение текста
            text = extract_text(file)

            # 2. Генерация SQL-схемы
            schema = llm.generate_schema(text)
            if not schema.get("sql") or "CREATE TABLE" not in schema["sql"].upper():
                print(f"❌  Ошибка: LLM не сгенерировала корректный SQL CREATE TABLE для {file.name}")
                return  # Прекращаем обработку этого файла

            try:
                engine = build_db(schema["sql"], DB_PATH)
                if engine is None:  # Предположим, build_db возвращает None при ошибке
                    raise ValueError("build_db не смог создать таблицу")
            except Exception as e:
                print(f"❌  Ошибка создания таблицы для {file.name}: {e}")
                return

            # 3. Создание / обновление БД
            # --- ИЗМЕНЕНО: Передаем DB_PATH в build_db ---
            engine = build_db(schema["sql"], DB_PATH) # <-- ИСПРАВЛЕНО
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # 4. Имя таблицы из SQL
            import re
            # --- УЛУЧШЕНИЕ: Более надежный способ найти имя таблицы ---
            # Ищем CREATE TABLE [IF NOT EXISTS] имя_таблицы
            table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', schema["sql"], re.I)
            table_name = table_match.group(1) if table_match else "tbl"
            # --- КОНЕЦ УЛУЧШЕНИЯ ---

            # 5. Генерация SQLAlchemy-модели (если нужно)
            # model_code = generate_model_py(table_name)
            # models_file = GENERATED / f"{file.stem}_models.py"
            # models_file.write_text(model_code, encoding="utf-8")

            # 6. Загрузка данных (для Excel)
            if file.suffix.lower() in {".xlsx", ".xls"}:
                print(f"[DEBUG] Заливаем Excel в таблицу '{table_name}'")
                # --- ИЗМЕНЕНО: Передаем DB_PATH в load_excel_to_db ---
                load_excel_to_db(file, table_name, DB_PATH) # <-- ИСПРАВЛЕНО
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            # 7. Векторный индекс (по строкам из БД)
            self._build_vector_index(engine, table_name, file)

            # 8. Перемещение
            dest = PROCESSED / file.name
            file.rename(dest)
            print("✅  Готово:", file.stem)

        except Exception as e:
            print(f"❌  Ошибка обработки {file.name}: {e}")
            # import traceback
            # traceback.print_exc() # Для более подробной отладки

    # --------- векторный индекс ---------
    def _build_vector_index(self, engine, table_name: str, file: Path): # <-- Тип table_name исправлен
        try:
            import pandas as pd
            # --- ИЗМЕНЕНО: Передаем DB_PATH в index_documents ---
            # index_documents(texts, metas) внутри функции index_documents уже знает путь к БД
            # Нам нужно передать engine и table_name правильно
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            # Читаем данные напрямую из таблицы
            rows = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 1000", con=engine)
            if rows.empty:
                return
            # Преобразуем строки в текст. Можно улучшить, объединяя значения по строкам.
            texts = rows.apply(lambda r: " ".join(r.astype(str)), axis=1).tolist()
            # Метаданные (например, источник файла)
            metas = [{"source": file.name, "table": table_name} for _ in texts] # <-- Используем правильное table_name
            index_documents(texts, metas)
        except Exception as e:
            print(f"[WARN] Индекс не построен: {e}")


# ---------------- запуск ----------------
def start_watch():
    PROCESSED.mkdir(exist_ok=True)
    GENERATED.mkdir(exist_ok=True)

    handler = NewFileHandler()
    observer = Observer()
    observer.schedule(handler, str(INBOX), recursive=False)
    observer.start()
    print(f"👀  Наблюдаю за {INBOX} ...  (Ctrl-C для остановки)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watch()
