# agent/llm/ollama_client.py
import re
from typing import Dict
import ollama

DEFAULT_MODEL = "digital_twin_analyst"
SYS_PROMPT = (
    "Ты — SQL-генератор для SQLite. "
    "Твоя задача - сгенерировать команду CREATE TABLE на основе предоставленного описания структуры данных. "
    "Формат: BLOCK_SQL:\n<одна команда SQLite CREATE TABLE>\nEND_SQL\n"
    "Не пиши пояснений, только SQL в указанном формате. "
    "Если столбец содержит числовые данные, используй REAL или INTEGER. "
    "Если столбец содержит текст, используй TEXT. "
    "Используй имена столбцов из предоставленного описания, нормализованные (без пробелов, с подчёркиваниями). "
    "Добавь PRIMARY KEY, если есть столбец с уникальным идентификатором (например, 'id', 'п_п')."
)


class AnalystLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def generate_schema(self, extracted_text: str) -> Dict[str, str]:
        # --- НОВОЕ: Ограничиваем объём текста ---
        # Предположим, что extracted_text - это CSV
        # Возьмём первые N строк (например, 10: заголовки + 9 строк данных)
        lines = extracted_text.splitlines()
        if len(lines) > 10: # Если строк больше 10
            # Берём заголовки (0-я строка) и первые 9 строк данных (1-9)
            truncated_text = "\n".join(lines[:10])
            print(f"[DEBUG generate_schema] Обрезанный текст (первые 10 строк) для LLM: {truncated_text}") # Отладка
        else:
            truncated_text = extracted_text
            print(f"[DEBUG generate_schema] Полный текст (меньше 10 строк) для LLM: {truncated_text}") # Отладка
        # --- КОНЕЦ НОВОГО ---

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Описание структуры данных (CSV-заголовки и первые строки):\n{truncated_text}\nСформируй SQL-схему таблицы."}
        ]
        resp = ollama.chat(model=self.model, messages=messages)
        raw = resp["message"]["content"]

        print(f"[DEBUG generate_schema] Ответ LLM (RAW): {repr(raw)}") # Отладка

        # --- НОВАЯ ЛОГИКА ИЗВЛЕЧЕНИЯ SQL ---
        sql_raw = ""
        # 1. Попробуем найти BLOCK_SQL ... END_SQL
        sql_match = re.search(r'BLOCK_SQL:\s*(?:```sql\s*)?(.*?)(?:```)?\s*END_SQL', raw, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql_raw = sql_match.group(1).strip()
            print(f"[DEBUG generate_schema] Извлечённый SQL из BLOCK_SQL...END_SQL: {repr(sql_raw)}") # Отладка
        else:
            print(f"[DEBUG generate_schema] Блок BLOCK_SQL...END_SQL не найден.") # Отладка
            # 2. Если не найден, попробуем найти CREATE TABLE в любом месте raw
            # Ищем CREATE TABLE ..., затем (опционально) IF NOT EXISTS, затем имя таблицы и содержимое в скобках
            lines_raw = raw.splitlines()
            create_start_idx = -1
            for i, line in enumerate(lines_raw):
                if line.strip().upper().startswith("CREATE TABLE"):
                    create_start_idx = i
                    print(f"[DEBUG generate_schema] Найдена строка CREATE TABLE на индексе {i}") # Отладка
                    break

            if create_start_idx != -1:
                 # Начинаем собирать SQL с этой строки
                 sql_lines = []
                 bracket_count = 0
                 found_opening_bracket = False
                 for j in range(create_start_idx, len(lines_raw)):
                     current_line = lines_raw[j]
                     sql_lines.append(current_line)

                     # Подсчитываем скобки
                     for char in current_line:
                         if char == '(':
                             bracket_count += 1
                             found_opening_bracket = True
                         elif char == ')':
                             bracket_count -= 1

                     # Если нашли открывающую скобку и баланс скобок равен 0, это конец определения таблицы
                     if found_opening_bracket and bracket_count == 0:
                         # Обрежем строку на последней скобке
                         last_line = sql_lines[-1]
                         last_bracket_pos = last_line.rfind(')')
                         if last_bracket_pos != -1:
                             sql_lines[-1] = last_line[:last_bracket_pos+1] # +1 чтобы включить скобку
                         break

                 sql_raw = "\n".join(sql_lines).strip()
                 print(f"[DEBUG generate_schema] Извлечённый SQL из CREATE TABLE: {repr(sql_raw)}") # Отладка
            else:
                 print(f"[DEBUG generate_schema] CREATE TABLE не найден в ответе LLM.") # Отладка
                 sql_raw = ""

        # --- КОНЕЦ НОВОЙ ЛОГИКИ ИЗВЛЕЧЕНИЯ ---

        # --- НОВОЕ: Пост-обработка SQL - очистка имён столбцов ---
        # Ищем строки, которые выглядят как определения столбцов: пробелы, имя_столбца, пробел, тип
        # Исправляем имена столбцов, убирая точки в начале/середине (но не внутри типа, например, 'TEXT NOT NULL')
        if sql_raw:
            # Паттерн: (пробелы) (имя_столбца_потенциально_с_ошибками) (пробелы) (тип_данных_и_опционально_ограничения)
            # Группа 1: начальные пробелы
            # Группа 2: имя столбца (здесь нужно очистить)
            # Группа 3: пробелы между именем и типом
            # Группа 4: тип данных и ограничения
            # Убираем точки из имени столбца (после нормализации в _ и т.п.)
            # Более аккуратный паттерн: ищем имя столбца как последовательность \w и _, но исправляем вхождения . перед \w
            # Заменяем ., за которой следует \w, на _
            # Это может быть грубовато. Лучше: разбить на токены и обработать имя столбца отдельно.
            # Попробуем заменить . в начале слова или ., за которой следует буква/цифра/подчёркивание, на _
            # Это может исправить .Podrodnik -> _Podrodnik или Podrodnik (если Podrodnik начинается с заглавной)
            # Но лучше: найти определение столбца и исправить только имя.
            # Паттерн: (пробелы) (имя_столбца_potentially_with_dots) (пробелы) (тип)
            # Используем re.MULTILINE для ^ и $
            # (\s*)([\w.]+)(\s+)(\w.*) - захватывает имя и тип
            # Но это не учитывает кавычки и сложные типы.
            # Попробуем более простой подход: заменим ., за которой следует \w, на _ внутри скобок определения таблицы.
            # Это может повлиять на типы данных, содержащие точки (редко в SQLite), но улучшит имена столбцов.
            # Лучше: найдём определения столбцов внутри CREATE TABLE ( ... )
            # (\s*)([^\s,)]+)(\s+)([^\s,)]+) - пробелы, имя, пробелы, тип (до запятой или скобки)
            # Это сложно. Попробуем простую замену точки в начале слова (после запятой или скобки или пробела).
            # Заменяем ., если за ней следует \w (буква/цифра/подчёркивание), на _
            # (?<=,|\s|\() - позитивное утверждение назад: запятая, пробел или (
            # \. - точка
            # (?=\w) - позитивное утверждение вперёд: \w
            # На что заменяем: _
            # sql_raw = re.sub(r'(?<=,|\s|\()\.(?=\w)', '_', sql_raw)

            # Ещё проще: удалим точки в начале потенциальных имён столбцов.
            # Имя столбца в CREATE TABLE обычно идёт после запятой или открывающей скобки, за ним пробел, затем тип.
            # Паттерн: ищем (запятая или скобка) + . + (буква/цифра/подчёркивание)
            # (?<=,|\() - позитивное утверждение назад: запятая или (
            # \. - точка
            # ([a-zA-Z_][a-zA-Z0-9_]*) - имя столбца после точки
            # Заменяем на \1 (имя столбца без точки)
            # Это не сработает, если . посередине. Попробуем другой подход: уберём точку в начале, если она есть.
            # (?<=,|\s|\() - после запятой, пробела или (
            # \.([a-zA-Z_][a-zA-Z0-9_]*) - точка и имя столбца
            # Заменяем на \1 (только имя столбца)
            # sql_raw = re.sub(r'(?<=,|\s|\()\.\b([a-zA-Z_][a-zA-Z0-9_]*)\b', r'\1', sql_raw, flags=re.MULTILINE)

            # Самый простой и грубый, но часто работающий способ для этого конкретного случая:
            # Просто заменим `.ИмяСтолбца` на `ИмяСтолбца` везде внутри определения CREATE TABLE.
            # Ищем ., за которой следует слово (без пробелов до слова).
            # (?<!\w) - не после \w
            # \. - точка
            # ([a-zA-Z_][a-zA-Z0-9_]*) - имя столбца
            # (?!\w) - не перед \w
            # sql_raw = re.sub(r'(?<!\w)\.([a-zA-Z_][a-zA-Z0-9_]*)\b', r'\1', sql_raw)

            # Или, учитывая контекст CREATE TABLE ( ... ):
            # Найдём всё внутри скобок и заменим . в начале слов
            def fix_column_names_in_create_table(match):
                full_create_table = match.group(0)
                # Найдём часть внутри скобок
                # Паттерн для скобок: \( ... \)
                inner_content_match = re.search(r'\((.*)\)', full_create_table, re.DOTALL)
                if inner_content_match:
                    inner_content = inner_content_match.group(1)
                    # Заменим . в начале слов (после запятой, скобки, пробела или начала строки внутри скобок)
                    # (?<=,|\(|\s) - позитивное утверждение назад: запятая, ( или пробел
                    # \. - точка
                    # ([a-zA-Z_][a-zA-Z0-9_]*) - имя столбца
                    fixed_inner = re.sub(r'(?<=,|\(|\s)\.([a-zA-Z_][a-zA-Z0-9_]*)\b', r'\1', inner_content)
                    # Соберём обратно
                    fixed_create_table = full_create_table.replace(inner_content, fixed_inner)
                    return fixed_create_table
                else:
                    # Если скобки не найдены, вернём как есть
                    return full_create_table

            # Применим функцию к самому CREATE TABLE выражению
            sql_raw = re.sub(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s*\([^)]*\)', fix_column_names_in_create_table, sql_raw, flags=re.IGNORECASE | re.MULTILINE)

            print(f"[DEBUG generate_schema] SQL после очистки имён столбцов: {repr(sql_raw)}") # Отладка
        # --- КОНЕЦ НОВОГО ---

        # --- НОВОЕ: Обеспечиваем IF NOT EXISTS (только если sql_raw не пустой и начинается с CREATE TABLE) ---
        if sql_raw and sql_raw.upper().startswith("CREATE TABLE"):
             # Заменяем CREATE TABLE на CREATE TABLE IF NOT EXISTS
             # Паттерн: CREATE (возможно с пробелами), затем TABLE, затем (возможно) IF NOT EXISTS, затем имя таблицы
             # Заменяем на CREATE TABLE IF NOT EXISTS + найденное имя таблицы
             # Это работает как если было 'CREATE TABLE имя_таблицы ...', так и 'CREATE TABLE IF NOT EXISTS имя_таблицы ...'
             sql_raw = re.sub(
                 r'^(\s*)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
                 r'\1CREATE TABLE IF NOT EXISTS \2',
                 sql_raw,
                 flags=re.IGNORECASE | re.MULTILINE
             )
             print(f"[DEBUG generate_schema] SQL после добавления IF NOT EXISTS: {repr(sql_raw)}") # Отладка
        elif sql_raw: # Если sql_raw не пустой, но не начинается с CREATE TABLE
             print(f"[ERROR generate_schema] Извлечённый SQL не начинается с CREATE TABLE: {repr(sql_raw)}")
             sql_raw = ""
        # --- КОНЕЦ НОВОГО ---

        return {"sql": sql_raw, "models": ""}


# быстрый self-test
if __name__ == "__main__":
    llm = AnalystLLM()
    dummy = "Object: Road 12 km, 4 lanes, asphalt."
    print(llm.generate_schema(dummy))
