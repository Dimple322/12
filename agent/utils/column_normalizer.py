# agent/utils/column_normalizer.py
import re

def normalize_column_name(col):
    """Нормализует имя столбца по правилам, используемым в universal_parser.py."""
    # Преобразуем в строку на случай, если имя числовое (например, 15)
    col_str = str(col).strip()
    # Проверяем, начинается ли имя с цифры
    if col_str and col_str[0].isdigit():
        # Если начинается с цифры, добавим префикс, например, 'col_'
        col_str = f"col_{col_str}"
    # Затем заменяем пробелы и специальные символы на подчёркивания
    normalized = re.sub(r'[^\w]', '_', col_str)
    # Убираем множественные подчёркивания
    normalized = re.sub(r'_+', '_', normalized)
    # Убираем ведущие/завершающие подчёркивания
    normalized = normalized.strip('_')
    # На всякий случай, если после всех преобразований имя пустое или только из подчёркиваний
    if not normalized:
         normalized = "unnamed_column"
    return normalized

def normalize_dataframe_columns(df):
    """Нормализует имена столбцов в DataFrame по правилам, используемым в universal_parser.py."""
    original_columns = df.columns
    new_columns = [normalize_column_name(col) for col in original_columns]

    # Обработка дубликатов (важно после нормализации)
    seen = set()
    final_columns = []
    for col in new_columns:
        counter = 1
        original_col = col
        while col in seen:
            col = f"{original_col}_{counter}"
            counter += 1
        seen.add(col)
        final_columns.append(col)

    df_normalized = df.copy()
    df_normalized.columns = final_columns
    return df_normalized
