from sqlalchemy import create_engine, inspect
from pathlib import Path
from typing import List

DB_PATH = Path(__file__).resolve().parent.parent.parent / "generated" / "digital_twin.db"


def _sqlalchemy_type(col: dict) -> str:
    """Переводит тип SQLite в SQLAlchemy."""
    sql_type = col["type"].__visit_name__.upper()
    return {
        "INTEGER": "Integer",
        "REAL": "Float",
        "TEXT": "String",
        "BLOB": "LargeBinary",
        "DATE": "Date",
    }.get(sql_type, "String")


def generate_model_py(table_name: str) -> str:
    """Генерирует текст SQLAlchemy-класса по реальной таблице в БД."""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

    # если таблицы ещё нет — вернём заглушку
    if table_name not in inspect(engine).get_table_names():
        return f"# Table '{table_name}' not found in DB yet\n"

    columns: List[dict] = inspect(engine).get_columns(table_name)

    fields = [f"    id = Column(Integer, primary_key=True)"]
    for col in columns:
        if col["name"] == "id":
            continue  # id уже объявили
        sa_type = _sqlalchemy_type(col)
        nullable = "" if col.get("nullable", True) is True else ", nullable=False"
        default = f", default={col['default']!r}" if col.get("default") is not None else ""
        fields.append(f"    {col['name']} = Column({sa_type}{nullable}{default})")

    fields_str = "\n".join(fields)

    return f"""# AUTO-GENERATED SQLAlchemy model
from sqlalchemy import Column, Integer, Float, String, Date, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class {table_name.title().replace('_', '')}(Base):
    __tablename__ = '{table_name}'
{fields_str}
"""