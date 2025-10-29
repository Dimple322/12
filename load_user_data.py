#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска продукта с ChromaDB.
Полная интеграция с пользовательскими данными и динамическими сценариями.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import argparse
import logging
import chardet, io
from gpu_embed_global import GpuMiniLMEmbedding  # GPU-эмбеддинг

_product_test_suite = None

def _get_product_test_suite():
    global _product_test_suite
    if _product_test_suite is None:
        from test_system_chroma import ProductTestSuite
        _product_test_suite = ProductTestSuite
    return _product_test_suite

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def check_chroma_installation():
    """Проверяет установку ChromaDB"""
    try:
        import chromadb
        print("✅ ChromaDB установлен")
        return True
    except ImportError:
        print("❌ ChromaDB не установлен")
        return False


def install_chroma():
    """Устанавливает ChromaDB"""
    print("📥 Установка ChromaDB...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "chromadb"], check=True)
        print("✅ ChromaDB установлен успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при установке ChromaDB: {e}")
        return False


def setup_environment():
    """Настраивает окружение для продукта"""

    print("🚀 НАСТРОЙКА ПРОДУКТА")
    print("=" * 40)

    # Проверка и установка ChromaDB
    if not check_chroma_installation():
        if not install_chroma():
            return False

    # Создание необходимых директорий
    directories = [
        "generated",
        "data/incoming",
        "data/processed",
        "logs",
        "exports",
        "user_data"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана директория: {directory}")

    # Проверка других зависимостей
    required_packages = [
        "pandas",
        "numpy",
        "ollama",
        "scikit-learn",
        "plotly",
        "matplotlib",
        "seaborn"
    ]

    print(f"\n📋 Проверка зависимостей...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n⚠️  Установка недостающих пакетов:")
        for package in missing_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ {package} установлен")
            except subprocess.CalledProcessError as e:
                print(f"❌ Ошибка установки {package}: {e}")

    print(f"\n✅ Окружение продукта настроено!")
    return True


def run_product_test():
    """Запускает тестирование продукта"""

    print(f"\n🧪 ТЕСТИРОВАНИЕ ПРОДУКТА")
    print("=" * 30)

    try:
        from test_system_chroma import ProductTestSuite

        tester = ProductTestSuite()
        results = asyncio.run(tester.run_comprehensive_test())

        successful_tests = len([r for r in results if r.get("passed", False)])
        total_tests = len(results)

        print(f"\n📈 Результаты тестирования:")
        print(f"   Успешных тестов: {successful_tests}/{total_tests} ({successful_tests / total_tests * 100:.1f}%)")

        if successful_tests >= total_tests * 0.8:  # 80% проходной балл для продукта
            print("✅ Продукт готов к использованию!")
            return True
        else:
            print("❌ Продукт требует доработки")
            return False

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False


def show_user_menu():
    """Показывает меню пользователя"""

    print(f"\n📋 МЕНЮ ПОЛЬЗОВАТЕЛЯ")
    print("=" * 25)
    print("1. Загрузить мои данные в систему")
    print("2. Начать анализ данных")
    print("3. Протестировать систему")
    print("4. Показать справку")
    print("5. Выход")

    choice = input(f"\nВыберите действие (1-5): ").strip()
    return choice


from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import uuid

def load_user_data():
    """Интерактивная загрузка данных без самовызыва CLI."""
    print(f"\n📂 ЗАГРУЗКА ПОЛЬЗОВАТЕЛЬСКИХ ДАННЫХ")
    print("=" * 40)

    print("Поддерживаемые форматы:")
    print("  • CSV файлы (.csv)")
    print("  • JSON файлы (.json)")
    print("  • SQLite базы данных (.db, .sqlite)")
    print("  • Текстовые файлы (.txt)")
    print("  • Директории с файлами")

    source_type = input("\nТип источника данных (csv/json/sqlite/directory): ").strip().lower()
    file_path   = input("Путь к данным: ").strip()
    collection  = input("Название коллекции в ChromaDB (по умолчанию 'documents'): ").strip() or "documents"
    clear_before = input("Очистить коллекцию перед загрузкой? (y/n): ").strip().lower() == 'y'

    path = Path(file_path)
    if not path.exists():
        print(f"❌ Файл или директория не найдены: {file_path}")
        return False

    # --- инициализация ChromaDB ---
    client = chromadb.PersistentClient(path="generated/chroma_db")
    from gpu_embed_global import gpu_embedding
    emb_fn = gpu_embedding
    try:
        client.delete_collection(name=collection)
        print("♻️ Старая коллекция удалена (из-за смены embedding-функции)")
    except Exception:
        pass
    coll = client.get_or_create_collection(name=collection, embedding_function=emb_fn)

    if clear_before:
        try:
            client.delete_collection(name=collection)
            coll = client.get_or_create_collection(name=collection, embedding_function=emb_fn)
            print("♻️ Коллекция очищена")
        except Exception:
            pass

    # --- чтение файла ---
    docs, meta = [], []
    try:
        if source_type == "csv":
            import chardet, io

            raw = path.read_bytes()  # читаем «как есть»
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines='skip')
            for _, row in df.iterrows():
                docs.append(" ".join(f"{k}: {v}" for k, v in row.items()))
                meta.append({"source": "csv", **row.to_dict()})
        elif source_type == "json":
            data = pd.read_json(path)
            if isinstance(data, pd.DataFrame):
                for _, row in data.iterrows():
                    docs.append(" ".join(f"{k}: {v}" for k, v in row.items()))
                    meta.append({"source": "json", **row.to_dict()})
            else:
                docs.append(str(data))
                meta.append({"source": "json", "data": str(data)})
        elif source_type == "sqlite":
            import sqlite3
            conn = sqlite3.connect(path)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            for tbl in tables["name"]:
                df = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 1000;", conn)
                for _, row in df.iterrows():
                    docs.append(" ".join(f"{k}: {v}" for k, v in row.items()))
                    meta.append({"source": "sqlite", "table": tbl, **row.to_dict()})
            conn.close()
        elif source_type == "txt":
            text = path.read_text(encoding="utf-8")
            docs.append(text)
            meta.append({"source": "txt", "file": path.name})
        elif source_type == "directory":
            for txt_file in path.rglob("*.txt"):
                docs.append(txt_file.read_text(encoding="utf-8"))
                meta.append({"source": "directory", "file": str(txt_file)})
        else:
            print("❌ Неизвестный тип источника")
            return False
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False

    # --- добавление в ChromaDB ---
    if not docs:
        print("❌ Документы не созданы – файл пустой или не удалось прочитать")
        return False

    ids = [str(uuid.uuid4()) for _ in docs]
    BATCH = 4_000  # безопасный размер
    for i in range(0, len(docs), BATCH):
        coll.add(
            documents=docs[i:i + BATCH],
            metadatas=meta[i:i + BATCH],
            ids=ids[i:i + BATCH]
        )
    print(f"✅ Загружено {len(docs)} документов в коллекцию '{collection}'")
    return True


def run_analysis():
    """Запускает анализ данных"""

    print(f"\n🔍 АНАЛИЗ ДАННЫХ")
    print("=" * 20)

    # Запуск основной системы
    try:
        from advanced_digital_twin_chroma import main as system_main
        asyncio.run(system_main())
        return True
    except Exception as e:
        print(f"❌ Ошибка запуска системы анализа: {e}")
        return False


def show_help():
    """Показывает справку"""

    help_text = """
🧠 ПРОДУКТИВНЫЙ ЦИФРОВОЙ АНАЛИТИК - СПРАВКА

ОПИСАНИЕ:
    Продвинутая система анализа данных с ChromaDB, динамическими сценариями
    и интеллектуальным Reasoning для работы с реальными данными.

ОСНОВНЫЕ ВОЗМОЖНОСТИ:
    • Работа с данными через ChromaDB (векторная база данных)
    • Динамическая генерация сценариев на основе пользовательских запросов
    • Продвинутое Reasoning с пошаговым планированием
    • Контекстуальное понимание с учетом истории запросов
    • Автоматическая валидация результатов
    • Интерактивные визуализации

ФАЙЛЫ СИСТЕМЫ:
    • advanced_digital_twin_chroma.py - Основная система
    • load_user_data.py - Загрузка пользовательских данных
    • test_system_chroma.py - Тестирование продукта
    • run_product.py - Этот скрипт запуска

ИСПОЛЬЗОВАНИЕ:
    1. Загрузите свои данные через опцию 'Загрузить мои данные'
    2. Начните анализ через 'Начать анализ данных'
    3. Или используйте прямые команды:
       python load_user_data.py --source csv --path data.csv --collection projects
       python advanced_digital_twin_chroma.py

ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ:
    • CSV файлы (.csv)
    • JSON файлы (.json)
    • SQLite базы данных (.db, .sqlite)
    • Текстовые файлы (.txt)
    • Директории с файлами

СТРУКТУРА ДАННЫХ:
    Данные хранятся в ChromaDB в следующих коллекциях:
    • projects - проекты и инициативы
    • documents - документы и отчеты
    • analytics - аналитические данные
    • risks - риски и проблемы
    • resources - ресурсы и персонал

ТИПЫ ЗАПРОСОВ:
    • analytics - стандартный анализ данных
    • scenario - генерация сценариев 'что если'
    • prediction - предиктивная аналитика
    • validation - проверка данных
    • explanation - объяснение результатов

ПРИМЕРЫ ЗАПРОСОВ:
    • 'Покажи проекты с высоким бюджетом'
    • 'Что если увеличить ресурсы на 30%?'
    • 'Какие риски есть в проектах?'
    • 'Сравни эффективность разных подходов'

ДЛЯ РАЗРАБОТЧИКОВ:
    • Модульная архитектура с агентами
    • Поддержка кастомных агентов
    • Расширяемая система плагинов
    • API для интеграции

ПОДДЕРЖКА:
    • Документация: README2.md
    • Тесты: test_system_chroma.py
    • Загрузка данных: load_user_data.py
    """

    print(help_text)


def main():
    """Главная функция для запуска продукта"""

    print("🚀 ЗАПУСК ПРОДУКТА ЦИФРОВОГО АНАЛИТИКА")
    print("=" * 50)

    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Продуктивный цифровой аналитик")
    parser.add_argument("--setup", action="store_true", help="Настройка окружения")
    parser.add_argument("--test", action="store_true", help="Тестирование продукта")
    parser.add_argument("--load-data", action="store_true", help="Загрузка данных")
    parser.add_argument("--analyze", action="store_true", help="Анализ данных")
    parser.add_argument("--help-product", action="store_true", help="Справка по продукту")
    parser.add_argument("--menu", action="store_true", help="Интерактивное меню")

    args = parser.parse_args()

    # Определение режима работы
    if args.setup:
        # Только настройка
        setup_environment()

    elif args.test:
        # Только тестирование
        if setup_environment():
            run_product_test()

    elif args.load_data:
        # Загрузка данных
        load_user_data()

    elif args.analyze:
        # Анализ данных
        run_analysis()

    elif args.help_product:
        # Справка
        show_help()

    elif args.menu:
        # Интерактивное меню
        while True:
            choice = show_user_menu()

            if choice == "1":
                load_user_data()
            elif choice == "2":
                run_analysis()
            elif choice == "3":
                if setup_environment():
                    run_product_test()
            elif choice == "4":
                show_help()
            elif choice == "5":
                print("👋 До свидания!")
                break
            else:
                print("❌ Неверный выбор")

    else:
        # По умолчанию - полный цикл
        print("Запуск полного цикла продукта...")

        if setup_environment():
            print(f"\n✅ Окружение настроено")

            # Предложение загрузить данные
            load_data_choice = input(f"\nХотите загрузить свои данные? (y/n): ").strip().lower()
            if load_data_choice == 'y':
                load_user_data()

            # Запуск тестирования
            if run_product_test():
                print(f"\n🎉 Продукт готов к использованию!")

                # Запуск анализа
                analyze_choice = input(f"\nНачать анализ данных? (y/n): ").strip().lower()
                if analyze_choice == 'y':
                    run_analysis()
            else:
                print(f"❌ Продукт требует доработки")


if __name__ == "__main__":
    main()