#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска продвинутой системы цифрового аналитика.
Обеспечивает полный цикл работы: от настройки до запуска интерфейса.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import argparse
import logging

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

def setup_environment():
    """Настраивает окружение для работы системы"""
    
    print("🚀 НАСТРОЙКА ПРОДВИНУТОЙ СИСТЕМЫ ЦИФРОВОГО АНАЛИТИКА")
    print("=" * 80)
    
    # Создание необходимых директорий
    directories = [
        "generated",
        "data/incoming",
        "data/processed",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана директория: {directory}")
    
    # Проверка наличия необходимых зависимостей
    print(f"\n📋 Проверка зависимостей...")
    
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "sqlalchemy",
        "ollama",
        "plotly",
        "sklearn",
        "matplotlib",
        "seaborn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Необходимо установить недостающие пакеты:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Проверка доступности Ollama
    print(f"\n🔌 Проверка Ollama...")
    try:
        import ollama
        # Проверяем доступность модели
        try:
            ollama.chat(model="digital_twin_analyst", messages=[{"role": "user", "content": "test"}])
            print("✅ Ollama и модель доступны")
        except Exception as e:
            print(f"❌ Ошибка доступа к модели: {e}")
            print("Убедитесь, что модель 'digital_twin_analyst' установлена и запущена")
            return False
    except ImportError:
        print("❌ Ollama не установлен")
        return False
    
    print(f"\n✅ Окружение настроено успешно!")
    return True

def generate_test_data():
    """Генерирует тестовые данные для демонстрации"""
    
    print(f"\n📊 ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
    print("=" * 40)
    
    try:
        # Импорт генератора данных
        from test_system_fixed import TestDataGenerator
        
        generator = TestDataGenerator(Path("generated/digital_twin_advanced.db"))
        success = generator.generate_construction_data()

        if success:
            print("✅ Тестовые данные сгенерированы успешно")
            return True
        else:
            print("❌ Ошибка генерации данных")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при генерации данных: {e}")
        return False

async def test_system():
    """Тестирует систему перед запуском"""
    
    print(f"\n🧪 ТЕСТИРОВАНИЕ СИСТЕМЫ")
    print("=" * 30)
    
    try:
        ProductTestSuite = _get_product_test_suite()
        ProductTestSuite = _get_product_test_suite()
        tester = ProductTestSuite()
        results = await tester.run_comprehensive_test()
        
        successful_tests = len([r for r in results if r.get("passed", False)])
        total_tests = len(results)
        
        print(f"\n📈 Результаты тестирования:")
        print(f"   Успешно: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        if successful_tests >= total_tests * 0.7:  # 70% проходной балл
            print("✅ Система готова к запуску")
            return True
        else:
            print("❌ Система требует доработки")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

def run_interface(mode="web"):
    """Запускает интерфейс системы"""
    
    print(f"\n🖥️  ЗАПУСК ИНТЕРФЕЙСА")
    print("=" * 25)
    
    if mode == "web":
        print("Запуск веб-интерфейса Streamlit...")
        
        try:
            # Запуск Streamlit приложения
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "advanced_interface.py",
                "--server.port=8501",
                "--server.address=localhost",
                "--theme.base=dark"
            ]
            
            print(f"Команда: {' '.join(cmd)}")
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Интерфейс остановлен пользователем")
        except Exception as e:
            print(f"❌ Ошибка запуска интерфейса: {e}")
            
    elif mode == "cli":
        print("Запуск командной строки...")
        
        try:
            from advanced_digital_twin_chroma import AdvancedDigitalTwin, QueryType
            
            digital_twin = AdvancedDigitalTwin()
            
            print("\n💬 Введите 'exit' для выхода")
            print("Примеры запросов:")
            print("  • Сколько длился вид работ 'X' для каждого уникального объекта?")
            print("  • Что если бы ресурсов было на 40% больше?")
            print("  • Какова средняя выработка в сутки?")
            
            while True:
                try:
                    query = input("\n📝 Ваш запрос: ").strip()
                    
                    if query.lower() in ['exit', 'quit', 'выход']:
                        print("👋 До свидания!")
                        break
                    
                    if not query:
                        continue
                    
                    # Определение типа запроса
                    query_type = QueryType.ANALYTICS
                    if "что если" in query.lower() or "если бы" in query.lower():
                        query_type = QueryType.SCENARIO
                    elif "предскаж" in query.lower() or "прогноз" in query.lower():
                        query_type = QueryType.PREDICTION
                    elif "провер" in query.lower() or "валидац" in query.lower():
                        query_type = QueryType.VALIDATION
                    
                    # Выполнение запроса
                    result = asyncio.run(digital_twin.process_query(query, query_type=query_type))
                    
                    # Вывод результатов
                    print(f"\n📊 РЕЗУЛЬТАТЫ:")
                    print(f"   Найдено записей: {len(result.data)}")
                    print(f"   Уровень уверенности: {result.confidence_score:.2f}")
                    
                    if result.insights:
                        print(f"\n💡 ИНСАЙТЫ:")
                        for insight in result.insights[:3]:
                            print(f"   • {insight}")
                    
                    if result.recommendations:
                        print(f"\n🎯 РЕКОМЕНДАЦИИ:")
                        for rec in result.recommendations[:2]:
                            print(f"   • {rec}")
                    
                    if result.scenario_analysis and result.scenario_analysis.get("scenarios"):
                        print(f"\n🔄 СЦЕНАРИИ:")
                        for scenario in result.scenario_analysis["scenarios"][:2]:
                            print(f"   • {scenario['name']}")
                
                except KeyboardInterrupt:
                    print(f"\n👋 До свидания!")
                    break
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
        
        except Exception as e:
            print(f"❌ Ошибка запуска CLI: {e}")

def show_help():
    """Показывает справку по использованию"""
    
    help_text = """
🧠 ПРОДВИНУТЫЙ ЦИФРОВОЙ АНАЛИТИК - СПРАВКА

ОПИСАНИЕ:
    Система с продвинутым Reasoning, контекстуальным пониманием и генерацией сценариев.

РЕЖИМЫ РАБОТЫ:
    --setup         Настройка окружения и генерация данных
    --test          Тестирование системы
    --web           Запуск веб-интерфейса (по умолчанию)
    --cli           Запуск командной строки
    --full          Полный цикл: настройка, тест, запуск

ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:
    python run_system.py --setup
    python run_system.py --test
    python run_system.py --web
    python run_system.py --cli
    python run_system.py --full

ОСОБЕННОСТИ СИСТЕМЫ:
    • Продвинутое Reasoning с пошаговым планированием
    • Контекстуальное понимание с учетом истории запросов
    • Генерация сценариев "что если" с анализом последствий
    • Автоматическая валидация результатов
    • Интерактивные визуализации данных
    • Экспорт результатов в JSON

ТРЕБОВАНИЯ:
    • Python 3.8+
    • Ollama с моделью 'digital_twin_analyst'
    • Зависимости из requirements.txt
    """
    
    print(help_text)

def main():
    """Главная функция"""

    # Парсинг аргументов командной строки
    # add_help=False отключает автоматическое добавление -h/--help
    parser = argparse.ArgumentParser(description="Продвинутый цифровой аналитик", add_help=False)
    parser.add_argument("--setup", action="store_true", help="Настройка окружения")
    parser.add_argument("--test", action="store_true", help="Тестирование системы")
    parser.add_argument("--web", action="store_true", help="Запуск веб-интерфейса")
    parser.add_argument("--cli", action="store_true", help="Запуск командной строки")
    parser.add_argument("--full", action="store_true", help="Полный цикл")
    # Теперь можно добавить свой --help
    parser.add_argument("--help", "-h", action="store_true", help="Показать справку")

    args = parser.parse_args()

    # Теперь проверка if args.help: будет работать
    if args.help:
        show_help()
        return # ВАЖНО: вернуться, не выполняя остальной код
    
    # Определение режима работы
    if args.full:
        # Полный цикл
        print("🔄 ЗАПУСК ПОЛНОГО ЦИКЛА")
        print("=" * 30)
        
        if not setup_environment():
            return
        
        if not generate_test_data():
            return
        
        if not asyncio.run(test_system()):
            return
        
        run_interface("web")
        
    elif args.setup:
        # Только настройка
        if setup_environment():
            generate_test_data()
    
    elif args.test:
        # Только тестирование
        if setup_environment():
            asyncio.run(test_system())
    
    elif args.cli:
        # Командная строка
        if setup_environment():
            run_interface("cli")
    
    elif args.web:
        # Веб-интерфейс
        if setup_environment():
            run_interface("web")
    
    else:
        # По умолчанию - веб-интерфейс
        print("Запуск веб-интерфейса (по умолчанию)")
        print("Используйте --help для просмотра всех опций")
        
        if setup_environment():
            # Проверяем наличие данных
            db_path = Path("generated/digital_twin_advanced.db")
            if not db_path.exists():
                print("Данные не найдены, генерирую тестовые...")
                generate_test_data()
            
            run_interface("web")

if __name__ == "__main__":
    main()