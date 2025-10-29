#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для улучшения качества работы системы через замену/настройку модели.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_available_models():
    """Проверяет доступные модели в Ollama"""
    print("🔍 Проверка доступных моделей...")

    try:
        result = subprocess.run(['ollama', 'list'],
                                capture_output=True, text=True, check=True)

        print("📋 Доступные модели:")
        print(result.stdout)

        # Парсим список моделей
        models = []
        lines = result.stdout.strip().split('\n')[1:]  # Пропускаем заголовок
        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    models.append(parts[0])

        return models

    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при получении списка моделей: {e}")
        return []


def pull_model(model_name):
    """Загружает модель в Ollama"""
    print(f"📥 Загрузка модели {model_name}...")

    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        print(f"✅ Модель {model_name} успешно загружена")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при загрузке модели {model_name}: {e}")
        return False


def create_custom_model():
    """Создает кастомную модель с улучшенными параметрами"""

    modelfile_content = """
FROM llama2
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "###"
SYSTEM "" Ты - продвинутый аналитик данных. Твоя задача - помогать пользователям анализировать данные, генерировать инсайты и создавать отчеты.

Ты должен:
1. Тщательно анализировать запрос пользователя
2. Создавать эффективные SQL запросы для получения данных
3. Генерировать полезные инсайты на основе результатов
4. Предлагать практические рекомендации
5. Проводить валидацию результатов
6. Создавать сценарии "что если" при необходимости

Работай точно, внимательно и предоставляй только достоверную информацию.
"""
""

# Создаем файл модели
with open("generated/Modelfile", 'w', encoding='utf-8') as f:
    f.write(modelfile_content)

print("🔧 Создание кастомной модели...")

try:
    subprocess.run(['ollama', 'create', 'advanced_analyst', '-f', 'generated/Modelfile'],
                   check=True)
    print("✅ Кастомная модель создана успешно")
    return True
except subprocess.CalledProcessError as e:
    print(f"❌ Ошибка при создании модели: {e}")
    return False


def test_model_quality(model_name, test_queries):
    """Тестирует качество модели на простых запросах"""

    print(f"🧪 Тестирование модели {model_name}...")

    import ollama

    results = {}

    for query in test_queries:
        try:
            response = ollama.chat(model=model_name, messages=[
                {"role": "user", "content": query}
            ])

            results[query] = {
                "success": True,
                "response": response["message"]["content"][:200] + "..." if len(
                    response["message"]["content"]) > 200 else response["message"]["content"]
            }

        except Exception as e:
            results[query] = {
                "success": False,
                "error": str(e)
            }

    return results


def main():
    """Главная функция для улучшения модели"""

    print("🚀 УЛУЧШЕНИЕ КАЧЕСТВА МОДЕЛИ")
    print("=" * 50)

    # Проверяем доступные модели
    available_models = check_available_models()

    if not available_models:
        print("❌ Нет доступных моделей. Установите Ollama и загрузите модели.")
        return

    print(f"\nНайдено моделей: {len(available_models)}")

    # Рекомендуемые модели для аналитики
    recommended_models = [
        "llama2",
        "mistral",
        "codellama",
        "neural-chat",
        "starling-lm"
    ]

    print(f"\n📋 Рекомендуемые модели для аналитики:")
    for model in recommended_models:
        status = "✅ Установлена" if model in available_models else "❌ Не установлена"
        print(f"  • {model}: {status}")

    # Предлагаем варианты действий
    print(f"\n🔧 Варианты улучшения:")
    print("1. Загрузить рекомендуемую модель")
    print("2. Создать кастомную модель")
    print("3. Протестировать существующие модели")
    print("4. Настроить параметры текущей модели")

    choice = input(f"\nВыберите действие (1-4): ").strip()

    if choice == "1":
        # Загрузка модели
        print(f"\nДоступные для загрузки модели:")
        for i, model in enumerate(recommended_models, 1):
            if model not in available_models:
                print(f"{i}. {model}")

        model_choice = input(f"Выберите модель для загрузки: ").strip()

        if model_choice.isdigit():
            idx = int(model_choice) - 1
            if 0 <= idx < len(recommended_models):
                model_to_pull = recommended_models[idx]
                if model_to_pull not in available_models:
                    pull_model(model_to_pull)
                else:
                    print(f"Модель {model_to_pull} уже установлена")

    elif choice == "2":
        # Создание кастомной модели
        create_custom_model()

    elif choice == "3":
        # Тестирование моделей
        test_queries = [
            "Сколько будет 2+2?",
            "Напиши SQL запрос для получения всех пользователей",
            "Какие типы анализа данных ты можешь выполнить?"
        ]

        print(f"\nРезультаты тестирования:")
        for model in available_models[:3]:  # Тестируем первые 3 модели
            print(f"\n🧪 Модель: {model}")
            results = test_model_quality(model, test_queries)

            for query, result in results.items():
                if result["success"]:
                    print(f"  ✅ '{query[:30]}...': {result['response'][:50]}")
                else:
                    print(f"  ❌ '{query[:30]}...': Ошибка - {result['error']}")

    elif choice == "4":
        # Настройка параметров
        print(f"\nДля настройки параметров модели отредактируйте файл advanced_digital_twin.py")
        print(f"и измените значения LLM_MODEL и параметры в промптах.")

        print(f"\nТекущие параметры:")
        print(f"• Модель: {LLM_MODEL}")
        print(f"• Размер контекста: 4096 токенов")
        print(f"• Температура: 0.7")
        print(f"• Top-p: 0.9")

    else:
        print("❌ Неверный выбор")
        return

    print(f"\n✅ Процесс улучшения модели завершен!")
    print(f"\nСледующие шаги:")
    print(f"1. Перезапустите систему: python run_system.py --test")
    print(f"2. Проверьте результаты тестирования")
    print(f"3. Используйте улучшенную модель для анализа данных")


if __name__ == "__main__":
    main()