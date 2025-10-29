# 🧠 ПРОДУКТИВНЫЙ ЦИФРОВОЙ АНАЛИТИК

**Полноценная система анализа данных с ChromaDB, динамическими сценариями и продвинутым Reasoning**

## 📋 ОГЛАВЛЕНИЕ

- [🚀 Быстрый старт](#-быстрый-старт)
- [🏗️ Архитектура системы](#️-архитектура-системы)
- [📊 Потоки данных и запросов](#-потоки-данных-и-запросов)
- [🔧 Настройка под ваши данные](#-настройка-под-ваши-данные)
- [📚 API и компоненты](#-api-и-компоненты)
- [🎯 Использование](#-использование)
- [🔍 Примеры запросов](#-примеры-запросов)
- [🛠️ Расширение функциональности](#️-расширение-функциональности)
- [📈 Производительность](#-производительность)
- [🔒 Безопасность](#-безопасность)

## 🚀 БЫСТРЫЙ СТАРТ

### 1. Установка и запуск

```bash
# Клонирование репозитория
git clone <repository-url>
cd productive-digital-analyst

# Установка зависимостей
pip install -r requirements.txt

# Установка ChromaDB (если не установлено)
pip install chromadb

# Запуск продукта
python run_product.py --full
```

### 2. Загрузка ваших данных

```bash
# Загрузка CSV файла
python load_user_data.py --source csv --path data/my_projects.csv --collection projects

# Загрузка JSON файла  
python load_user_data.py --source json --path data/analytics_data.json --collection analytics

# Загрузка из SQLite
python load_user_data.py --source sqlite --path data/database.db --collection documents
```

### 3. Начало анализа

```bash
# Запуск системы анализа
python run_product.py --analyze

# Или через меню
python run_product.py --menu
```

## 🏗️ АРХИТЕКТУРА СИСТЕМЫ

### Основные компоненты

```
┌─────────────────────────────────────────────────────────────┐
│                    AdvancedDigitalTwin                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  ContextAgent   │  │ ReasoningAgent  │  │  DataAgent   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ValidationAgent  │  │DynamicScenario  │  │Explanation   │ │
│  └─────────────────┘  │     Agent       │  │   Agent      │ │
│                       └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ChromaDBManager                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  projects   │  │  documents  │  │      scenarios      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  analytics  │  │    risks    │  │      queries        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Модули и их назначение

#### Core System (`advanced_digital_twin_chroma.py`)
- **`AdvancedDigitalTwin`** - Главный класс системы
- **`ChromaDBManager`** - Управление ChromaDB и коллекциями
- **`Context`** - Контекст пользовательской сессии
- **`AnalysisResult`** - Результаты анализа
- **`ReasoningStep`** - Шаги в процессе рассуждений

#### Агенты (Модули специализации)

1. **ContextAgent** - Работа с контекстом и историей запросов
   - `extract_relevant_context()` - Извлечение релевантного контекста из ChromaDB
   - Анализ истории запросов пользователя

2. **ReasoningAgent** - Продвинутое планирование и рассуждение
   - `create_solution_plan()` - Создание плана решения для запроса
   - Определение стратегии анализа

3. **DataAgent** - Работа с данными через ChromaDB
   - `generate_chroma_query()` - Генерация оптимальных запросов
   - `execute_chroma_query()` - Выполнение запросов и получение данных

4. **ValidationAgent** - Валидация результатов
   - `validate_results()` - Проверка корректности результатов
   - Выявление аномалий и проверка сходства

5. **DynamicScenarioAgent** - Генерация динамических сценариев
   - `generate_dynamic_scenarios()` - Создание сценариев на основе запроса
   - `extract_scenario_elements()` - Анализ запроса на сценарные элементы

6. **ExplanationAgent** - Генерация инсайтов и объяснений
   - `generate_insights()` - Создание инсайтов на основе данных
   - Формирование рекомендаций

## 📊 ПОТОКИ ДАННЫХ И ЗАПРОСОВ

### Поток обработки запроса

```
Пользовательский запрос
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 1. AdvancedDigitalTwin.process_query()                  │
│    - Создание контекста                                 │
│    - Определение типа запроса                           │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 2. ContextAgent.extract_relevant_context()             │
│    - Поиск похожих запросов в ChromaDB                │
│    - Анализ истории пользователя                      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 3. ReasoningAgent.create_solution_plan()               │
│    - Анализ доступных коллекций ChromaDB              │
│    - Созданение плана решения через LLM               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 4. DataAgent.generate_chroma_query()                   │
│    - Оптимизация запроса для ChromaDB                 │
│    - Определение целевых коллекций                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 5. DataAgent.execute_chroma_query()                    │
│    - Выполнение запроса в ChromaDB                    │
│    - Получение результатов с метаданными              │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 6. ExplanationAgent.generate_insights()                │
│    - Анализ полученных данных                         │
│    - Генерация инсайтов и рекомендаций               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 7. ValidationAgent.validate_results()                  │
│    - Проверка качества результатов                    │
│    - Валидация логической консистентности            │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 8. DynamicScenarioAgent.generate_dynamic_scenarios()   │
│    - Генерация сценариев на основе запроса          │
│    - Сохранение сценариев в ChromaDB               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Результат анализа (AnalysisResult)
```

### Поток загрузки данных

```
Пользовательские данные
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 1. UserDataLoader.load_from_*()                        │
│    - Поддержка CSV, JSON, SQLite, директорий          │
│    - Автоматическое определение структуры             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Преобразование в формат ChromaDB                   │
│    - Создание контента из данных                      │
│    - Формирование метаданных                          │
│    - Генерация уникальных ID                          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 3. ChromaDBManager.add_documents()                     │
│    - Векторизация контента                           │
│    - Сохранение в указанную коллекцию                │
│    - Индексирование для быстрого поиска            │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Данные готовы к анализу
```

## 🔧 НАСТРОЙКА ПОД ВАШИ ДАННЫЕ

### 1. Подготовка данных

#### CSV файлы
```csv
project_id,project_name,type,budget,status,manager
PROJ_001,Северный жилой комплекс,строительство,500000000,в работе,Иванов
PROJ_002,CRM система,внедрение,1500000,завершен,Петрова
PROJ_003,Аудит безопасности,аудит,500000,планирование,Сидоров
```

#### JSON файлы
```json
[
  {
    "project_id": "PROJ_001",
    "project_name": "Северный жилой комплекс",
    "type": "строительство",
    "budget": 500000000,
    "status": "в работе",
    "manager": "Иванов",
    "risks": ["задержка поставок", "погодные условия"],
    "timeline": {
      "start": "2024-01-01",
      "planned_end": "2025-12-31"
    }
  }
]
```

#### SQLite база данных
```sql
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT,
    type TEXT,
    budget INTEGER,
    status TEXT,
    manager TEXT,
    start_date DATE,
    end_date DATE,
    risks TEXT
);
```

### 2. Загрузка данных

```bash
# CSV файл
python load_user_data.py --source csv --path data/projects.csv --collection projects

# JSON файл
python load_user_data.py --source json --path data/projects.json --collection projects

# SQLite база
python load_user_data.py --source sqlite --path data/database.db --collection projects

# Директория с файлами
python load_user_data.py --source directory --path data/files/ --collection documents
```

### 3. Настройка коллекций

Система использует следующие коллекции в ChromaDB:

- **`projects`** - проекты и инициативы
- **`documents`** - документы и отчеты
- **`analytics`** - аналитические данные и метрики
- **`risks`** - риски и проблемы
- **`resources`** - ресурсы и персонал

### 4. Кастомизация под ваши данные

#### Добавление новой коллекции

```python
# В UserDataLoader._initialize_collections()
self.collections['my_custom_data'] = self.client.get_or_create_collection(
    name='my_custom_data',
    embedding_function=self.embedding_function,
    metadata={"hnsw:space": "cosine", "description": "Мои кастомные данные"}
)
```

#### Настройка обработки данных

```python
# В load_from_csv() - настройка преобразования данных
content_columns = ['project_name', 'description']  # Столбцы для контента
metadata_columns = ['project_id', 'budget', 'status', 'manager']  # Метаданные
```

## 📚 API И КОМПОНЕНТЫ

### Основной API

```python
from advanced_digital_twin_chroma import AdvancedDigitalTwin, QueryType

# Инициализация системы
digital_twin = AdvancedDigitalTwin()

# Обработка запроса
result = await digital_twin.process_query(
    query="Покажи проекты с бюджетом выше 1 млн",
    query_type=QueryType.ANALYTICS,
    session_id="user_123"
)

# Доступ к результатам
print(f"Найдено: {len(result.data)} записей")
print(f"Инсайты: {result.insights}")
print(f"Рекомендации: {result.recommendations}")
print(f"Сценарии: {result.scenario_analysis}")
```

### Работа с ChromaDB

```python
from advanced_digital_twin_chroma import ChromaDBManager

# Инициализация
chroma_manager = ChromaDBManager()

# Добавление данных
chroma_manager.add_documents(
    documents=["Текст документа 1", "Текст документа 2"],
    metadatas=[{"type": "отчет"}, {"type": "анализ"}],
    ids=["doc_1", "doc_2"],
    collection_name="documents"
)

# Поиск данных
results = chroma_manager.query_documents(
    query_text="проекты строительства",
    n_results=10,
    collection_name="projects"
)
```

### Создание кастомного агента

```python
class CustomAgent:
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager
    
    async def analyze_custom_metric(self, query: str, context: Context) -> Dict:
        # Ваша логика анализа
        return {
            "metric": "custom_value",
            "analysis": "результат анализа"
        }

# Интеграция в систему
digital_twin.custom_agent = CustomAgent(digital_twin.chroma_manager)
```

## 🎯 ИСПОЛЬЗОВАНИЕ

### 1. Запуск через меню

```bash
python run_product.py --menu
```

### 2. Прямой запуск анализа

```bash
python run_product.py --analyze
```

### 3. Загрузка данных и анализ

```bash
# Загрузка данных
python load_user_data.py --source csv --path my_data.csv --collection projects

# Запуск анализа
python advanced_digital_twin_chroma.py
```

### 4. Тестирование системы

```bash
python run_product.py --test
```

## 🔍 ПРИМЕРЫ ЗАПРОСОВ

### Аналитические запросы

```
"Покажи все проекты с бюджетом выше 1 миллиона"
"Какие проекты просрочены по срокам?"
"Сравни эффективность разных типов проектов"
"Найди проекты с высокими рисками"
```

### Сценарные запросы

```
"Что если увеличить ресурсы на 30%?"
"Как сокращение сроков на 25% повлияет на качество?"
"Что если бюджет будет на 40% больше?"
"Какие будут последствия уменьшения команды на 20%?"
```

### Предиктивные запросы

```
"Предскажи сроки завершения проектов"
"Какие проекты рискуют не уложиться в бюджет?"
"Когда потребуется больше ресурсов?"
```

### Валидационные запросы

```
"Проверь корректность данных по проектам"
"Выяви аномалии в бюджетах"
"Проверь логическую консистентность данных"
```

## 🛠️ РАСШИРЕНИЕ ФУНКЦИОНАЛЬНОСТИ

### Добавление нового типа анализа

```python
# Создание нового типа запроса
class QueryType(Enum):
    ANALYTICS = "analytics"
    SCENARIO = "scenario" 
    PREDICTION = "prediction"
    VALIDATION = "validation"
    EXPLANATION = "explanation"
    CUSTOM_ANALYSIS = "custom_analysis"  # Новый тип

# Добавление обработки в AdvancedDigitalTwin
async def process_query(self, query: str, session_id: str = "default", 
                      query_type: QueryType = QueryType.ANALYTICS):
    
    if query_type == QueryType.CUSTOM_ANALYSIS:
        return await self._execute_custom_analysis(query, context)
```

### Расширение DataAgent

```python
class DataAgent:
    async def execute_custom_query(self, query: str) -> List[Dict]:
        # Кастомная логика получения данных
        return await self._get_custom_data(query)
```

### Интеграция внешних API

```python
class ExternalAPIAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def fetch_external_data(self, query: str) -> Dict:
        # Логика работы с внешним API
        pass
```

## 📈 ПРОИЗВОДИТЕЛЬНОСТЬ

### Оптимизации

1. **Кэширование результатов ChromaDB**
2. **Пакетная обработка запросов**
3. **Асинхронная обработка**
4. **Оптимизация векторизации**

### Мониторинг производительности

```python
# Добавление метрик производительности
import time

start_time = time.time()
result = await digital_twin.process_query(query)
processing_time = time.time() - start_time

logger.info(f"Query processed in {processing_time:.2f} seconds")
```

### Настройка ChromaDB для производительности

```python
# Оптимизация параметров ChromaDB
self.collections[name] = self.client.get_or_create_collection(
    name=name,
    embedding_function=self.embedding_function,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,  # Оптимизация построения индекса
        "hnsw:M": 16  # Оптимизация поиска
    }
)
```

## 🔒 БЕЗОПАСНОСТЬ

### Меры безопасности

1. **Изоляция данных пользователей**
2. **Валидация входных данных**
3. **Контроль доступа к данным**
4. **Логирование операций**

### Рекомендации по безопасности

```python
# Валидация пользовательского ввода
def validate_query(query: str) -> bool:
    # Проверка на SQL инъекции, XSS и другие уязвимости
    forbidden_patterns = ['<script', 'DROP TABLE', 'DELETE FROM']
    return not any(pattern in query.upper() for pattern in forbidden_patterns)

# Ограничение доступа к данным
def check_data_access(user_id: str, collection: str) -> bool:
    # Проверка прав доступа пользователя к коллекции
    return user_has_access(user_id, collection)
```

## 📋 ТРЕБОВАНИЯ К СИСТЕМЕ

### Минимальные требования
- Python 3.8+
- 4GB RAM
- 1GB свободного места на диске
- ChromaDB

### Рекомендуемые требования
- Python 3.9+
- 8GB RAM
- 10GB свободного места на диске
- GPU для ускорения эмбеддингов (опционально)

### Зависимости

```txt
# Основные библиотеки
chromadb>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
ollama>=0.1.0
scikit-learn>=1.3.0

# Визуализация
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Работа с данными
openpyxl>=3.1.0
xlrd>=2.0.0

# Асинхронная работа
asyncio-mqtt>=0.11.0

# Утилиты
python-dotenv>=1.0.0
tqdm>=4.65.0
```

## 🤝 ПОДДЕРЖКА И РАЗВИТИЕ

### Сообщество
- 📧 Email: support@digital-analyst.com
- 💬 Discord: [Productive Digital Analyst Community](https://discord.gg/productive-analyst)
- 🐛 Issues: [GitHub Issues](https://github.com/productive-analyst/system/issues)

### Документация
- 📖 Wiki: [Product Documentation](https://github.com/productive-analyst/system/wiki)
- 📚 API Reference: [API Documentation](https://github.com/productive-analyst/system/api)
- 🎓 Tutorials: [Video Tutorials](https://youtube.com/productive-analyst-tutorials)

### Вклад в развитие
1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Реализуйте изменения
4. Добавьте тесты
5. Отправьте pull request

## 📄 ЛИЦЕНЗИЯ

MIT License - см. файл [LICENSE](LICENSE) для деталей.

---

<div align="center">
    <p><strong>🧠 ПРОДУКТИВНЫЙ ЦИФРОВОЙ АНАЛИТИК</strong></p>
    <p>Система с интеллектуальным Reasoning, ChromaDB и динамическими сценариями</p>
    <p>⭐ Поддержите проект - поставьте звезду на GitHub!</p>
</div>