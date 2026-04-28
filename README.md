# RAG System - Домашнее задание

Реализация RAG-системы с поиском по собственной базе документов на Python 3.12 с использованием:
- **Qdrant** - векторная база данных
- **Ollama** - локальные LLM модели (оптимизировано для Mac M1)
- **Hugging Face** - русскоязычный датасет

## 📋 Требования

- Docker и Docker Compose
- Python 3.12
- Mac M1 (или другая ARM архитектура)

## 🚀 Быстрый старт

### 1. Запуск сервисов (Docker)

```bash
docker-compose up -d
```

Это запустит:
- **Qdrant** на порту 6333 (HTTP) и 6334 (gRPC)
- **Ollama** на порту 11434

### 2. Загрузка моделей в Ollama

```bash
# Эмбеддинги (мультиязычная, поддерживает русский)
docker exec ollama_rag ollama pull nomic-embed-text

# Модель для генерации
docker exec ollama_rag ollama pull qwen2.5:3b
```

### 3. Установка зависимостей Python

```bash
uv sync
```

### 4. Запуск RAG системы

```bash
uv run main.py
```

## 📁 Структура проекта

```
llm_driven_development_otus_hw6/
├── docker-compose.yml          # Конфигурация Docker
├── requirements.txt            # Python зависимости
├── main.py                     # Главный скрипт
├── src/
│   ├── config.py              # Конфигурация системы
│   ├── qdrant_manager.py      # Работа с Qdrant
│   ├── ollama_client.py       # Работа с Ollama
│   ├── dataset_processor.py   # Обработка датасетов
│   └── rag_system.py          # Основная RAG логика
├── data/                       # Данные (игнорируются git)
├── qdrant_storage/            # Хранилище Qdrant (игнорируется git)
└── ollama_data/               # Модели Ollama (игнорируется git)
```

### Датасет

Используется **"dariaz/mineral_wiki_ru"** - статьи о минералах на русском языке:


## 📚 Выполненные задания

### Часть 1. Настройка и индексация ✓

- [x] Установка и настройка Qdrant через Docker
- [x] Изучение параметров конфигурации (HNSW)
- [x] Создание оптимальной схемы для индексации
- [x] Изучение ANN алгоритма HNSW:
  - `m` - количество связей на узел
  - `ef_construct` - размер кандидата при построении
  - `ef` - размер поиска при запросе

### Часть 2. Реализация поиска ✓

- [x] Семантический поиск по векторам
- [x] Similarity metrics: Cosine, Euclid, Dot Product (сравнение trade-offs)
- [x] Фильтрация по метаданным (source, doc_id)
- [x] Настройка параметров top-k
- [x] Оптимизация параметров поиска (ef значения)
- [x] Тестирование производительности на разных запросах
- [x] Сравнение точность vs скорость при разных настройках

### Бонусные задания ✓

- [x] Сравнение нескольких ANN-алгоритмов на одном датасете (HNSW, FLAT, IVF)
- [x] Hybrid search (векторный + фильтрация по метаданным)
- [x] Batch processing при индексации

### Дополнительно ✓

- [x] Изучение используемых алгоритмов и их параметров для разных БД
- [x] Понимание trade-offs между скоростью и точностью
- [x] Явные проверки для всех заданий

## 🔬 Trade-offs: Скорость vs Точность

| ef значение | Скорость | Точность | Рекомендация |
|------------|----------|----------|--------------|
| 32 | ⚡⚡⚡ Очень быстро | Низкая | Демо, тесты |
| 64 | ⚡⚡ Быстро | Средняя | Development |
| 128 | ⚡ Баланс | Хорошая | **Production** |
| 256 | Медленно | ⭐⭐⭐ Отличная | Research, анализ |

## 💡 Примеры использования

### Поиск документов

```python
from src.rag_system import RAGSystem

rag = RAGSystem()
rag.initialize()

results = rag.search("Что такое кварц?", top_k=5)
for r in results:
    print(f"Score: {r['score']}, Text: {r['payload']['text'][:100]}")
```

### Генерация ответа

```python
response = rag.generate_answer("Где находят алмазы?")
print(response['answer'])
print(f"Источников: {response['num_sources']}")
```

### Гибридный поиск с фильтром

```python
results = rag.hybrid_search(
    query="свойства минералов",
    filter_dict={"source": "dariaz/mineral_wiki_ru"},
    top_k=5
)
```

### Бенчмарк производительности

```python
benchmark = rag.benchmark_search(
    queries=["вопрос 1", "вопрос 2"],
    ef_values=[64, 128, 256]
)
```

## 🛠️ API Endpoints

### Qdrant
- Web UI: http://localhost:6333/dashboard
- REST API: http://localhost:6333

### Ollama
- API: http://localhost:11434/api

## 🧹 Очистка

```bash
# Остановить сервисы
docker-compose down

# Удалить данные (полный сброс)
docker-compose down -v
rm -rf qdrant_storage/ ollama_data/
```

## 📝 Примечания для Mac M1

1. **Память**: Ollama ограничен 8GB в docker-compose.yml
2. **Модели**: Выбраны легкие модели для комфортной работы
3. **Производительность**: Apple Silicon хорошо ускоряет inference

## 🔗 Полезные ссылки

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [HNSW Algorithm](https://github.com/nmslib/hnswlib)
- [Dataset на HuggingFace](https://huggingface.co/datasets/dariaz/mineral_wiki_ru)
