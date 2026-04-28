# RAG System - Домашнее задание

Реализация RAG-системы с поиском по собственной базе документов на Python 3.12 с использованием:
- **Qdrant** - векторная база данных
- **Ollama** - локальные LLM модели (оптимизировано для Mac M1)
- **Hugging Face** - русскоязычный датасет

## 📋 Требования

- Docker и Docker Compose
- Python 3.12

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

Используется **"misterkirill/ru-wikipedia."** 


## 🔬 Trade-offs: Скорость vs Точность

| ef значение | Скорость | Точность | Рекомендация |
|------------|----------|----------|--------------|
| 32 | ⚡⚡⚡ Очень быстро | Низкая | Демо, тесты |
| 64 | ⚡⚡ Быстро | Средняя | Development |
| 128 | ⚡ Баланс | Хорошая | **Production** |
| 256 | Медленно | ⭐⭐⭐ Отличная | Research, анализ |


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
