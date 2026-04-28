"""
Конфигурация RAG системы
"""

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# Модель для эмбеддингов
EMBEDDING_MODEL = "nomic-embed-text"

# Модель для генерации ответов
GENERATION_MODEL = "qwen2.5:3b"

# Qdrant конфигурация
# Используем in-memory базу для тестирования без docker
# Для production замените на: QDRANT_URL = "http://localhost:6333"
QDRANT_URL = ":memory:"  # In-memory режим (не требует docker)
QDRANT_GRPC_PORT = None

# Коллекция для векторов
COLLECTION_NAME = "rag_documents"

# Параметры вектора
VECTOR_SIZE = 768  # Размерность для nomic-embed-text
DISTANCE_METRIC = "Cosine"  # Cosine similarity для семантического поиска

# Параметры поиска
TOP_K_DEFAULT = 5
SCORE_THRESHOLD = 0.3  # Минимальный порог схожести

# ANN алгоритмы в Qdrant:
# HNSW - быстрый, хорошая точность (по умолчанию)
# FLAT - точный, но медленный для больших данных
# Для сравнения можно тестировать разные параметры HNSW

# HNSW параметры для оптимизации
HNSW_CONFIG = {
    "m": 16,  # Количество связей на узел (больше = точнее, но медленнее)
    "ef_construct": 100,  # Размер кандидата при построении (больше = точнее)
    "full_scan_threshold": 10000,  # Порог для полного сканирования
}

# Параметры поиска для разных сценариев
SEARCH_CONFIGS = {
    "fast": {"ef": 64},      # Быстрый поиск, меньше точность
    "balanced": {"ef": 128}, # Баланс скорость/точность
    "accurate": {"ef": 256}, # Высокая точность, медленнее
}

# Датасет - русскоязычный
DATASET_CONFIG = {
    "name": "misterkirill/ru-wikipedia",
    "split": "train",
    "text_column": "text",
    "max_samples": 1000,
}


# Параметры чанкинга (разбиения текста)
# Уменьшаем размер чанков для all-minilm (имеет ограничение контекста ~256 токенов)
CHUNKING_CONFIG = {
    "chunk_size": 500,  # Размер чанка в символах
    "chunk_overlap": 50,  # Перекрытие между чанками # Перекрытие между чанками (уменьшено с 50)
}

# Настройки для сравнения алгоритмов
BENCHMARK_CONFIG = {
    "num_queries": 50,
    "warmup_queries": 10,
    "test_ef_values": [32, 64, 128, 256],  # Для тестирования HNSW
}

# Примечание: для работы с Ollama и Qdrant server необходимо:
# 1. Запустить docker-compose up -d
# 2. Изменить QDRANT_URL на "http://localhost:6333"
# 3. Убедиться что Ollama запущен и модели загружены
