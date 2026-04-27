"""
Конфигурация RAG системы
"""

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# Модель для эмбеддингов - multilingual, поддерживает русский
# all-minilm имеет ограничение контекста, используем меньшие чанки
EMBEDDING_MODEL = "all-minilm"  # 384 dimensions, ~80MB

# Модель для генерации ответов
GENERATION_MODEL = "qwen2.5:3b"  # 3B параметров, оптимизирована

# Qdrant конфигурация
QDRANT_URL = "http://localhost:6333"
QDRANT_GRPC_PORT = 6334

# Коллекция для векторов
COLLECTION_NAME = "rag_documents"

# Параметры вектора
VECTOR_SIZE = 384  # Размерность для all-minilm
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
    "chunk_size": 300,  # Размер чанка в символах (уменьшено с 500)
    "chunk_overlap": 30,  # Перекрытие между чанками (уменьшено с 50)
}

# Настройки для сравнения алгоритмов
BENCHMARK_CONFIG = {
    "num_queries": 50,
    "warmup_queries": 10,
    "test_ef_values": [32, 64, 128, 256],  # Для тестирования HNSW
}
