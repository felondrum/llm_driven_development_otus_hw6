"""
Конфигурация RAG системы
Модели подобраны для Mac M1 (Apple Silicon)
"""

# Ollama модели (оптимизированы для Apple Silicon)
OLLAMA_BASE_URL = "http://localhost:11434"

# Модель для эмбеддингов - multilingual, поддерживает русский
# all-minilm легкая и быстрая, хорошо работает на M1
EMBEDDING_MODEL = "all-minilm"  # 384 dimensions, ~80MB

# Модель для генерации ответов - русскоязычная
# llama3.2 - легкая версия, хорошо работает на M1 с 8GB RAM
GENERATION_MODEL = "llama3.2"  # 3B параметров, оптимизирована

# Альтернативные модели (если нужно больше точности):
# GENERATION_MODEL_ALT = "mistral"  # 7B параметров
# GENERATION_MODEL_ALT = "qwen2.5"  # 7B, хорошая поддержка русского

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
# Используем "dariaz/mineral_wiki_ru" - статьи о минералах на русском
# или "t-tech/tnews" - новости на русском
DATASET_CONFIG = {
    "name": "dariaz/mineral_wiki_ru",  # Компактный датасет на русском
    "split": "train",
    "text_column": "text",
    "max_samples": 1000,  # Ограничение для тестирования
}

# Альтернативные русскоязычные датасеты:
# - "dariaz/mineral_wiki_ru" - Википедия о минералах (~1.5K статей)
# - "t-tech/tnews" - Новости на русском (большой)
# - "bigscience/xlsum" - Многоязычный, есть русский
# - "wikimedia/wikipedia" - Wikipedia 20231101.ru (очень большой)

# Параметры чанкинга (разбиения текста)
CHUNKING_CONFIG = {
    "chunk_size": 500,  # Размер чанка в символах
    "chunk_overlap": 50,  # Перекрытие между чанками
}

# Настройки для сравнения алгоритмов
BENCHMARK_CONFIG = {
    "num_queries": 50,
    "warmup_queries": 10,
    "test_ef_values": [32, 64, 128, 256],  # Для тестирования HNSW
}
