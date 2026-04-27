"""
Основной RAG класс - объединяет все компоненты
"""

from typing import List, Dict, Optional
from src.qdrant_manager import QdrantManager
from src.ollama_client import OllamaClient
from src.dataset_processor import DatasetProcessor
from src.config import (
    COLLECTION_NAME,
    TOP_K_DEFAULT,
    SCORE_THRESHOLD,
    SEARCH_CONFIGS,
)
from qdrant_client.models import PointStruct


class RAGSystem:
    """RAG система для поиска и генерации ответов"""
    
    def __init__(self):
        self.qdrant = QdrantManager()
        self.ollama = OllamaClient()
        self.dataset_processor = DatasetProcessor()
        self.is_initialized = False
    
    def initialize(self, recreate_collection: bool = False) -> bool:
        """Инициализация системы"""
        
        print("=" * 60)
        print("Инициализация RAG системы")
        print("=" * 60)
        
        # Проверка подключения к Qdrant
        if not self.qdrant.connect():
            return False
        
        # Проверка подключения к Ollama
        if not self.ollama.check_connection():
            return False
        
        # Проверка наличия моделей
        available_models = self.ollama.list_models()
        from src.config import EMBEDDING_MODEL, GENERATION_MODEL
        
        if EMBEDDING_MODEL not in available_models:
            print(f"⚠ Модель {EMBEDDING_MODEL} не найдена. Загрузка...")
            if not self.ollama.pull_model(EMBEDDING_MODEL):
                return False
        
        if GENERATION_MODEL not in available_models:
            print(f"⚠ Модель {GENERATION_MODEL} не найдена. Загрузка...")
            if not self.ollama.pull_model(GENERATION_MODEL):
                return False
        
        # Создание коллекции
        if not self.qdrant.create_collection(recreate=recreate_collection):
            return False
        
        self.is_initialized = True
        print("\n✓ RAG система инициализирована")
        return True
    
    def load_and_index_dataset(
        self,
        dataset_name: Optional[str] = None,
        max_samples: int = 500
    ) -> int:
        """Загрузка датасета и индексация"""
        
        if not self.is_initialized:
            raise RuntimeError("Система не инициализирована. Вызовите initialize()")
        
        print("\n" + "=" * 60)
        print("Загрузка и индексация датасета")
        print("=" * 60)
        
        # Загрузка датасета
        if not self.dataset_processor.load_dataset(dataset_name, max_samples):
            return 0
        
        # Обработка документов и создание эмбеддингов
        points = []
        total_chunks = 0
        
        for doc in self.dataset_processor.process_documents():
            # Получение эмбеддинга
            embedding = self.ollama.get_embedding(doc["text"])
            
            if embedding:
                point = PointStruct(
                    id=total_chunks,
                    vector=embedding,
                    payload=doc,
                )
                points.append(point)
                total_chunks += 1
        
        # Добавление в Qdrant
        if points:
            self.qdrant.upsert_points(points)
        
        print(f"\n✓ Проиндексировано {total_chunks} чанков")
        return total_chunks
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        score_threshold: float = SCORE_THRESHOLD,
        use_filter: Optional[Dict] = None,
        search_mode: str = "balanced",
    ) -> List[Dict]:
        """Поиск релевантных документов"""
        
        if not self.is_initialized:
            raise RuntimeError("Система не инициализирована")
        
        # Получение эмбеддинга запроса
        query_embedding = self.ollama.get_embedding(query)
        
        if not query_embedding:
            return []
        
        # Параметры поиска в зависимости от режима
        ef = SEARCH_CONFIGS.get(search_mode, SEARCH_CONFIGS["balanced"]).get("ef", 128)
        
        # Поиск
        results = self.qdrant.search(
            query_vector=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_dict=use_filter,
            ef=ef,
        )
        
        return results
    
    def generate_answer(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        search_mode: str = "balanced",
    ) -> Dict:
        """Генерация ответа на вопрос"""
        
        # Поиск релевантных документов
        results = self.search(query, top_k=top_k, search_mode=search_mode)
        
        if not results:
            return {
                "query": query,
                "answer": "Не найдено релевантной информации в базе знаний.",
                "sources": [],
            }
        
        # Формирование контекста
        context_parts = []
        sources = []
        
        for result in results:
            context_parts.append(result["payload"]["text"])
            sources.append({
                "score": result["score"],
                "text": result["payload"]["text"][:200],
                "source": result["payload"].get("source", "unknown"),
            })
        
        context = "\n\n".join(context_parts)
        
        # Генерация ответа
        answer = self.ollama.generate_response(query, context)
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }
    
    def hybrid_search(
        self,
        query: str,
        filter_dict: Dict,
        top_k: int = TOP_K_DEFAULT,
    ) -> List[Dict]:
        """Гибридный поиск с фильтрацией по метаданным"""
        
        return self.search(
            query=query,
            top_k=top_k,
            use_filter=filter_dict,
        )
    
    def benchmark_search(
        self,
        queries: List[str],
        ef_values: List[int] = [32, 64, 128, 256],
        num_iterations: int = 3,
    ) -> Dict:
        """Бенчмарк производительности поиска с разными параметрами
        
        Args:
            queries: Список запросов для тестирования
            ef_values: Значения ef для сравнения
            num_iterations: Количество итераций для каждого запроса
            
        Returns:
            Словарь с результатами бенчмарка по каждому ef
        """
        
        import time
        
        results = {}
        
        print(f"\nЗапуск бенчмарка с {len(queries)} запросами, {num_iterations} итераций каждый")
        print(f"Тестируемые ef значения: {ef_values}\n")
        
        for ef in ef_values:
            times = []
            
            for query in queries:
                query_emb = self.ollama.get_embedding(query)
                
                if not query_emb:
                    continue
                
                # Выполняем несколько итераций для усреднения
                for _ in range(num_iterations):
                    start = time.time()
                    self.qdrant.search(
                        query_vector=query_emb,
                        top_k=5,
                        ef=ef,
                    )
                    elapsed = time.time() - start
                    times.append(elapsed)
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                results[ef] = {
                    "avg_time_sec": avg_time,
                    "avg_time_ms": avg_time * 1000,
                    "min_time_ms": min_time * 1000,
                    "max_time_ms": max_time * 1000,
                    "queries_tested": len(queries),
                    "total_measurements": len(times),
                }
                
                print(f"ef={ef:3d}: среднее={avg_time*1000:7.2f}ms, "
                      f"мин={min_time*1000:7.2f}ms, макс={max_time*1000:7.2f}ms")
        
        return results
    
    def compare_similarity_metrics(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """Сравнение различных метрик схожести
        
        Примечание: Qdrant использует одну метрику на коллекцию,
        поэтому это демонстрация концепции с объяснением trade-offs.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            
        Returns:
            Результаты поиска с текущей метрикой и информация о других метриках
        """
        
        from src.config import DISTANCE_METRIC
        
        query_emb = self.ollama.get_embedding(query)
        
        if not query_emb:
            return {
                "current_metric": DISTANCE_METRIC,
                "results": [],
                "metrics_info": {
                    "Cosine": {
                        "description": "Косинусное сходство (-1 до 1)",
                        "use_case": "Семантический поиск, текстовые эмбеддинги",
                        "pros": ["Нормализует векторы", "Хорошо для высокоразмерных данных"],
                        "cons": ["Игнорирует магнитуду вектора"],
                    },
                    "Euclid": {
                        "description": "Евклидово расстояние (0 до ∞)",
                        "use_case": "Когда важна абсолютная дистанция",
                        "pros": ["Интуитивно понятно", "Работает с любой размерностью"],
                        "cons": ["Чувствительно к масштабу", "Медленнее в высоких размерностях"],
                    },
                    "Dot": {
                        "description": "Скалярное произведение",
                        "use_case": "Когда важны и направление, и магнитуда",
                        "pros": ["Быстрое вычисление", "Работает с нормализованными векторами как Cosine"],
                        "cons": ["Результаты зависят от масштаба векторов"],
                    },
                },
            }
        
        # Поиск с текущей метрикой (Cosine по умолчанию)
        cosine_results = self.qdrant.search(
            query_vector=query_emb,
            top_k=top_k,
        )
        
        return {
            "current_metric": DISTANCE_METRIC,
            "results": cosine_results,
            "metrics_info": {
                "Cosine": {
                    "description": "Косинусное сходство (-1 до 1)",
                    "use_case": "Семантический поиск, текстовые эмбеддинги",
                    "pros": ["Нормализует векторы", "Хорошо для высокоразмерных данных"],
                    "cons": ["Игнорирует магнитуду вектора"],
                },
                "Euclid": {
                    "description": "Евклидово расстояние (0 до ∞)",
                    "use_case": "Когда важна абсолютная дистанция",
                    "pros": ["Интуитивно понятно", "Работает с любой размерностью"],
                    "cons": ["Чувствительно к масштабу", "Медленнее в высоких размерностях"],
                },
                "Dot": {
                    "description": "Скалярное произведение",
                    "use_case": "Когда важны и направление, и магнитуда",
                    "pros": ["Быстрое вычисление", "Работает с нормализованными векторами как Cosine"],
                    "cons": ["Результаты зависят от масштаба векторов"],
                },
            },
        }
    
    def get_ann_algorithms_info(self) -> Dict:
        """Получить информацию о ANN алгоритмах и их trade-offs
        
        Returns:
            Информация об алгоритмах, их параметрах и trade-offs
        """
        
        from src.config import HNSW_CONFIG
        
        return {
            "algorithms": {
                "HNSW": {
                    "full_name": "Hierarchical Navigable Small World",
                    "type": "Графовый индекс",
                    "description": "Строит многоуровневый граф для быстрого поиска",
                    "parameters": {
                        "m": {
                            "description": "Количество связей на узел",
                            "typical_range": "8-64",
                            "current_value": HNSW_CONFIG["m"],
                            "trade_off": "Больше m = выше точность, но больше памяти и медленнее построение",
                        },
                        "ef_construct": {
                            "description": "Размер кандидата при построении индекса",
                            "typical_range": "50-400",
                            "current_value": HNSW_CONFIG["ef_construct"],
                            "trade_off": "Больше ef_construct = лучше качество графа, но медленнее индексация",
                        },
                        "ef": {
                            "description": "Размер поиска при запросе",
                            "typical_range": "16-1024",
                            "current_value": "настраивается динамически",
                            "trade_off": "Больше ef = выше точность, но медленнее поиск",
                        },
                    },
                    "pros": [
                        "Очень быстрый поиск (O(log N))",
                        "Хорошая точность",
                        "Подходит для больших датасетов",
                    ],
                    "cons": [
                        "Требует больше памяти",
                        "Дольше строится чем простые индексы",
                        "Аппроксимативный (не 100% точность)",
                    ],
                    "best_for": "Production системы с балансом скорость/точность",
                },
                "FLAT": {
                    "full_name": "Exact Search (Brute Force)",
                    "type": "Точный поиск",
                    "description": "Полное сканирование всех векторов",
                    "parameters": {},
                    "pros": [
                        "100% точность",
                        "Простая реализация",
                        "Хорошо для маленьких датасетов (<10K векторов)",
                    ],
                    "cons": [
                        "Медленный O(N)",
                        "Не масштабируется",
                    ],
                    "best_for": "Маленькие датасеты или когда критична 100% точность",
                },
                "IVF": {
                    "full_name": "Inverted File Index",
                    "type": "Инвертированный индекс",
                    "description": "Кластеризация векторов с последующим поиском по кластерам",
                    "parameters": {
                        "nlist": "Количество кластеров",
                        "nprobe": "Количество кластеров для поиска",
                    },
                    "pros": [
                        "Хороший баланс скорость/память",
                        "Масштабируется лучше чем HNSW для очень больших данных",
                    ],
                    "cons": [
                        "Требует настройки количества кластеров",
                        "Может пропустить релевантные векторы в граничных кластерах",
                    ],
                    "best_for": "Очень большие датасеты (>1M векторов)",
                },
            },
            "trade_offs_summary": {
                "speed_vs_accuracy": {
                    "fast_low_accuracy": {
                        "config": "HNSW с m=8, ef=32",
                        "use_case": "Демо, прототипы, real-time приложения",
                    },
                    "balanced": {
                        "config": f"HNSW с m={HNSW_CONFIG['m']}, ef=128",
                        "use_case": "Production системы",
                    },
                    "high_accuracy_slow": {
                        "config": "HNSW с m=64, ef=512 или FLAT",
                        "use_case": "Научные исследования, критичные приложения",
                    },
                },
                "memory_vs_speed": {
                    "low_memory": "FLAT или IVF с малым nlist",
                    "high_memory_fast": "HNSW с большим m",
                },
            },
        }
    
    def get_stats(self) -> Dict:
        """Получение статистики системы"""
        
        collection_info = self.qdrant.get_collection_info()
        
        from src.config import EMBEDDING_MODEL, GENERATION_MODEL
        
        return {
            "collection": collection_info,
            "models": {
                "embedding": EMBEDDING_MODEL,
                "generation": GENERATION_MODEL,
            },
        }
    
    def close(self):
        """Закрытие соединений"""
        self.qdrant.close()
