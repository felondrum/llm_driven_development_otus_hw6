"""
Модуль для работы с Qdrant векторной базой данных
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
)
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)
from typing import List, Dict, Any, Optional
from src.config import (
    QDRANT_URL,
    COLLECTION_NAME,
    VECTOR_SIZE,
    DISTANCE_METRIC,
    HNSW_CONFIG,
)


class QdrantManager:
    """Менеджер для работы с Qdrant"""
    
    def __init__(self, url: str = QDRANT_URL):
        self.url = url
        self.client = None
    
    def connect(self) -> bool:
        """Подключение к Qdrant"""
        try:
            self.client = QdrantClient(url=self.url)
            # Проверка подключения
            collections = self.client.get_collections()
            print(f"✓ Подключение к Qdrant успешно. Коллекций: {len(collections.collections)}")
            return True
        except Exception as e:
            print(f"✗ Ошибка подключения к Qdrant: {e}")
            return False
    
    def create_collection(
        self,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
        distance: str = DISTANCE_METRIC,
        hnsw_config: Optional[Dict] = None,
        recreate: bool = False
    ) -> bool:
        """Создание коллекции с настройками HNSW"""
        
        try:
            # Проверка существования коллекции
            exists = self.client.collection_exists(collection_name=collection_name)
            
            if exists and recreate:
                print(f"Удаление существующей коллекции {collection_name}...")
                self.client.delete_collection(collection_name=collection_name)
                exists = False
            
            if exists:
                print(f"Коллекция {collection_name} уже существует")
                return True
            
            # Настройка distance metric
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            distance_metric = distance_map.get(distance, Distance.COSINE)
            
            # Параметры вектора
            vectors_config = VectorParams(
                size=vector_size,
                distance=distance_metric,
            )
            
            # HNSW конфигурация для оптимизации ANN поиска
            if hnsw_config is None:
                hnsw_config = HNSW_CONFIG
            
            hnsw = HnswConfigDiff(
                m=hnsw_config.get("m", 16),
                ef_construct=hnsw_config.get("ef_construct", 100),
                full_scan_threshold=hnsw_config.get("full_scan_threshold", 10000),
            )
            
            print(f"Создание коллекции {collection_name}...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw,
            )
            
            # Создание индекса для метаданных (для фильтрации)
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema="keyword",
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="doc_id",
                field_schema="integer",
            )
            
            print(f"✓ Коллекция {collection_name} создана с HNSW индексом")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка создания коллекции: {e}")
            return False
    
    def upsert_points(
        self,
        points: List[PointStruct],
        collection_name: str = COLLECTION_NAME,
        batch_size: int = 100
    ) -> bool:
        """Добавление точек в коллекцию (батчами)"""
        
        try:
            total = len(points)
            for i in range(0, total, batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                print(f"\rОбработано {min(i + batch_size, total)}/{total}", end='')
            
            print(f"\n✓ Добавлено {total} точек в коллекцию")
            return True
            
        except Exception as e:
            print(f"\n✗ Ошибка добавления точек: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        collection_name: str = COLLECTION_NAME,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        ef: Optional[int] = None,
    ) -> List[Dict]:
        """Поиск похожих векторов"""
        
        try:
            # Построение фильтра через HTTP модели
            search_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Параметры поиска
            search_params = None
            if ef:
                search_params = SearchParams(hnsw_ef=ef)
            
            # Поиск с использованием query_points (новый API Qdrant)
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
                params=search_params,
            )
            
            # Форматирование результатов
            formatted_results = []
            for result in results.points:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "vector": None,  # Не запрашиваем векторы для экономии памяти
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"✗ Ошибка поиска: {e}")
            return []
    
    def delete_collection(self, collection_name: str = COLLECTION_NAME) -> bool:
        """Удаление коллекции"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"✓ Коллекция {collection_name} удалена")
            return True
        except Exception as e:
            print(f"✗ Ошибка удаления коллекции: {e}")
            return False
    
    def get_collection_info(self, collection_name: str = COLLECTION_NAME) -> Optional[Dict]:
        """Получение информации о коллекции"""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            # Получаем количество векторов (совместимо с разными версиями Qdrant)
            vectors_count = getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0)
            points_count = getattr(info, 'points_count', 0)
            
            return {
                "vectors_count": vectors_count if vectors_count is not None else points_count,
                "points_count": points_count,
                "status": info.status,
                "config": {
                    "hnsw": info.config.hnsw_config if info.config else None,
                }
            }
        except Exception as e:
            print(f"✗ Ошибка получения информации: {e}")
            return None
    
    def close(self):
        """Закрытие соединения"""
        if self.client:
            self.client.close()
