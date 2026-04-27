"""
Модуль для загрузки и обработки датасетов
"""

from datasets import load_dataset
from typing import List, Dict, Generator
import hashlib
from src.config import DATASET_CONFIG, CHUNKING_CONFIG


class DatasetProcessor:
    """Обработчик датасетов из Hugging Face"""
    
    def __init__(self):
        self.dataset = None
        self.config = DATASET_CONFIG
    
    def load_dataset(self, dataset_name: str = None, max_samples: int = None) -> bool:
        """Загрузка датасета с Hugging Face"""
        
        name = dataset_name or self.config["name"]
        max_samples = max_samples or self.config.get("max_samples", 1000)
        
        try:
            print(f"Загрузка датасета {name}...")
            
            # Загрузка датасета
            dataset = load_dataset(name, split=self.config.get("split", "train"))
            
            # Ограничение количества образцов (для тестирования)
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                print(f"Выбрано {max_samples} образцов из {len(dataset)}")
            
            self.dataset = dataset
            print(f"✓ Датасет загружен. Всего записей: {len(dataset)}")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка загрузки датасета: {e}")
            return False
    
    def chunk_text(self, text: str) -> List[str]:
        """Разбиение текста на чанки"""
        
        chunk_size = CHUNKING_CONFIG["chunk_size"]
        chunk_overlap = CHUNKING_CONFIG["chunk_overlap"]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Если это не последний чанк, пытаемся разбить по предложению
            if end < len(text):
                # Ищем последнюю точку или перенос строки
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                split_point = max(last_period, last_newline)
                
                if split_point > chunk_size // 2:  # Если нашли разумную точку разбиения
                    chunk = chunk[:split_point + 1]
                    start = start + split_point + 1 - chunk_overlap
                else:
                    start = end - chunk_overlap
            else:
                start = end
            
            chunks.append(chunk.strip())
        
        return chunks
    
    def get_document_id(self, text: str) -> str:
        """Генерация уникального ID для документа"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def process_documents(self) -> Generator[Dict, None, None]:
        """Обработка документов и генерация чанков"""
        
        if not self.dataset:
            raise ValueError("Датасет не загружен")
        
        text_column = self.config.get("text_column", "text")
        doc_id = 0
        
        for item in self.dataset:
            text = item.get(text_column, "")
            
            if not text or len(text) < 50:  # Пропускаем слишком короткие тексты
                continue
            
            # Разбиение на чанки
            chunks = self.chunk_text(text)
            
            for chunk in chunks:
                if len(chunk) < 20:  # Пропускаем очень короткие чанки
                    continue
                
                yield {
                    "id": f"{doc_id}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}",
                    "doc_id": doc_id,
                    "text": chunk,
                    "source": self.config["name"],
                    "original_length": len(text),
                    "chunk_length": len(chunk),
                }
            
            doc_id += 1
            
            if doc_id % 100 == 0:
                print(f"\rОбработано документов: {doc_id}", end='')
        
        print(f"\n✓ Всего обработано документов: {doc_id}")
    
    def get_sample(self, n: int = 3) -> List[Dict]:
        """Получить несколько примеров из датасета"""
        
        if not self.dataset:
            return []
        
        samples = []
        text_column = self.config.get("text_column", "text")
        
        for i, item in enumerate(self.dataset):
            if i >= n:
                break
            samples.append({
                "text": item.get(text_column, "")[:200],
                "length": len(item.get(text_column, "")),
            })
        
        return samples


def load_russian_dataset():
    """
    Загрузка русскоязычного датасета
    Рекомендованные датасеты:
    1. dariaz/mineral_wiki_ru - статьи о минералах (компактный, ~1.5K)
    2. t-tech/tnews - новости на русском (большой)
    3. bigscience/xlsum - многоязычный с русским
    """
    
    processor = DatasetProcessor()
    
    # Пробуем загрузить компактный датасет о минералах
    success = processor.load_dataset("dariaz/mineral_wiki_ru", max_samples=500)
    
    if not success:
        print("Попробуем альтернативный датасет...")
        # Альтернатива - можно использовать любой текстовый датасет
        processor.config["name"] = "t-tech/tnews"
        success = processor.load_dataset(max_samples=500)
    
    return processor if success else None
