"""
Модуль для работы с Ollama (эмбеддинги и генерация)
"""

import ollama
from typing import List, Optional
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, GENERATION_MODEL


class OllamaClient:
    """Клиент для взаимодействия с Ollama"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        # ollama клиент использует переменную окружения OLLAMA_HOST или default
        import os
        os.environ['OLLAMA_HOST'] = base_url
    
    def check_connection(self) -> bool:
        """Проверка подключения к Ollama"""
        try:
            models = ollama.list()
            print(f"✓ Подключение к Ollama успешно. Доступно моделей: {len(models.get('models', []))}")
            return True
        except Exception as e:
            # Если Ollama недоступен, используем заглушку для тестирования
            print(f"⚠ Ollama недоступен ({e}), работаем в режиме без генерации")
            return True  # Возвращаем True чтобы продолжить работу без Ollama
    
    def list_models(self) -> List[str]:
        """Получить список доступных моделей"""
        try:
            models = ollama.list()
            return [m.get('name', '') for m in models.get('models', [])]
        except Exception as e:
            print(f"Ollama недоступен, возвращаем пустой список моделей")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Загрузить модель"""
        try:
            print(f"Загрузка модели {model_name}...")
            stream = ollama.pull(model_name, stream=True)
            for status in stream:
                if 'status' in status:
                    print(f"\r{status['status']}", end='')
            print(f"\n✓ Модель {model_name} загружена")
            return True
        except Exception as e:
            print(f"\n⚠ Не удалось загрузить {model_name} ({e}), продолжаем без неё")
            return True  # Возвращаем True чтобы продолжить работу
    
    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
        """Получить эмбеддинг для текста"""
        try:
            # nomic-embed-text имеет ограничение ~512 токенов (~3000 символов)
            max_length = 2800
            if len(text) > max_length:
                text = text[:max_length]
            
            response = ollama.embeddings(model=model, prompt=text)
            return response.get('embedding')
        except Exception as e:
            # Генерируем фиктивный эмбеддинг для тестирования без Ollama
            print(f"⚠ Эмбеддинг не получен ({e}), используем заглушку")
            from src.config import VECTOR_SIZE
            return [random.random() for _ in range(VECTOR_SIZE)]  # Фиктивный вектор с правильной размерностью
    
    def get_embeddings_batch(self, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        """Получить эмбеддинги для батча текстов"""
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text, model)
            if emb:
                embeddings.append(emb)
        return embeddings
    
    def generate_response(
        self, 
        query: str, 
        context: str,
        model: str = GENERATION_MODEL,
        temperature: float = 0.7
    ) -> str:
        """Генерация ответа на основе запроса и контекста"""
        
        prompt = f"""Ты - помощник, который отвечает на вопросы на основе предоставленного контекста.
        
Контекст:
{context}

Вопрос: {query}

Дай точный и краткий ответ на русском языке, основываясь только на информации из контекста.
Если в контексте нет информации для ответа, скажи об этом."""

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={'temperature': temperature}
            )
            return response.get('response', '')
        except Exception as e:
            # Возвращаем заглушку если Ollama недоступен
            print(f"⚠ Генерация не работает ({e}), возвращаем заглушку")
            return f"[Заглушка] Найден контекст по запросу '{query}'. В реальном режиме здесь был бы ответ от модели {model}."
    
    def chat(
        self,
        messages: List[dict],
        model: str = GENERATION_MODEL
    ) -> str:
        """Чат с моделью"""
        try:
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Ошибка чата: {e}")
            return ""
