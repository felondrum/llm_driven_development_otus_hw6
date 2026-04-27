"""
Основной скрипт для запуска RAG системы
Выполняет все шаги ДЗ: настройка, индексация, поиск, бенчмарк
"""

import sys
from src.rag_system import RAGSystem
from src.config import BENCHMARK_CONFIG


def main():
    """Главная функция"""
    
    print("\n" + "=" * 60)
    print("RAG СИСТЕМА - Домашнее задание")
    print("Векторная БД: Qdrant | Модели: Ollama (Mac M1)")
    print("=" * 60 + "\n")
    
    # Инициализация системы
    rag = RAGSystem()
    
    try:
        # Шаг 1: Инициализация
        if not rag.initialize(recreate_collection=True):
            print("\n✗ Не удалось инициализировать систему")
            print("Убедитесь, что:")
            print("  1. Запущен docker-compose (qdrant и ollama)")
            print("  2. Модели загружены в ollama")
            sys.exit(1)
        
        # Шаг 2: Загрузка и индексация датасета
        num_chunks = rag.load_and_index_dataset(
            dataset_name="dariaz/mineral_wiki_ru",
            max_samples=300  # Ограничим для быстрого теста
        )
        
        if num_chunks == 0:
            print("\n✗ Не удалось проиндексировать документы")
            sys.exit(1)
        
        # Шаг 3: Получение статистики
        print("\n" + "=" * 60)
        print("Статистика системы")
        print("=" * 60)
        stats = rag.get_stats()
        print(f"Векторов в базе: {stats['collection']['vectors_count']}")
        print(f"Модель эмбеддингов: {stats['models']['embedding']}")
        print(f"Модель генерации: {stats['models']['generation']}")
        
        # Шаг 4: Тестовый поиск
        print("\n" + "=" * 60)
        print("Тестовый поиск")
        print("=" * 60)
        
        test_queries = [
            "Что такое кварц?",
            "Какие свойства у гранита?",
            "Где находят алмазы?",
        ]
        
        for query in test_queries:
            print(f"\n📝 Запрос: {query}")
            results = rag.search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Текст: {result['payload']['text'][:100]}...")
        
        # Шаг 5: Генерация ответа
        print("\n" + "=" * 60)
        print("Генерация ответа RAG")
        print("=" * 60)
        
        query = "Что такое кварц и где он применяется?"
        print(f"\n📝 Запрос: {query}\n")
        
        response = rag.generate_answer(query, top_k=3)
        
        print(f"💬 Ответ:\n{response['answer']}")
        print(f"\n📚 Использовано источников: {response['num_sources']}")
        
        # Шаг 6: Бенчмарк с разными параметрами HNSW (ef)
        print("\n" + "=" * 60)
        print("Бенчмарк производительности (разные ef значения)")
        print("=" * 60)
        print("Сравнение скорости поиска при разных параметрах HNSW:\n")
        
        benchmark_results = rag.benchmark_search(
            queries=test_queries,
            ef_values=BENCHMARK_CONFIG["test_ef_values"],
        )
        
        # Анализ результатов
        print("\n" + "=" * 60)
        print("Анализ trade-offs скорость/точность")
        print("=" * 60)
        
        fastest_ef = min(benchmark_results.keys(), 
                        key=lambda x: benchmark_results[x]['avg_time_ms'])
        
        print(f"\n⚡ Самый быстрый: ef={fastest_ef} "
              f"({benchmark_results[fastest_ef]['avg_time_ms']:.2f}ms)")
        print(f"🎯 Самый точный: ef={max(benchmark_results.keys())} "
              f"(больше кандидатов = выше точность)")
        print("\nРекомендация:")
        print("  - Для production: ef=128 (баланс)")
        print("  - Для тестирования: ef=256 (макс. точность)")
        print("  - Для демо: ef=64 (макс. скорость)")
        
        # Шаг 7: Гибридный поиск (с фильтрацией)
        print("\n" + "=" * 60)
        print("Гибридный поиск с фильтрацией по метаданным")
        print("=" * 60)
        
        filter_query = "минералы"
        filter_dict = {"source": "dariaz/mineral_wiki_ru"}
        
        print(f"\n📝 Запрос: {filter_query}")
        print(f"🔍 Фильтр: {filter_dict}")
        
        hybrid_results = rag.hybrid_search(
            query=filter_query,
            filter_dict=filter_dict,
            top_k=3,
        )
        
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. Score: {result['score']:.3f}, Source: {result['payload']['source']}")
        
        print("\n" + "=" * 60)
        print("✓ Все задания выполнены успешно!")
        print("=" * 60)
        
        # Краткое резюме
        print("\n📋 РЕЗЮМЕ ВЫПОЛНЕННОГО:")
        print("  ✓ Часть 1: Настройка Qdrant с HNSW индексом")
        print("  ✓ Часть 1: Изучены параметры ANN (m, ef_construct, ef)")
        print("  ✓ Часть 2: Реализован семантический поиск")
        print("  ✓ Часть 2: Настроены similarity metrics (Cosine)")
        print("  ✓ Часть 2: Реализована фильтрация по метаданным")
        print("  ✓ Часть 2: Протестированы разные top-k")
        print("  ✓ Бонус: Сравнение ef параметров (скорость vs точность)")
        print("  ✓ Бонус: Hybrid search (вектор + фильтр)")
        print("  ✓ Бонус: Batch processing при индексации")
        
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        rag.close()


if __name__ == "__main__":
    main()
