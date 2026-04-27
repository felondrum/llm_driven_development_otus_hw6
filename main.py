import sys
from src.rag_system import RAGSystem
from src.config import BENCHMARK_CONFIG


def main():
    """Главная функция"""
    
    print("\n" + "=" * 60)
    print("RAG СИСТЕМА")
    print("Векторная БД: Qdrant | Модели: Ollama")
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
            dataset_name="misterkirill/ru-wikipedia",
            max_samples=None  # Загружаем весь датасет
        )
        
        if num_chunks == 0:
            print("\n✗ Не удалось проиндексировать документы")
            sys.exit(1)
        
        # Шаг 3: Получение статистики
        print("\n" + "=" * 60)
        print("Статистика системы")
        print("=" * 60)
        stats = rag.get_stats()
        if stats and stats.get('collection'):
            print(f"Векторов в базе: {stats['collection'].get('vectors_count', 'N/A')}")
        else:
            print("Векторов в базе: N/A")
        if stats and stats.get('models'):
            print(f"Модель эмбеддингов: {stats['models'].get('embedding', 'N/A')}")
            print(f"Модель генерации: {stats['models'].get('generation', 'N/A')}")
        else:
            print("Модели: N/A")
        
        # Шаг 4: Тестовый поиск
        print("\n" + "=" * 60)
        print("Тестовый поиск")
        print("=" * 60)
        
        test_queries = [
            "Что такое Адыгэ макъ (Голос адыга)?",
            "Где находится деревня Мухино?",
            "В какой день начался регулярный чемпионат начался НМХЛ в 2018 году?",
        ]
        
        for query in test_queries:
            print(f"\n📝 Запрос: {query}")
            results = rag.search(query, top_k=5)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Текст: {result['payload']['text'][:100]}...")
        
        # Шаг 5: Генерация ответа
        print("\n" + "=" * 60)
        print("Генерация ответа RAG")
        print("=" * 60)
        
        query = "Мухино"
        print(f"\n📝 Запрос: {query}\n")
        
        response = rag.generate_answer(query, top_k=6)
        
        print(f"💬 Ответ:\n{response['answer']}")
        print(f"\n📚 Использовано источников: {response.get('num_sources', len(response.get('sources', [])))}")
        
        # Шаг 6: Бенчмарк с разными параметрами HNSW (ef)
        print("\n" + "=" * 60)
        print("Бенчмарк производительности (разные ef значения)")
        print("=" * 60)
        print("Сравнение скорости поиска при разных параметрах HNSW:\n")
        
        benchmark_results = rag.benchmark_search(
            queries=test_queries,
            ef_values=BENCHMARK_CONFIG["test_ef_values"],
            num_iterations=3,
        )
        
        # Анализ результатов
        print("\n" + "=" * 60)
        print("Анализ trade-offs скорость/точность")
        print("=" * 60)
        
        if benchmark_results:
            fastest_ef = min(benchmark_results.keys(), 
                            key=lambda x: benchmark_results[x]['avg_time_ms'])
            
            print(f"\n⚡ Самый быстрый: ef={fastest_ef} "
                  f"({benchmark_results[fastest_ef]['avg_time_ms']:.2f}ms)")
            print(f"🎯 Самый точный: ef={max(benchmark_results.keys())} "
                  f"(больше кандидатов = выше точность)")
        
        # Шаг 6b: Информация о метриках схожести
        print("\n" + "=" * 60)
        print("Метрики схожести (Similarity Metrics)")
        print("=" * 60)
        
        metrics_info = rag.compare_similarity_metrics(
            query="Что такое Адыгэ макъ?",
            top_k=3,
        )
        
        print(f"\nТекущая метрика: {metrics_info.get('current_metric', 'N/A')}")
        print(f"Результатов найдено: {len(metrics_info.get('results', []))}")
        
        # Шаг 7: Гибридный поиск (с фильтрацией)
        print("\n" + "=" * 60)
        print("Гибридный поиск с фильтрацией по метаданным")
        print("=" * 60)
        
        filter_query = "адыгея"
        filter_dict = {"source": "misterkirill/ru-wikipedia"}
        
        print(f"\n📝 Запрос: {filter_query}")
        print(f"🔍 Фильтр: {filter_dict}")
        
        hybrid_results = rag.hybrid_search(
            query=filter_query,
            filter_dict=filter_dict,
            top_k=3,
        )
        
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. Score: {result['score']:.3f}, Source: {result['payload']['source']}")
        
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
