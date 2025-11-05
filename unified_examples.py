"""Примеры использования Construction AI Agent.

Этот файл демонстрирует различные способы использования единого агента
для выполнения разных задач.
"""

import logging
from unified_agent import ConstructionAIAgent, ConstructionAIAgentConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def example_1_simple_material_search():
    """Пример 1: Простой поиск материала."""
    print("\n" + "="*60)
    print("Пример 1: Простой поиск материала")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    # Поиск одного материала
    result = agent.find_material_price("Cement Portland")
    
    print(f"\n🔍 Материал: {result.material_name}")
    print(f"🇵🇹 Португальское название: {result.analysis.pt_name}")
    print(f"💰 Лучшая цена: {result.best_offer.price}")
    print(f"🏪 Поставщик: {result.best_offer.best_supplier}")
    print(f"🔗 Ссылка: {result.best_offer.url}")
    print(f"📝 Обоснование: {result.best_offer.reasoning}")


def example_2_batch_search():
    """Пример 2: Пакетный поиск материалов."""
    print("\n" + "="*60)
    print("Пример 2: Пакетный поиск материалов")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    materials = [
        "Cement Portland",
        "Ceramic tiles 30x30",
        "Sand 0-4mm"
    ]
    
    print(f"\n🔍 Ищем {len(materials)} материалов...")
    results = agent.find_materials_batch(materials, use_cache=True)
    
    print("\n📊 Результаты:")
    for result in results:
        print(f"\n  • {result.material_name}")
        print(f"    Цена: {result.best_offer.price}")
        print(f"    Поставщик: {result.best_offer.best_supplier}")
    
    # Вывод в Markdown
    print("\n📝 Markdown таблица:")
    markdown = agent.materials_to_markdown(results)
    print(markdown)


def example_3_universal_command():
    """Пример 3: Универсальный интерфейс."""
    print("\n" + "="*60)
    print("Пример 3: Универсальный интерфейс")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    commands = [
        "Найди цену на цемент в Португалии",
        "Сколько стоит керамическая плитка?",
    ]
    
    for cmd in commands:
        print(f"\n💬 Команда: {cmd}")
        result = agent.process_command(cmd)
        print(f"✅ Результат:\n{result}")


def example_4_google_sheets():
    """Пример 4: Работа с Google Sheets."""
    print("\n" + "="*60)
    print("Пример 4: Работа с Google Sheets")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    if not agent.sheets_ai:
        print("⚠️  Google Sheets не настроен. Пропускаем пример.")
        return
    
    # Чтение таблицы
    try:
        data = agent.read_sheet_data()
        print(f"\n📊 Прочитано строк: {len(data)}")
        if data:
            print(f"   Колонок: {len(data[0])}")
            print(f"   Первая строка: {data[0]}")
    except Exception as e:
        print(f"❌ Ошибка чтения: {e}")
    
    # Команда на естественном языке
    commands = [
        "Покажи информацию о таблице",
        "Прочитай первые 5 строк"
    ]
    
    for cmd in commands:
        print(f"\n💬 Команда: {cmd}")
        try:
            result = agent.process_sheets_command(cmd)
            print(f"✅ Результат:\n{result[:200]}...")  # Первые 200 символов
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def example_5_estimate_check():
    """Пример 5: Проверка строительной сметы."""
    print("\n" + "="*60)
    print("Пример 5: Проверка строительной сметы")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    if not agent.sheets_ai or not agent.sheets_ai.estimate_checker:
        print("⚠️  Estimate checker не настроен. Пропускаем пример.")
        return
    
    try:
        report = agent.check_estimate(
            estimate_sheet="Sheet1",
            master_sheet="Master List",
            quantity_col="F"
        )
        print("\n📋 Отчет о проверке:")
        print(report[:500])  # Первые 500 символов
    except Exception as e:
        print(f"❌ Ошибка проверки: {e}")


def example_6_cache_management():
    """Пример 6: Управление кэшем."""
    print("\n" + "="*60)
    print("Пример 6: Управление кэшем")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    # Получение статистики
    stats = agent.cache.get_stats()
    print("\n📊 Статистика кэша:")
    print(f"   Всего записей: {stats['total_materials']}")
    print(f"   Актуальных: {stats['fresh_materials']}")
    print(f"   Устаревших: {stats['expired_materials']}")
    
    # Поиск с кэшированием
    print("\n🔍 Первый поиск (будет закэширован):")
    result1 = agent.find_material_price("Cement Portland", use_cache=True)
    print(f"   Результат: {result1.best_offer.price}")
    
    print("\n🔍 Второй поиск (из кэша):")
    result2 = agent.find_material_price("Cement Portland", use_cache=True)
    print(f"   Результат: {result2.best_offer.price}")
    print("   ⚡ Мгновенный ответ из кэша!")
    
    # Обновленная статистика
    stats = agent.cache.get_stats()
    print(f"\n📊 Обновленная статистика:")
    print(f"   Всего записей: {stats['total_materials']}")


def example_7_advanced_search():
    """Пример 7: Продвинутый поиск с LangChain."""
    print("\n" + "="*60)
    print("Пример 7: Продвинутый поиск с LangChain")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    if not agent.advanced_agent:
        print("⚠️  Advanced agent не настроен. Пропускаем пример.")
        return
    
    print("\n🔍 Используем продвинутый поиск с веб-поиском...")
    result = agent.find_material_price(
        "Wood plywood 18mm",
        use_advanced_search=True
    )
    
    print(f"\n💰 Результат:")
    print(f"   Материал: {result.material_name}")
    print(f"   Цена: {result.best_offer.price}")
    print(f"   Поставщик: {result.best_offer.best_supplier}")


def example_8_custom_config():
    """Пример 8: Кастомная конфигурация."""
    print("\n" + "="*60)
    print("Пример 8: Кастомная конфигурация")
    print("="*60)
    
    # Создание кастомной конфигурации
    config = ConstructionAIAgentConfig(
        openai_api_key="your-key-here",  # Замените на реальный ключ
        llm_model="gpt-4o-mini",
        temperature=0.1,  # Более детерминированные ответы
        cache_ttl_seconds=3600,  # Кэш на 1 час
        enable_known_sites=True,
        known_sites_only=False,
    )
    
    print("\n⚙️  Конфигурация:")
    print(f"   Модель: {config.llm_model}")
    print(f"   Температура: {config.temperature}")
    print(f"   TTL кэша: {config.cache_ttl_seconds} секунд")
    print(f"   Известные сайты: {config.enable_known_sites}")
    
    # Создание агента с кастомной конфигурацией
    # agent = ConstructionAIAgent(config)
    print("\n✅ Агент можно создать с этой конфигурацией")


def example_9_error_handling():
    """Пример 9: Обработка ошибок."""
    print("\n" + "="*60)
    print("Пример 9: Обработка ошибок")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    # Попытка поиска несуществующего материала
    print("\n🔍 Поиск несуществующего материала...")
    try:
        result = agent.find_material_price("Nonexistent Material XYZ123")
        if result.best_offer.best_supplier == "N/A":
            print("⚠️  Материал не найден, но агент вернул альтернативу:")
            print(f"   {result.best_offer.reasoning}")
        else:
            print(f"✅ Найдено: {result.best_offer.best_supplier}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def example_10_stats():
    """Пример 10: Общая статистика агента."""
    print("\n" + "="*60)
    print("Пример 10: Общая статистика агента")
    print("="*60)
    
    agent = ConstructionAIAgent()
    
    stats = agent.get_stats()
    
    print("\n📊 Статистика агента:")
    print(f"\n🔧 Конфигурация:")
    print(f"   Модель: {stats['config']['model']}")
    print(f"   Кэш: {'Включен' if stats['config']['cache_enabled'] else 'Выключен'}")
    print(f"   Google Sheets: {'Включен' if stats['config']['sheets_enabled'] else 'Выключен'}")
    print(f"   Продвинутый поиск: {'Включен' if stats['config']['advanced_search_enabled'] else 'Выключен'}")
    
    print(f"\n💾 Кэш:")
    print(f"   Всего записей: {stats['cache']['total_materials']}")
    print(f"   Актуальных: {stats['cache']['fresh_materials']}")
    print(f"   Устаревших: {stats['cache']['expired_materials']}")


def main():
    """Запуск всех примеров."""
    print("\n" + "="*60)
    print("🚀 Construction AI Agent - Примеры использования")
    print("="*60)
    
    examples = [
        ("Простой поиск материала", example_1_simple_material_search),
        ("Пакетный поиск", example_2_batch_search),
        ("Универсальный интерфейс", example_3_universal_command),
        ("Google Sheets", example_4_google_sheets),
        ("Проверка смет", example_5_estimate_check),
        ("Управление кэшем", example_6_cache_management),
        ("Продвинутый поиск", example_7_advanced_search),
        ("Кастомная конфигурация", example_8_custom_config),
        ("Обработка ошибок", example_9_error_handling),
        ("Статистика", example_10_stats),
    ]
    
    print("\nДоступные примеры:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nВыберите пример (1-10) или 'all' для запуска всех:")
    choice = input("Ваш выбор: ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n❌ Ошибка в примере '{name}': {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        name, func = examples[idx]
        try:
            func()
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
    else:
        print("❌ Неверный выбор")
    
    print("\n" + "="*60)
    print("✅ Примеры завершены")
    print("="*60)


if __name__ == "__main__":
    main()
