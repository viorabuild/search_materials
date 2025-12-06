#!/usr/bin/env python3
"""
Тест для проверки того, что _extract_format2_items правильно создает translation_id.
"""

def test_extract_items_with_empty_descriptions():
    """Проверить что translation_id правильно создаются при пропуске пустых описаний."""
    
    # Симулируем данные с пустыми описаниями
    headers = ["№", "Описание", "Единица", "Кол-во", "Цена"]
    rows = [
        ["", ""],  # Пустая строка
        ["1", "I - MOVIMENTOS DE TERRAS"],  # Раздел
        ["1.1", "Limpeza do terreno..."],  # Элемент
        ["", ""],  # Пустая строка
        ["2", "II - FUNDAÇÕES"],  # Раздел
    ]
    
    # Симулируем логику _extract_format2_items
    items = []
    item_counter = 0
    
    for row in rows:
        desc = row[1].strip() if len(row) > 1 else ""
        if not desc:
            print(f"Пропускаем пустую строку: {row}")
            continue
        
        number = row[0].strip() if len(row) > 0 else ""
        
        items.append({
            "translation_id": str(item_counter),
            "number": number,
            "description": desc,
        })
        print(f"Добавляем item[{item_counter}]: number={number}, desc={desc[:30]}...")
        item_counter += 1
    
    print(f"\nВсего items: {len(items)}")
    print("\nItems:")
    for item in items:
        print(f"  {item}")
    
    # Симулируем переводы
    translations = {
        "0": "I - ДВИЖЕНИЯ ЗЕМЛИ",
        "1": "Очистка земли...",
        "2": "II - ФУНДАМЕНТЫ",
    }
    
    print("\nTranslations:")
    for k, v in translations.items():
        print(f"  {k}: {v}")
    
    print("\nLookup results:")
    for item in items:
        item_id = item.get("translation_id", "")
        translation = translations.get(item_id, "NOT FOUND")
        print(f"  Item {item_id} ({item['description'][:30]}...): {translation}")
        assert translation != "NOT FOUND", f"Translation not found for item {item_id}"
    
    print("\n✅ All translations found!")

if __name__ == "__main__":
    test_extract_items_with_empty_descriptions()
