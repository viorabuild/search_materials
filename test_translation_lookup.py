#!/usr/bin/env python3
"""
Тест для проверки того, что translation_id правильно соответствуют переводам.
"""

def test_translation_lookup():
    """Проверить что translation_id совпадают при пропуске пустых описаний."""
    
    # Симулируем items с пропущенными пустыми описаниями
    items = []
    item_counter = 0
    for desc in ["", "Item 1", "", "Item 3", "", "Item 5"]:
        if desc:  # пропускаем пустые
            items.append({
                "translation_id": str(item_counter),
                "description": desc,
            })
            item_counter += 1
    
    # Переводы приходят с id: 0, 1, 2 (последовательные)
    translations = {
        "0": "Элемент 1",
        "1": "Элемент 3",
        "2": "Элемент 5",
    }
    
    print("Items:")
    for item in items:
        print(f"  {item}")
    
    print("\nTranslations:")
    for k, v in translations.items():
        print(f"  {k}: {v}")
    
    print("\nLookup results:")
    for item in items:
        item_id = item.get("translation_id", "")
        translation = translations.get(item_id, "NOT FOUND")
        print(f"  Item {item_id} ({item['description']}): {translation}")
        assert translation != "NOT FOUND", f"Translation not found for item {item_id}"
    
    print("\n✅ All translations found!")

if __name__ == "__main__":
    test_translation_lookup()
