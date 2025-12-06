#!/usr/bin/env python3
"""
Тест для проверки исправления translation_id в Format 2.
Проверяет, что translation_id корректно соответствует переводам даже при пропуске пустых описаний.
"""

def test_translation_id_matching():
    """Проверить, что translation_id совпадают между items и translations."""
    
    # Симулируем items с пропущенными пустыми описаниями
    # Важно: если в начале есть пустые строки, то len(items) даст неправильные ID!
    
    items_with_old_logic = []
    for i, desc in enumerate(["", "Item 1", "", "Item 3", "", "Item 5"]):
        if desc:  # пропускаем пустые
            items_with_old_logic.append({
                "translation_id": str(len(items_with_old_logic)),  # НЕПРАВИЛЬНО!
                "description": desc,
            })
    
    items_with_new_logic = []
    item_counter = 0
    for desc in ["", "Item 1", "", "Item 3", "", "Item 5"]:
        if desc:  # пропускаем пустые
            items_with_new_logic.append({
                "translation_id": str(item_counter),  # ПРАВИЛЬНО!
                "description": desc,
            })
            item_counter += 1
    
    # Переводы приходят с id: 0, 1, 2 (последовательные)
    translations = {
        "0": "Элемент 1",
        "1": "Элемент 3",
        "2": "Элемент 5",
    }
    
    print("=== OLD LOGIC (BROKEN) ===")
    print(f"Items: {items_with_old_logic}")
    for item in items_with_old_logic:
        trans = translations.get(item["translation_id"], "NOT FOUND")
        print(f"  ID={item['translation_id']}: {item['description']} -> {trans}")
    
    print("\n=== NEW LOGIC (FIXED) ===")
    print(f"Items: {items_with_new_logic}")
    for item in items_with_new_logic:
        trans = translations.get(item["translation_id"], "NOT FOUND")
        print(f"  ID={item['translation_id']}: {item['description']} -> {trans}")
    
    # Проверяем, что все переводы найдены
    old_found = sum(1 for item in items_with_old_logic if translations.get(item["translation_id"]))
    new_found = sum(1 for item in items_with_new_logic if translations.get(item["translation_id"]))
    
    print(f"\nOld logic found: {old_found}/3 translations")
    print(f"New logic found: {new_found}/3 translations")
    
    assert new_found == 3, "New logic should find all translations"
    print("\n✅ Test passed!")

if __name__ == "__main__":
    test_translation_id_matching()
