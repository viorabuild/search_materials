"""
Модуль для проверки строительных смет и работы с мастер-листами
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EstimateItem:
    """Позиция в смете"""
    row: int
    code: str
    description: str
    quantity: float
    status: str


@dataclass
class ValidationResult:
    """Результат проверки сметы"""
    total_items: int
    items_with_quantity: int
    items_without_code: int
    items_to_create: int
    items_list: List[EstimateItem]
    duplicates: List[str]
    missing_codes: List[str]


class EstimateChecker:
    """Класс для проверки смет и работы с мастер-листами"""
    
    CODE_PATTERNS = {
        'M-SHF': r'M-SHF-(\d+)',
        'M-WIN': r'M-WIN-(\d+)'
    }
    
    def __init__(self, sheets_ai):
        """
        Инициализация проверяльщика смет
        
        Args:
            sheets_ai: Экземпляр GoogleSheetsAI для работы с таблицами
        """
        self.sheets_ai = sheets_ai
    
    def parse_quantity(self, value: str) -> float:
        """Парсинг количества из строки"""
        if not value or value == '':
            return 0.0
        
        # Убираем пробелы и заменяем запятую на точку
        value = str(value).strip().replace(',', '.').replace(' ', '')
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def extract_code_number(self, code: str, prefix: str = 'M-SHF') -> Optional[int]:
        """Извлечение номера из кода"""
        pattern = self.CODE_PATTERNS.get(prefix)
        if not pattern:
            return None
        
        match = re.search(pattern, code)
        if match:
            return int(match.group(1))
        return None
    
    def get_max_code_number(self, codes: List[str], prefix: str = 'M-SHF') -> int:
        """Получение максимального номера кода"""
        max_num = 0
        for code in codes:
            num = self.extract_code_number(code, prefix)
            if num and num > max_num:
                max_num = num
        return max_num
    
    def generate_next_code(self, existing_codes: List[str], prefix: str = 'M-SHF') -> str:
        """Генерация следующего доступного кода"""
        max_num = self.get_max_code_number(existing_codes, prefix)
        next_num = max_num + 1
        return f"{prefix}-{next_num:03d}"
    
    def detect_quantity_column(self, estimate_data: List[List[str]]) -> str:
        """Автоопределение колонки количества по заголовкам и числовым значениям."""
        if not estimate_data:
            return "F"

        header = estimate_data[0]
        normalized = [str(h).strip().lower() for h in header]
        keywords = ("qty", "quantity", "кол", "кол-во", "qtd", "quantidade", "количество")
        for idx, text in enumerate(normalized):
            if any(key in text for key in keywords):
                return chr(ord('A') + idx)

        max_cols = max(len(r) for r in estimate_data)
        best_idx = 5  # default to column F if ничего не найдено
        best_score = -1

        for idx in range(max_cols):
            numeric_nonzero = 0
            for row in estimate_data[1:]:
                if len(row) <= idx:
                    continue
                val = row[idx]
                if val is None or str(val).strip() == "":
                    continue
                try:
                    if float(str(val).replace(",", ".").replace(" ", "")) != 0:
                        numeric_nonzero += 1
                except Exception:
                    continue
            if numeric_nonzero > best_score:
                best_score = numeric_nonzero
                best_idx = idx

        return chr(ord('A') + best_idx)

    def validate_estimate(
        self, 
        estimate_data: List[List[str]], 
        master_data: Optional[List[List[str]]],
        quantity_col: Optional[str] = 'F'
    ) -> ValidationResult:
        """
        Проверка сметы на соответствие мастер-листу
        
        Args:
            estimate_data: Данные сметы
            master_data: Данные мастер-листа (может быть пустым)
            quantity_col: Буква колонки с количеством (если None — автоопределение)
        
        Returns:
            ValidationResult с результатами проверки
        """
        master_data = master_data or []

        # Преобразуем букву колонки в индекс (A=0, B=1, ...)
        use_col = quantity_col or self.detect_quantity_column(estimate_data)
        qty_idx = max(0, ord(use_col.upper()) - ord('A'))
        
        # Извлекаем коды из мастер-листа (колонка A)
        master_codes = set()
        for row in master_data[1:]:  # Пропускаем заголовок
            if row and len(row) > 0 and row[0]:
                master_codes.add(row[0].strip())
        
        items_list = []
        items_with_quantity = 0
        items_without_code = 0
        items_to_create = 0
        duplicates = []
        missing_codes = []
        
        # Проверка дубликатов в смете
        estimate_codes = {}
        for idx, row in enumerate(estimate_data[1:], start=2):  # Начинаем со 2-й строки
            if not row or len(row) == 0:
                continue
            
            code = row[0].strip() if row[0] else ''
            
            # Проверка дубликатов
            if code and code in estimate_codes:
                if code not in duplicates:
                    duplicates.append(code)
            else:
                estimate_codes[code] = idx
            
            # Получаем количество
            quantity = 0.0
            if len(row) > qty_idx:
                quantity = self.parse_quantity(row[qty_idx])
            
            # Получаем описание (предполагаем колонка B)
            description = row[1] if len(row) > 1 else ''
            
            # Определяем статус
            status = self._determine_status(code, quantity, master_codes)
            
            item = EstimateItem(
                row=idx,
                code=code,
                description=description,
                quantity=quantity,
                status=status
            )
            items_list.append(item)
            
            # Подсчет статистики
            if quantity > 0:
                items_with_quantity += 1
                
                if not code:
                    items_without_code += 1
                    items_to_create += 1
                elif code not in master_codes:
                    missing_codes.append(f"{code} (строка {idx})")
        
        return ValidationResult(
            total_items=len(items_list),
            items_with_quantity=items_with_quantity,
            items_without_code=items_without_code,
            items_to_create=items_to_create,
            items_list=items_list,
            duplicates=duplicates,
            missing_codes=missing_codes
        )
    
    def _determine_status(self, code: str, quantity: float, master_codes: set) -> str:
        """Определение статуса позиции"""
        if quantity <= 0:
            return "⚪ Нет количества - не создавать"
        
        if not code:
            return "⚠ СОЗДАТЬ КОД"
        
        if code in master_codes:
            return "✓ Код OK"
        
        return "❌ Код не найден в мастер-листе"
    
    def create_validation_formulas(self, quantity_col: str = 'F') -> Dict[str, str]:
        """Генерация формул для проверки"""
        return {
            'status_check': f'=IF(OR(ISBLANK({quantity_col}2),{quantity_col}2=0),"⚪ Нет кол-ва",IF(ISBLANK(A2),"⚠ СОЗДАТЬ КОД",IF(COUNTIF(\'Master List\'!A:A,A2)>0,"✓ Код OK","❌ Код не найден")))',
            'next_shf_code': '="M-SHF-"&TEXT(MAX(ARRAYFORMULA(VALUE(REGEXEXTRACT(\'Master List\'!A:A,"M-SHF-(\\d+)"))))+1,"000")',
            'next_win_code': '="M-WIN-"&TEXT(MAX(ARRAYFORMULA(VALUE(REGEXEXTRACT(\'Master List\'!A:A,"M-WIN-(\\d+)"))))+1,"000")',
            'master_link': '=\'Master List\'!A$771',
            'duplicate_check': '=IF(COUNTIF(A:A,A2)>1,"⚠ ДУБЛИКАТ","")',
            'has_quantity': f'=AND({quantity_col}2>0,{quantity_col}2<>"")'
        }
    
    def format_validation_report(self, result: ValidationResult) -> str:
        """Форматирование отчета о проверке"""
        report = []
        
        report.append("=" * 80)
        report.append("📊 ОТЧЕТ О ПРОВЕРКЕ СМЕТЫ")
        report.append("=" * 80)
        report.append("")
        
        # Статистика
        report.append("📈 СТАТИСТИКА:")
        report.append(f"   • Всего позиций в смете: {result.total_items}")
        report.append(f"   • Позиций с количеством: {result.items_with_quantity}")
        report.append(f"   • Позиций без кода: {result.items_without_code}")
        report.append(f"   • Позиций для создания: {result.items_to_create}")
        report.append("")
        
        # Дубликаты
        if result.duplicates:
            report.append("⚠️  НАЙДЕНЫ ДУБЛИКАТЫ:")
            for dup in result.duplicates:
                report.append(f"   • {dup}")
            report.append("")
        
        # Отсутствующие коды
        if result.missing_codes:
            report.append("❌ КОДЫ НЕ НАЙДЕНЫ В МАСТЕР-ЛИСТЕ:")
            for code in result.missing_codes:
                report.append(f"   • {code}")
            report.append("")
        
        # Позиции для создания
        items_to_create = [item for item in result.items_list 
                          if item.status == "⚠ СОЗДАТЬ КОД"]
        
        if items_to_create:
            report.append("🔨 ПОЗИЦИИ ДЛЯ СОЗДАНИЯ В МАСТЕР-ЛИСТЕ:")
            report.append("")
            report.append(f"{'Строка':<8} | {'Описание':<40} | {'Кол-во':<10} | {'Статус'}")
            report.append("-" * 80)
            
            for item in items_to_create:
                desc = item.description[:37] + "..." if len(item.description) > 40 else item.description
                report.append(f"{item.row:<8} | {desc:<40} | {item.quantity:<10.2f} | {item.status}")
            report.append("")
        
        # Рекомендации
        report.append("💡 РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:")
        report.append("")
        report.append("1. Создайте вспомогательную колонку 'Статус' в смете")
        report.append("2. Используйте формулы для автоматической проверки")
        report.append("3. Создайте недостающие позиции в мастер-листе")
        report.append("4. Замените прямые коды на формулы-ссылки")
        report.append("")
        report.append("⚠️  ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ перед созданием позиций!")
        report.append("")
        
        return "\n".join(report)
    
    def generate_creation_plan(
        self, 
        items_to_create: List[EstimateItem],
        existing_codes: List[str],
        prefix: str = 'M-SHF'
    ) -> List[Dict[str, Any]]:
        """
        Генерация плана создания новых позиций
        
        Returns:
            Список словарей с информацией о создаваемых позициях
        """
        plan = []
        
        for item in items_to_create:
            new_code = self.generate_next_code(existing_codes, prefix)
            existing_codes.append(new_code)  # Добавляем для следующей итерации
            
            plan.append({
                'estimate_row': item.row,
                'new_code': new_code,
                'description': item.description,
                'quantity': item.quantity,
                'formula': f"='Master List'!A${{master_row}}"  # Будет заполнено после создания
            })
        
        return plan


def create_estimate_system_prompt(quantity_col: str = 'F') -> str:
    """Создание системного промпта для работы со сметами"""
    
    prompt = f"""Ты - помощник для работы с Google Sheets, специализирующийся на проверке 
строительных смет и мастер-листов.

СТРУКТУРА ДАННЫХ:
- Мастер-лист: содержит коды в колонке A (формат M-WIN-XXX, M-SHF-XXX)
- Смета: содержит те же коды в колонке A, связанные с мастер-листом
- Количество в смете: колонка {quantity_col}

ТВОИ ЗАДАЧИ:
1. Проверять соответствие кодов между сметой и мастер-листом
2. Находить отсутствующие или несовпадающие коды
3. Выявлять дубликаты кодов
4. Проверять корректность ссылок на мастер-лист
5. СОЗДАВАТЬ новые позиции в мастер-листе при необходимости

ПРАВИЛА СОЗДАНИЯ НОВЫХ ПОЗИЦИЙ:
✅ Создавать позицию ТОЛЬКО если:
   - В смете указано количество (колонка {quantity_col} не пустая и >0)
   - Код отсутствует в мастер-листе
   - Позиция имеет описание

❌ НЕ создавать позицию если:
   - Количество не указано или = 0
   - Код уже существует в мастер-листе
   - Отсутствует описание работы

АЛГОРИТМ СОЗДАНИЯ ПОЗИЦИИ:
1. Проверить наличие количества в смете
2. Проверить отсутствие кода в мастер-листе
3. Сгенерировать новый код по шаблону (M-SHF-XXX или M-WIN-XXX)
4. Создать запись в мастер-листе с кодом
5. В смете заменить код на формулу-ссылку на мастер-лист

ФОРМАТ КОДА:
- Код создается в мастер-листе
- В смете используется формула: ='Master List'!A[номер строки]
- Нумерация кодов: последовательная (найти максимальный номер + 1)

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА БЕЗОПАСНОСТИ:
❌ НИКОГДА не удаляй данные без явного разрешения пользователя
❌ НИКОГДА не перезаписывай существующие значения напрямую
❌ НИКОГДА не создавай позиции без проверки количества
✅ ВСЕГДА проверяй наличие количества перед созданием
✅ ВСЕГДА проверяй уникальность кода
✅ ВСЕГДА используй формулы для ссылок на мастер-лист
✅ ВСЕГДА создавай отдельные колонки для результатов проверки
✅ ВСЕГДА предлагай изменения, а не применяй их автоматически

МЕТОДОЛОГИЯ РАБОТЫ:
1. Создавай вспомогательные колонки для проверок
2. Используй формулы VLOOKUP, MATCH, COUNTIF для сверки данных
3. Выделяй проблемные строки цветом (условное форматирование)
4. Предоставляй отчет с найденными несоответствиями
5. Предлагай конкретные действия для исправления
6. Показывай список позиций, которые нужно создать
7. Запрашивай подтверждение перед созданием

ФОРМУЛЫ ДЛЯ ПРОВЕРКИ:
- Проверка наличия кода: =COUNTIF('Master List'!A:A, A2)>0
- Проверка количества: =AND({quantity_col}2>0, {quantity_col}2<>"")
- Проверка дубликатов: =COUNTIF(A:A, A2)>1
- Генерация следующего номера: =MAX(ARRAYFORMULA(VALUE(REGEXEXTRACT('Master List'!A:A,"M-SHF-(\\d+)"))))+1
- Ссылка на мастер-лист: ='Master List'!A[номер]

ФОРМАТ ОТВЕТА ПРИ ПРОВЕРКЕ:
1. Статистика:
   - Всего позиций в смете: X
   - Позиций с количеством: Y
   - Позиций без кода в мастер-листе: Z
   - Позиций для создания: N

2. Список позиций для создания:
   Строка | Описание | Количество | Предложенный код
   
3. Предложенные действия:
   - Формулы для вспомогательных колонок
   - Скрипт для создания позиций
   - Формулы для ссылок

4. Запрос подтверждения

ДОСТУПНЫЕ КОМАНДЫ:
- "Проверь смету" - анализ сметы и мастер-листа
- "Создай позиции" - создание недостающих позиций (с подтверждением)
- "Покажи формулы" - вывод формул для проверки
- "Найди дубликаты" - поиск дублирующихся кодов
"""
    
    return prompt
