"""
Estimate Constructor - AI-powered construction estimate builder.

Этот модуль предоставляет функциональность для быстрого создания строительных смет:
- Создание и управление сметами
- ИИ-парсинг текстовых описаний работ
- Автоматический поиск цен на материалы
- Визуализация и экспорт в различные форматы
- Шаблоны типовых смет
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ItemType(Enum):
    """Тип позиции сметы"""
    MATERIAL = "material"  # Материал
    WORK = "work"  # Работа
    EQUIPMENT = "equipment"  # Оборудование
    SERVICE = "service"  # Услуга
    OTHER = "other"  # Другое


@dataclass
class EstimateItem:
    """Позиция в смете"""
    id: str
    name: str  # Название материала/работы
    description: str = ""  # Подробное описание
    item_type: ItemType = ItemType.MATERIAL
    quantity: float = 0.0  # Количество
    unit: str = "шт"  # Единица измерения (м², м³, шт, кг, л и т.д.)
    unit_price: float = 0.0  # Цена за единицу
    total_price: float = 0.0  # Общая стоимость (quantity * unit_price)
    code: str = ""  # Код материала/работы из базы
    supplier: str = ""  # Поставщик
    supplier_url: str = ""  # Ссылка на товар
    notes: str = ""  # Примечания
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Автоматический пересчет общей стоимости"""
        if self.total_price == 0.0 and self.quantity > 0 and self.unit_price > 0:
            self.total_price = round(self.quantity * self.unit_price, 2)
        if isinstance(self.item_type, str):
            self.item_type = ItemType(self.item_type)
    
    def update_total(self):
        """Пересчет общей стоимости"""
        self.total_price = round(self.quantity * self.unit_price, 2)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        data = asdict(self)
        data['item_type'] = self.item_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EstimateItem:
        """Создание из словаря"""
        if 'item_type' in data and isinstance(data['item_type'], str):
            data['item_type'] = ItemType(data['item_type'])
        return cls(**data)


@dataclass
class EstimateMetadata:
    """Метаданные сметы"""
    id: str
    name: str  # Название сметы
    description: str = ""  # Описание проекта
    client_name: str = ""  # Имя клиента
    client_contact: str = ""  # Контактные данные клиента
    project_address: str = ""  # Адрес объекта
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "draft"  # draft, approved, archived
    currency: str = "€"  # Валюта
    discount_percent: float = 0.0  # Скидка в процентах
    tax_percent: float = 0.0  # НДС в процентах
    notes: str = ""  # Общие примечания
    template_id: Optional[str] = None  # ID шаблона, если использовался
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EstimateMetadata:
        """Создание из словаря"""
        return cls(**data)


@dataclass
class Estimate:
    """Строительная смета"""
    metadata: EstimateMetadata
    items: List[EstimateItem] = field(default_factory=list)
    
    def add_item(self, item: EstimateItem) -> None:
        """Добавить позицию в смету"""
        self.items.append(item)
        self.metadata.updated_at = time.time()
    
    def remove_item(self, item_id: str) -> bool:
        """Удалить позицию из сметы"""
        original_len = len(self.items)
        self.items = [item for item in self.items if item.id != item_id]
        if len(self.items) < original_len:
            self.metadata.updated_at = time.time()
            return True
        return False
    
    def get_item(self, item_id: str) -> Optional[EstimateItem]:
        """Получить позицию по ID"""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def update_item(self, item_id: str, **kwargs) -> bool:
        """Обновить позицию"""
        item = self.get_item(item_id)
        if not item:
            return False
        
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        item.update_total()
        self.metadata.updated_at = time.time()
        return True
    
    def calculate_subtotal(self) -> float:
        """Рассчитать подитог (сумма всех позиций)"""
        return round(sum(item.total_price for item in self.items), 2)
    
    def calculate_discount(self) -> float:
        """Рассчитать скидку"""
        subtotal = self.calculate_subtotal()
        return round(subtotal * (self.metadata.discount_percent / 100), 2)
    
    def calculate_tax(self) -> float:
        """Рассчитать НДС"""
        subtotal = self.calculate_subtotal()
        discount = self.calculate_discount()
        taxable = subtotal - discount
        return round(taxable * (self.metadata.tax_percent / 100), 2)
    
    def calculate_total(self) -> float:
        """Рассчитать итоговую сумму"""
        subtotal = self.calculate_subtotal()
        discount = self.calculate_discount()
        tax = self.calculate_tax()
        return round(subtotal - discount + tax, 2)
    
    def get_summary(self) -> Dict[str, float]:
        """Получить сводку по смете"""
        return {
            'subtotal': self.calculate_subtotal(),
            'discount': self.calculate_discount(),
            'tax': self.calculate_tax(),
            'total': self.calculate_total(),
            'items_count': len(self.items),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'metadata': self.metadata.to_dict(),
            'items': [item.to_dict() for item in self.items],
            'summary': self.get_summary(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Estimate:
        """Создание из словаря"""
        metadata = EstimateMetadata.from_dict(data['metadata'])
        items = [EstimateItem.from_dict(item_data) for item_data in data.get('items', [])]
        return cls(metadata=metadata, items=items)


class EstimateConstructor:
    """
    AI-powered конструктор смет.
    
    Основная функциональность:
    - Создание новых смет
    - Парсинг текстовых описаний работ с помощью ИИ
    - Автоматический поиск цен на материалы
    - Сохранение и загрузка смет
    - Экспорт в различные форматы
    """
    
    def __init__(
        self,
        llm_client: Any,
        llm_model: str,
        material_agent: Optional[Any] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Инициализация конструктора смет.
        
        Args:
            llm_client: Клиент для работы с LLM
            llm_model: Модель LLM
            material_agent: Агент для поиска цен на материалы (MaterialPriceAgent)
            storage_path: Путь для сохранения смет
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.material_agent = material_agent
        self.storage_path = storage_path or Path("./data/estimates")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.estimates: Dict[str, Estimate] = {}
        self._load_estimates()
    
    def create_estimate(
        self,
        name: str,
        description: str = "",
        client_name: str = "",
        **kwargs
    ) -> Estimate:
        """
        Создать новую смету.
        
        Args:
            name: Название сметы
            description: Описание проекта
            client_name: Имя клиента
            **kwargs: Дополнительные параметры метаданных
        
        Returns:
            Созданная смета
        """
        estimate_id = str(uuid.uuid4())
        metadata = EstimateMetadata(
            id=estimate_id,
            name=name,
            description=description,
            client_name=client_name,
            **kwargs
        )
        estimate = Estimate(metadata=metadata)
        self.estimates[estimate_id] = estimate
        self._save_estimate(estimate)
        
        logger.info(f"Created new estimate: {name} (ID: {estimate_id})")
        return estimate
    
    def get_estimate(self, estimate_id: str) -> Optional[Estimate]:
        """Получить смету по ID"""
        return self.estimates.get(estimate_id)
    
    def list_estimates(self) -> List[Dict[str, Any]]:
        """Получить список всех смет"""
        return [
            {
                'id': est.metadata.id,
                'name': est.metadata.name,
                'client': est.metadata.client_name,
                'status': est.metadata.status,
                'items_count': len(est.items),
                'total': est.calculate_total(),
                'created_at': est.metadata.created_at,
                'updated_at': est.metadata.updated_at,
            }
            for est in self.estimates.values()
        ]
    
    def delete_estimate(self, estimate_id: str) -> bool:
        """Удалить смету"""
        if estimate_id in self.estimates:
            del self.estimates[estimate_id]
            estimate_path = self.storage_path / f"{estimate_id}.json"
            if estimate_path.exists():
                estimate_path.unlink()
            logger.info(f"Deleted estimate ID: {estimate_id}")
            return True
        return False
    
    async def parse_text_to_items(
        self,
        text: str,
        auto_find_prices: bool = True,
    ) -> List[EstimateItem]:
        """
        Парсинг текстового описания работ в позиции сметы с помощью ИИ.
        
        Args:
            text: Текстовое описание работ и материалов
            auto_find_prices: Автоматически искать цены на материалы
        
        Returns:
            Список позиций сметы
        """
        logger.info("Parsing text to estimate items using AI")
        
        prompt = f"""Ты - эксперт по строительным сметам. Проанализируй следующее описание работ и извлеки из него позиции сметы.

Для каждой позиции определи:
- name: Название материала или работы
- description: Подробное описание
- item_type: Тип (material, work, equipment, service, other)
- quantity: Количество (число)
- unit: Единица измерения (м², м³, шт, кг, л, м.п., компл.)

Описание работ:
{text}

Верни ответ СТРОГО в формате JSON (массив объектов):
[
  {{
    "name": "Название",
    "description": "Описание",
    "item_type": "material",
    "quantity": 10,
    "unit": "м²"
  }}
]

ВАЖНО: Верни только JSON, без дополнительного текста!"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Извлечение JSON из ответа (убираем markdown форматирование)
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                content = json_match.group(0)
            
            parsed_items = json.loads(content)
            
            items = []
            for item_data in parsed_items:
                item_id = str(uuid.uuid4())
                item = EstimateItem(
                    id=item_id,
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    item_type=ItemType(item_data.get('item_type', 'material')),
                    quantity=float(item_data.get('quantity', 0)),
                    unit=item_data.get('unit', 'шт'),
                )
                items.append(item)
            
            logger.info(f"Parsed {len(items)} items from text")
            
            # Автоматический поиск цен
            if auto_find_prices and self.material_agent and items:
                logger.info("Auto-finding prices for parsed items")
                await self._find_prices_for_items(items)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to parse text: {e}")
            raise
    
    async def _find_prices_for_items(self, items: List[EstimateItem]) -> None:
        """Поиск цен для списка позиций"""
        for item in items:
            if item.item_type == ItemType.MATERIAL:
                try:
                    result = await self.material_agent.find_material_price_async(
                        item.name,
                        exact_match_only=False,
                    )
                    
                    if result and result.best_offer:
                        item.unit_price = result.best_offer.price
                        item.supplier = result.best_offer.supplier
                        item.supplier_url = result.best_offer.url or ""
                        item.update_total()
                        logger.info(f"Found price for {item.name}: {item.unit_price} {item.unit}")
                
                except Exception as e:
                    logger.warning(f"Failed to find price for {item.name}: {e}")
    
    def add_items_to_estimate(
        self,
        estimate_id: str,
        items: List[EstimateItem],
    ) -> bool:
        """Добавить позиции в смету"""
        estimate = self.get_estimate(estimate_id)
        if not estimate:
            return False
        
        for item in items:
            estimate.add_item(item)
        
        self._save_estimate(estimate)
        logger.info(f"Added {len(items)} items to estimate {estimate_id}")
        return True
    
    def format_estimate_markdown(self, estimate: Estimate) -> str:
        """Форматирование сметы в Markdown"""
        lines = []
        
        # Заголовок
        lines.append(f"# {estimate.metadata.name}\n")
        
        # Метаданные
        if estimate.metadata.description:
            lines.append(f"**Описание:** {estimate.metadata.description}\n")
        if estimate.metadata.client_name:
            lines.append(f"**Клиент:** {estimate.metadata.client_name}")
        if estimate.metadata.project_address:
            lines.append(f"**Адрес объекта:** {estimate.metadata.project_address}")
        if estimate.metadata.client_contact:
            lines.append(f"**Контакт:** {estimate.metadata.client_contact}")
        
        lines.append(f"\n**Дата создания:** {datetime.fromtimestamp(estimate.metadata.created_at).strftime('%d.%m.%Y %H:%M')}")
        lines.append(f"**Статус:** {estimate.metadata.status}")
        lines.append("")
        
        # Таблица позиций
        lines.append("## Позиции сметы\n")
        lines.append("| № | Название | Тип | Количество | Ед. изм. | Цена за ед. | Сумма |")
        lines.append("|---|----------|-----|------------|----------|-------------|-------|")
        
        for idx, item in enumerate(estimate.items, 1):
            lines.append(
                f"| {idx} | {item.name} | {item.item_type.value} | {item.quantity} | "
                f"{item.unit} | {item.unit_price:.2f} {estimate.metadata.currency} | "
                f"{item.total_price:.2f} {estimate.metadata.currency} |"
            )
        
        lines.append("")
        
        # Итоги
        summary = estimate.get_summary()
        lines.append("## Итого\n")
        lines.append(f"**Подитог:** {summary['subtotal']:.2f} {estimate.metadata.currency}")
        
        if estimate.metadata.discount_percent > 0:
            lines.append(f"**Скидка ({estimate.metadata.discount_percent}%):** -{summary['discount']:.2f} {estimate.metadata.currency}")
        
        if estimate.metadata.tax_percent > 0:
            lines.append(f"**НДС ({estimate.metadata.tax_percent}%):** {summary['tax']:.2f} {estimate.metadata.currency}")
        
        lines.append(f"\n**ИТОГО:** {summary['total']:.2f} {estimate.metadata.currency}")
        
        if estimate.metadata.notes:
            lines.append(f"\n## Примечания\n\n{estimate.metadata.notes}")
        
        return "\n".join(lines)
    
    def export_to_csv(self, estimate: Estimate, output_path: Path) -> None:
        """Экспорт сметы в CSV"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Заголовки
            writer.writerow([
                'Название', 'Описание', 'Тип', 'Количество', 'Ед. изм.',
                'Цена за ед.', 'Сумма', 'Код', 'Поставщик', 'Примечания'
            ])
            
            # Позиции
            for item in estimate.items:
                writer.writerow([
                    item.name,
                    item.description,
                    item.item_type.value,
                    item.quantity,
                    item.unit,
                    item.unit_price,
                    item.total_price,
                    item.code,
                    item.supplier,
                    item.notes,
                ])
        
        logger.info(f"Exported estimate to CSV: {output_path}")
    
    def _save_estimate(self, estimate: Estimate) -> None:
        """Сохранить смету в файл"""
        estimate_path = self.storage_path / f"{estimate.metadata.id}.json"
        with open(estimate_path, 'w', encoding='utf-8') as f:
            json.dump(estimate.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _load_estimates(self) -> None:
        """Загрузить все сметы из хранилища"""
        if not self.storage_path.exists():
            return
        
        for estimate_file in self.storage_path.glob("*.json"):
            try:
                with open(estimate_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    estimate = Estimate.from_dict(data)
                    self.estimates[estimate.metadata.id] = estimate
            except Exception as e:
                logger.warning(f"Failed to load estimate from {estimate_file}: {e}")
        
        logger.info(f"Loaded {len(self.estimates)} estimates from storage")


def create_quick_estimate(
    constructor: EstimateConstructor,
    name: str,
    description: str,
    text_input: str,
    client_name: str = "",
    auto_find_prices: bool = True,
) -> Estimate:
    """
    Быстрое создание сметы из текстового описания.
    
    Args:
        constructor: Экземпляр EstimateConstructor
        name: Название сметы
        description: Описание проекта
        text_input: Текстовое описание работ и материалов
        client_name: Имя клиента
        auto_find_prices: Автоматически искать цены
    
    Returns:
        Созданная и заполненная смета
    """
    # Создаем смету
    estimate = constructor.create_estimate(
        name=name,
        description=description,
        client_name=client_name,
    )
    
    # Парсим текст и добавляем позиции
    import asyncio
    items = asyncio.run(constructor.parse_text_to_items(
        text_input,
        auto_find_prices=auto_find_prices,
    ))
    
    constructor.add_items_to_estimate(estimate.metadata.id, items)
    
    return estimate
