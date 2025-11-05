"""Database of known suppliers and typical prices for construction materials in Portugal.

This module provides fallback data when web scraping fails.
Prices are approximate and should be verified with suppliers.
"""

from typing import Dict, List, Optional
import re


class KnownSuppliersDB:
    """Database of known construction material suppliers in Portugal."""
    
    # Major suppliers in Portugal
    SUPPLIERS = {
        "Leroy Merlin": {
            "url": "https://www.leroymerlin.pt",
            "contact": "https://www.leroymerlin.pt/atendimento-cliente",
            "description": "Major home improvement and construction retailer"
        },
        "AKI": {
            "url": "https://www.aki.pt",
            "contact": "https://www.aki.pt/contactos",
            "description": "DIY and construction materials store"
        },
        "Bricomarche": {
            "url": "https://www.bricomarche.pt",
            "contact": "https://www.bricomarche.pt/contactos",
            "description": "Construction and DIY materials"
        },
        "Maxmat": {
            "url": "https://www.maxmat.pt",
            "contact": "https://www.maxmat.pt/contactos",
            "description": "Construction materials specialist"
        },
        "Sotecnisol": {
            "url": "https://www.sotecnisol.pt",
            "contact": "https://www.sotecnisol.pt/contactos",
            "description": "Building materials and solutions"
        },
    }
    
    # Typical price ranges for common materials (EUR)
    MATERIAL_PRICES = {
        # Cement and concrete
        "cement": {"price": "5-8 EUR/25kg", "suppliers": ["Leroy Merlin", "AKI", "Maxmat"]},
        "cimento": {"price": "5-8 EUR/25kg", "suppliers": ["Leroy Merlin", "AKI", "Maxmat"]},
        "concrete": {"price": "80-120 EUR/m³", "suppliers": ["Sotecnisol", "Maxmat"]},
        "betão": {"price": "80-120 EUR/m³", "suppliers": ["Sotecnisol", "Maxmat"]},
        
        # Steel and metal
        "steel rebar": {"price": "600-800 EUR/ton", "suppliers": ["Maxmat", "Sotecnisol"]},
        "vergalhão": {"price": "600-800 EUR/ton", "suppliers": ["Maxmat", "Sotecnisol"]},
        "steel beam": {"price": "800-1200 EUR/ton", "suppliers": ["Maxmat", "Sotecnisol"]},
        
        # Bricks and blocks
        "brick": {"price": "0.30-0.60 EUR/unit", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        "tijolo": {"price": "0.30-0.60 EUR/unit", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        "concrete block": {"price": "1.50-3.00 EUR/unit", "suppliers": ["Maxmat", "Sotecnisol"]},
        
        # Tiles
        "ceramic tile": {"price": "15-40 EUR/m²", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        "azulejo": {"price": "15-40 EUR/m²", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        
        # Wood
        "plywood": {"price": "20-50 EUR/sheet", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        "madeira": {"price": "20-50 EUR/sheet", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        
        # Sand and gravel
        "sand": {"price": "15-30 EUR/m³", "suppliers": ["Maxmat", "Sotecnisol"]},
        "areia": {"price": "15-30 EUR/m³", "suppliers": ["Maxmat", "Sotecnisol"]},
        "gravel": {"price": "20-35 EUR/m³", "suppliers": ["Maxmat", "Sotecnisol"]},
        "brita": {"price": "20-35 EUR/m³", "suppliers": ["Maxmat", "Sotecnisol"]},
        
        # Insulation
        "insulation": {"price": "5-15 EUR/m²", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
        "isolamento": {"price": "5-15 EUR/m²", "suppliers": ["Leroy Merlin", "AKI", "Bricomarche"]},
    }
    
    def find_material(self, query: str) -> Optional[Dict]:
        """Find material info by query (fuzzy matching)."""
        query_lower = query.lower()
        
        # Direct match
        if query_lower in self.MATERIAL_PRICES:
            return self._build_result(query_lower)
        
        # Fuzzy match - check if query contains material name
        for material_key in self.MATERIAL_PRICES:
            if material_key in query_lower or query_lower in material_key:
                return self._build_result(material_key)
        
        # Check for common variations
        variations = {
            "portland cement": "cement",
            "cimento portland": "cimento",
            "steel bar": "steel rebar",
            "rebar": "steel rebar",
            "ceramic tiles": "ceramic tile",
            "tiles": "ceramic tile",
        }
        
        for variation, material in variations.items():
            if variation in query_lower:
                return self._build_result(material)
        
        return None
    
    def _build_result(self, material_key: str) -> Dict:
        """Build result dict for a material."""
        material_info = self.MATERIAL_PRICES[material_key]
        
        result = {
            "material": material_key,
            "price": material_info["price"],
            "suppliers": []
        }
        
        for supplier_name in material_info["suppliers"]:
            if supplier_name in self.SUPPLIERS:
                supplier = self.SUPPLIERS[supplier_name]
                result["suppliers"].append({
                    "name": supplier_name,
                    "url": supplier["url"],
                    "contact": supplier["contact"],
                    "description": supplier["description"],
                    "price": material_info["price"],
                })
        
        return result
    
    def get_all_suppliers(self) -> List[Dict]:
        """Get list of all known suppliers."""
        return [
            {
                "name": name,
                **info
            }
            for name, info in self.SUPPLIERS.items()
        ]


# Singleton instance
_db = KnownSuppliersDB()


def find_material(query: str) -> Optional[Dict]:
    """Find material information from known database."""
    return _db.find_material(query)


def get_all_suppliers() -> List[Dict]:
    """Get all known suppliers."""
    return _db.get_all_suppliers()


# Example usage
if __name__ == "__main__":
    import json
    
    # Test queries
    queries = [
        "Cement Portland",
        "Steel rebar 12mm",
        "Ceramic tiles",
        "Cimento",
        "Vergalhão",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = find_material(query)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("  Not found in database")
