"""Flask веб-приложение для Construction AI Agent.

Современный веб-интерфейс, объединяющий все возможности:
- Поиск цен на материалы
- Работа с Google Sheets
- Проверка строительных смет
- Веб-поиск и парсинг
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from unified_agent import ConstructionAIAgent, ConstructionAIAgentConfig

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Создание Flask приложения
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Инициализация агента
try:
    agent = ConstructionAIAgent()
    logger.info("Construction AI Agent initialized successfully")
except Exception as e:
    logger.error("Failed to initialize agent: %s", e)
    agent = None


@app.route('/')
def index():
    """Главная страница."""
    return send_from_directory('frontend', 'unified.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья сервиса."""
    return jsonify({
        'status': 'ok',
        'agent_initialized': agent is not None,
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получить статистику агента."""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        stats = agent.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error("Error getting stats: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/command', methods=['POST'])
def process_command():
    """
    Универсальный обработчик команд.
    
    Принимает команду на естественном языке и автоматически
    определяет, что нужно сделать.
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        command = data.get('command', '').strip()
        raw_context = data.get('context')
        context = raw_context if isinstance(raw_context, dict) else None
        
        if not command:
            return jsonify({'error': 'Command is required'}), 400
        
        logger.info("Processing command: %s", command)
        result = agent.process_command(command, context=context)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat(),
        })
    
    except Exception as e:
        logger.error("Error processing command: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/materials/search', methods=['POST'])
def search_material():
    """
    Поиск цены на материал.
    
    Body:
    {
        "material_name": "Cement Portland",
        "use_cache": true,
        "use_scraping": false,
        "use_advanced_search": false
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        material_name = data.get('material_name', '').strip()
        
        if not material_name:
            return jsonify({'error': 'material_name is required'}), 400
        
        use_cache = data.get('use_cache', True)
        use_scraping = data.get('use_scraping', False)
        use_advanced_search = data.get('use_advanced_search', False)
        timeout_seconds = float(data.get('timeout_seconds', 60.0))

        logger.info("Searching for material: %s", material_name)
        result = agent.find_material_price(
            material_name,
            use_cache=use_cache,
            use_scraping=use_scraping,
            use_advanced_search=use_advanced_search,
            timeout_seconds=timeout_seconds,
        )
        
        return jsonify({
            'success': True,
            'material': {
                'name': result.material_name,
                'pt_name': result.analysis.pt_name,
                'best_supplier': result.best_offer.best_supplier,
                'price': result.best_offer.price,
                'url': result.best_offer.url,
                'reasoning': result.best_offer.reasoning,
                'quotes_count': len(result.quotes),
                'quotes': [
                    {
                        'supplier': q.supplier,
                        'price': q.price,
                        'url': q.url,
                        'notes': q.notes,
                    }
                    for q in result.quotes
                ],
            },
            'timestamp': datetime.now().isoformat(),
        })
    
    except Exception as e:
        logger.error("Error searching material: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/project-chat', methods=['POST'])
def project_chat():
    """Пообщаться с проектным ассистентом."""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        message_raw = data.get('message', '')
        message = message_raw.strip() if isinstance(message_raw, str) else ''
        reset = bool(data.get('reset', False))
        extra_context_raw = data.get('extra_context')
        extra_context = (
            extra_context_raw.strip()
            if isinstance(extra_context_raw, str) and extra_context_raw.strip()
            else None
        )
        
        if not message and not reset:
            return jsonify({'error': 'message is required'}), 400
        
        try:
            reply = agent.chat_about_project(
                message,
                extra_context=extra_context,
                reset_history=reset,
            )
        except RuntimeError as exc:
            status_code = getattr(exc, 'status_code', 503)
            return jsonify({'error': str(exc)}), status_code
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        
        return jsonify({
            'success': True,
            'reply': reply,
            'reset': reset,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error("Error in project chat: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/materials/batch', methods=['POST'])
def search_materials_batch():
    """
    Поиск цен на несколько материалов.
    
    Body:
    {
        "materials": ["Cement", "Sand", "Gravel"],
        "use_cache": true,
        "use_scraping": false
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        materials = data.get('materials', [])
        
        if not materials:
            return jsonify({'error': 'materials list is required'}), 400
        
        use_cache = data.get('use_cache', True)
        use_scraping = data.get('use_scraping', False)
        timeout_seconds = float(data.get('timeout_seconds', 60.0))

        logger.info("Batch search for %d materials", len(materials))
        results = agent.find_materials_batch(
            materials,
            use_cache=use_cache,
            use_scraping=use_scraping,
            timeout_seconds=timeout_seconds,
        )
        
        # Конвертация в JSON
        materials_data = []
        for result in results:
            materials_data.append({
                'name': result.material_name,
                'pt_name': result.analysis.pt_name,
                'best_supplier': result.best_offer.best_supplier,
                'price': result.best_offer.price,
                'url': result.best_offer.url,
                'reasoning': result.best_offer.reasoning,
            })
        
        # Markdown таблица
        markdown = agent.materials_to_markdown(results)
        
        return jsonify({
            'success': True,
            'materials': materials_data,
            'markdown': markdown,
            'timestamp': datetime.now().isoformat(),
        })
    
    except Exception as e:
        logger.error("Error in batch search: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sheets/command', methods=['GET', 'POST'])
def sheets_command():
    """
    Выполнить команду для Google Sheets.
    
    Body:
    {
        "command": "Прочитай таблицу"
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        if request.method == 'GET':
            history = agent.get_sheets_chat_history()
            return jsonify({
                'success': True,
                'history': history,
                'timestamp': datetime.now().isoformat(),
            })

        data = request.get_json() or {}
        raw_command = data.get('command', '')
        command = raw_command.strip() if isinstance(raw_command, str) else ''
        reset = bool(data.get('reset', False))

        if not command and not reset:
            return jsonify({'error': 'command is required'}), 400

        logger.info("Processing sheets command: %s (reset=%s)", command, reset)

        if reset and not command:
            reply = agent.reset_sheets_chat()
        else:
            reply = agent.chat_about_sheets(command, reset_history=reset)

        history = agent.get_sheets_chat_history()
        
        return jsonify({
            'success': True,
            'reply': reply,
            'history': history,
            'timestamp': datetime.now().isoformat(),
        })
    
    except Exception as e:
        logger.error("Error processing sheets command: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sheets/read', methods=['GET'])
def read_sheet():
    """Прочитать данные из Google Sheets."""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = agent.read_sheet_data()
        return jsonify({
            'success': True,
            'data': data,
            'rows': len(data),
            'columns': len(data[0]) if data else 0,
        })
    except Exception as e:
        logger.error("Error reading sheet: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/sheets/write', methods=['POST'])
def write_sheet():
    """
    Записать данные в Google Sheets.
    
    Body:
    {
        "data": [["Header1", "Header2"], ["Row1Col1", "Row1Col2"]],
        "title": "Optional new title"
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        sheet_data = data.get('data', [])
        title = data.get('title')
        
        if not sheet_data:
            return jsonify({'error': 'data is required'}), 400
        
        agent.write_sheet_data(sheet_data, title)
        
        return jsonify({
            'success': True,
            'message': 'Data written successfully',
        })
    except Exception as e:
        logger.error("Error writing sheet: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/estimate/create', methods=['POST'])
def create_estimate():
    """
    Создать смету на основе текстового описания.

    Body:
    {
        "name": "Новый объект",
        "description": "Короткое описание",
        "client_name": "ООО Ромашка",
        "text_input": "Список работ и материалов",
        "auto_find_prices": true
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500

    try:
        data = request.get_json() or {}
        name = (data.get('name') or "").strip()
        description = data.get('description') or ""
        client_name = data.get('client_name') or ""
        text_input = data.get('text_input') or ""
        auto_find_prices = bool(data.get('auto_find_prices', True))

        logger.info("Creating estimate")
        estimate_data = agent.create_estimate(
            name=name,
            description=description,
            text_input=text_input,
            client_name=client_name,
            auto_find_prices=auto_find_prices,
        )

        return jsonify({
            'success': True,
            'estimate': estimate_data,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error("Error creating estimate: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/estimate/check', methods=['POST'])
def check_estimate():
    """
    Проверить строительную смету.
    
    Body:
    {
        "estimate_sheet": "Sheet1",
        "master_sheet": "Master List",
        "quantity_col": "F"
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        estimate_sheet = data.get('estimate_sheet')
        master_sheet = data.get('master_sheet', 'Master List')
        quantity_col = data.get('quantity_col', 'F')
        
        logger.info("Checking estimate")
        result = agent.check_estimate(estimate_sheet, master_sheet, quantity_col)
        
        return jsonify({
            'success': True,
            'report': result,
            'timestamp': datetime.now().isoformat(),
        })
    
    except Exception as e:
        logger.error("Error checking estimate: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/estimate/setup-constructor', methods=['POST'])
def setup_estimate_constructor():
    """Настроить лист конструктора сметы в Google Sheets.

    Body (optional):
    {
        "db_worksheet": "DB_Works",
        "calc_worksheet": "Estimate_Calculator"
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500

    try:
        data = request.get_json(silent=True) or {}
        db_worksheet = data.get('db_worksheet', 'DB_Works')
        calc_worksheet = data.get('calc_worksheet', 'Estimate_Calculator')

        logger.info(
            "Setting up estimate constructor (db=%s, calc=%s)",
            db_worksheet,
            calc_worksheet,
        )
        message = agent.setup_estimate_constructor(
            db_worksheet_name=db_worksheet,
            calc_sheet_name=calc_worksheet,
        )

        return jsonify({
            'success': True,
            'message': message,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error("Error setting up estimate constructor: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/web/search', methods=['POST'])
def web_search():
    """
    Выполнить веб-поиск.
    
    Body:
    {
        "query": "building materials Portugal"
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'query is required'}), 400
        
        logger.info("Web search: %s", query)
        results = agent.search_web(query)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
        })
    
    except Exception as e:
        logger.error("Error in web search: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/web/fetch', methods=['POST'])
def fetch_url():
    """
    Загрузить контент с URL.
    
    Body:
    {
        "url": "https://example.com"
    }
    """
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'url is required'}), 400
        
        logger.info("Fetching URL: %s", url)
        content = agent.fetch_web_content(url)
        
        return jsonify({
            'success': True,
            'content': content,
            'length': len(content),
        })
    
    except Exception as e:
        logger.error("Error fetching URL: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Получить статистику кэша."""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        stats = agent.cache.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error("Error getting cache stats: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Очистить кэш."""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        agent.cache.clear_all()
        return jsonify({
            'success': True,
            'message': 'Cache cleared',
        })
    except Exception as e:
        logger.error("Error clearing cache: %s", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8501))  # Порт 8501 (как Streamlit)
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info("Starting Construction AI Agent web server on port %d", port)
    app.run(host='0.0.0.0', port=port, debug=debug)
