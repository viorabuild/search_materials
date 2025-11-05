#!/bin/bash

# Скрипт быстрого запуска Construction AI Agent
echo "🚀 Запуск Construction AI Agent..."
echo "================================"

# Определяем подходящую команду Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Python не найден в PATH. Установите Python 3.10+."
    exit 1
fi

# Проверка виртуального окружения
if [ ! -d ".venv" ]; then
    echo "⚠️  Виртуальное окружение не найдено. Создаем..."
    $PYTHON_BIN -m venv .venv
fi

# Активация виртуального окружения
echo "📦 Активация виртуального окружения..."
if [ -f ".venv/bin/activate" ]; then
    # Unix/macOS
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # Windows Git Bash / WSL
    source .venv/Scripts/activate
else
    echo "❌ Не удалось найти скрипт активации виртуального окружения."
    exit 1
fi

# Проверка зависимостей
echo "📚 Проверка зависимостей..."
"$PYTHON_BIN" -m pip install -q -r requirements.txt

# Проверка .env файла
if [ ! -f ".env" ]; then
    echo "⚠️  Файл .env не найден!"
    echo "Создайте .env файл на основе .env.example"
    echo ""
    echo "Минимальная конфигурация:"
    echo "OPENAI_API_KEY=your-api-key"
    exit 1
fi

# Загрузка переменных окружения
set -a
source .env
set +a

# Значение порта по умолчанию
PORT=${PORT:-8501}

# Создание директории для кэша
mkdir -p cache

# Запуск веб-приложения
echo ""
echo "✅ Запуск веб-сервера..."
echo "🌐 Откройте браузер: http://localhost:${PORT}"
echo ""
echo "Для остановки нажмите Ctrl+C"
echo "================================"
echo ""

if command -v waitress-serve >/dev/null 2>&1; then
    echo "🚀 Используем Waitress (production-ready WSGI сервер)"
    waitress-serve --listen=0.0.0.0:${PORT} unified_app:app
else
    echo "ℹ️ Waitress не найден, стартуем встроенный Flask (development) сервер"
    "$PYTHON_BIN" unified_app.py
fi
