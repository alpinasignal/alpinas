# 🚀 Alpina Signal - AI Система Предсказаний Крипторынка

Профессиональная AI-система для предсказания движения криптовалют с использованием настоящих нейронных сетей.

## 🎯 Что это?

**Alpina Signal** - это серьезная количественная ML система, которая использует **настоящие нейронные сети** для генерации вероятностных прогнозов криптовалютных рынков.

Это **НЕ** простой бот с индикаторами. Это **production-grade AI система** для платной подписки.

### Ключевые особенности

✅ **Настоящие нейронные сети** (Transformer/LSTM) обученные на 3 годах данных
✅ **Вероятностные предсказания** - не фейковая уверенность
✅ **Профессиональный подход** - walk-forward validation, без утечки данных
✅ **15 ликвидных пар** - BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK, DOT, MATIC, LTC, OP, ARB, DOGE, TRX
✅ **3 таймфрейма** - 15m, 1h, 4h
✅ **Telegram Mini App** - чистый UI, весь AI на бэкенде
✅ **Система подписок** - бесплатный и платные тарифы

## 🏗️ Архитектура

```
Binance (публичный API)
        ↓
Feature Engineering (30+ фичей)
        ↓
Transformer Neural Network
        ↓
Вероятности [Up, Down, Flat]
        ↓
Фильтр волатильности + пороги
        ↓
Сигнал: LONG / SHORT / NO TRADE
        ↓
REST API (FastAPI)
        ↓
Telegram Mini App (только UI)
```

## ⚡ Быстрый старт

### Вариант 1: Одной командой (Windows)

```bash
quick_start.bat
```

Это:
1. Проверит Python
2. Установит зависимости
3. Запустит Telegram бота

**Затем:** Найдите бота в Telegram и нажмите `/start`

### Вариант 2: Пошагово

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Проверить все функции (DEMO)
python demo.py

# 3. Запустить бота
python main.py bot
```

**Подробная инструкция:** См. [БЫСТРЫЙ_СТАРТ.md](БЫСТРЫЙ_СТАРТ.md)

## 📱 Как работает система?

### 1. Данные
- Загружает исторические OHLCV с Binance
- Публичный API, ключ НЕ нужен
- 3 года истории для обучения

### 2. Фичи (30+)
- Лог-доходности, моментум
- Скользящая волатильность
- Нормализованные тела/хвосты свечей
- Объемные дельты
- Расстояние до EMA50/200
- Нормализованный ATR
- Режимы волатильности
- Сжатие/расширение цены

### 3. Нейросеть
- **Transformer Encoder** (4 слоя, 8 голов внимания)
- Вход: последовательность 128 свечей × 30 фичей
- Выход: 3 класса [NO TRADE, LONG, SHORT]
- Обучение: Walk-forward валидация, early stopping

### 4. Генерация сигнала
- AI выдает вероятности: P(вверх), P(вниз), P(нейтрально)
- Фильтр волатильности (отсекает очень волатильные периоды)
- Пороги уверенности (≥60% для сигнала)
- Результат: **LONG / SHORT / NO TRADE + уверенность %**

### 5. Telegram Mini App
- **Только UI!**
- НЕ подключается к Binance
- НЕ запускает ML модели
- Только получает данные с API

**Весь AI живет на бэкенде!**

## 📦 Структура проекта

```
crypto_ai_bot/
├── main.py                 # Точка входа
├── config.py               # Настройки
├── requirements.txt        # Зависимости
│
├── data/                   # Данные
│   ├── market.py          # Загрузчик данных Binance
│   ├── features.py        # Feature engineering
│   └── datasets.py        # PyTorch datasets
│
├── ai/                     # AI компоненты
│   ├── model.py           # Нейросети (Transformer/LSTM)
│   ├── train.py           # Обучение
│   ├── predict.py         # Предсказания
│   ├── backtest.py        # Бэктестинг
│   └── model_store.py     # Управление моделями
│
├── api/                    # Backend API
│   └── server.py          # FastAPI REST API
│
├── bot/                    # Telegram
│   └── bot.py             # Telegram бот
│
├── payments/               # Подписки
│   └── subscriptions.py   # Управление подписками
│
├── webapp/                 # Mini App UI
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── models/                 # Обученные модели
├── data_cache/            # Кэш данных
└── logs/                  # Логи
```

## 🎓 Обучение моделей

### Одна модель (тест)

```bash
python main.py train-single --symbol BTCUSDT --timeframe 1h
```

Займет ~10 минут на CPU, ~2 минуты на GPU.

### Все модели (продакшен)

```bash
python main.py full-train
```

Обучит 45 моделей (15 монет × 3 таймфрейма).
Займет 4-8 часов на CPU, 1-2 часа на GPU.

## 🔮 Получение предсказаний

```bash
python main.py predict --symbol BTCUSDT --timeframe 1h
```

Вывод:
```
BTCUSDT | 1H
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal: LONG
Confidence: 64%
Volatility: Normal
Model: Neural Network
Price: $43,250.50
Time: 2024-01-15 14:30:00 UTC

Probabilities:
  LONG: 64.2%
  SHORT: 22.1%
  NO TRADE: 13.7%
```

## 💎 Тарифы подписки

| Тариф | Цена | Монеты | Предсказания |
|-------|------|--------|--------------|
| Бесплатный | $0 | 2 | 2 всего |
| Базовый | $14.99/мес | 3 | Безлимит |
| Про | $24.99/мес | 7 | Безлимит |
| Премиум | $49.99/мес | 15 | Безлимит |

## 🛠️ Команды

| Команда | Что делает |
|---------|-----------|
| `python demo.py` | Проверка всех функций |
| `python main.py bot` | Запуск Telegram бота |
| `python main.py api` | Запуск API сервера |
| `python main.py download` | Загрузка данных (3 года) |
| `python main.py train-single --symbol X --timeframe Y` | Обучение 1 модели |
| `python main.py full-train` | Обучение всех моделей |
| `python main.py predict --symbol X --timeframe Y` | Предсказание |
| `python main.py backtest --symbol X --timeframe Y` | Бэктест |

## 📊 Пример использования

### 1. Быстрая проверка

```bash
# Проверить все компоненты
python demo.py
```

### 2. Запустить бота

```bash
# В первом терминале
python main.py bot
```

Найдите бота в Telegram → `/start`

### 3. Запустить API (опционально)

```bash
# Во втором терминале
python main.py api
```

Откройте http://localhost:8000/docs

### 4. Обучить модель

```bash
python main.py train-single --symbol BTCUSDT --timeframe 1h
```

### 5. Получить предсказание

```bash
python main.py predict --symbol BTCUSDT --timeframe 1h
```

## ⚠️ Важные замечания

### Дисклеймер

⚠️ **Эта система только для образовательных и информационных целей.**

- Это НЕ финансовый совет
- Прошлые результаты НЕ гарантируют будущих
- Торговля криптовалютами связана с риском потери средств
- AI предсказания вероятностные, не гарантированные
- Всегда проводите собственное исследование
- Никогда не инвестируйте больше, чем можете потерять

### Профессиональные стандарты

Эта система следует профессиональным ML практикам:

✓ Настоящие нейронные сети на исторических данных
✓ Walk-forward валидация (без заглядывания в будущее)
✓ Правильный feature engineering
✓ Обработка дисбаланса классов
✓ Early stopping и регуляризация
✓ Вероятностные выходы (не фейковая уверенность)
✓ Управление рисками с учетом волатильности

### Что НЕ входит

Система НЕ:
- Не выполняет сделки автоматически
- Не гарантирует прибыль
- Не использует инсайдерскую информацию
- Не манипулирует рынками
- Не дает финансовых советов

## 🌐 Развертывание в продакшен

### 1. Обучить все модели

```bash
python main.py full-train
```

### 2. Задеплоить API

Варианты:
- AWS EC2 / Lambda
- DigitalOcean Droplet
- Google Cloud Run
- Heroku

### 3. Задеплоить Mini App

Варианты:
- Vercel
- Netlify
- GitHub Pages
- AWS S3 + CloudFront

### 4. Настроить бота

1. Загрузите Mini App на хостинг
2. Получите URL
3. Обновите `MINI_APP_URL` в `.env`
4. Настройте в [@BotFather](https://t.me/botfather):
   - `/mybots` → Ваш бот → Bot Settings → Menu Button
   - Установите URL Mini App

### 5. Настроить базу данных (опционально)

По умолчанию SQLite, для продакшена лучше PostgreSQL:

```bash
DATABASE_URL=postgresql://user:pass@host/db
```

## 🔧 Настройка

Основные настройки в [config.py](config.py):

```python
# Поддерживаемые монеты
SUPPORTED_PAIRS = [...]

# Таймфреймы
TIMEFRAMES = ["15m", "1h", "4h"]

# Архитектура модели
MODEL_TYPE = "transformer"  # или "lstm"
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8

# Обучение
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
MAX_EPOCHS = 100

# Пороги для сигналов
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.60
```

## 📚 Документация

- **[README.md](README.md)** - Полная документация (English)
- **[БЫСТРЫЙ_СТАРТ.md](БЫСТРЫЙ_СТАРТ.md)** - Пошаговая инструкция (Русский)
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide (English)

## 🆘 Поддержка

Если что-то не работает:

1. **Запустите DEMO:**
   ```bash
   python demo.py
   ```
   Покажет где проблема

2. **Проверьте логи:**
   ```
   logs/demo.log
   logs/bot.log
   logs/api.log
   ```

3. **Проверьте .env файл:**
   - Должен содержать `TELEGRAM_BOT_TOKEN`

4. **Проверьте зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Ваш токен уже настроен!

Файл `.env` уже содержит ваш токен:
```
TELEGRAM_BOT_TOKEN=8380851959:AAFKX1nmlTDDTi9d-rUFBXDHxV6zW4opjHk
```

**Просто запустите:**
```bash
python main.py bot
```

И найдите бота в Telegram!

---

## 🚀 Следующие шаги

### Минимум (только проверить бота):
1. `pip install -r requirements.txt`
2. `python main.py bot`
3. Найти бота в Telegram → `/start`

### Полноценная работа:
1. Запустить DEMO: `python demo.py`
2. Обучить модель: `python main.py train-single --symbol BTCUSDT --timeframe 1h`
3. Запустить API: `python main.py api`
4. Запустить бота: `python main.py bot`
5. Тестировать предсказания

### Для продакшена:
1. Обучить все модели: `python main.py full-train`
2. Задеплоить API на сервер
3. Задеплоить Mini App
4. Настроить автообновление моделей
5. Настроить мониторинг

---

Нет хайпа. Нет гарантий прибыли. Профессиональный количественный подход.

**Готов к запуску!** 🚀
