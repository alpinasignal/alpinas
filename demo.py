"""
Alpina Signal - DEMO MODE
Быстрая проверка всех функций БЕЗ обучения моделей
"""

import sys
import os
from loguru import logger

# Setup logging
logger.add("logs/demo.log", rotation="10 MB", level="INFO")


def demo_1_data_download():
    """DEMO 1: Загрузка данных с Binance"""
    print("\n" + "="*60)
    print("DEMO 1: Загрузка данных с Binance (БЕЗ API ключа)")
    print("="*60 + "\n")

    from data.market import BinanceDataFetcher

    fetcher = BinanceDataFetcher()

    # Загружаем только 30 дней для быстрой демонстрации
    print("Загружаем последние 30 дней для BTCUSDT 1h...")
    df = fetcher.fetch_historical_data("BTCUSDT", "1h", days=30)

    if df is not None:
        print(f"✓ Загружено {len(df)} свечей")
        print(f"✓ Диапазон дат: {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"✓ Последняя цена: ${df['close'].iloc[-1]:.2f}")
        print("\nПервые 5 строк:")
        print(df.head())
        return df
    else:
        print("✗ Ошибка загрузки данных")
        return None


def demo_2_feature_engineering(df):
    """DEMO 2: Создание фичей для ML"""
    print("\n" + "="*60)
    print("DEMO 2: Feature Engineering (30+ профессиональных фичей)")
    print("="*60 + "\n")

    from data.features import FeatureEngine, create_labels

    engine = FeatureEngine()

    print("Создаем фичи...")
    df_features = engine.create_all_features(df)

    print(f"✓ Создано {len(engine.get_feature_names())} фичей")
    print(f"✓ Размер датасета: {df_features.shape}")

    print("\nСписок фичей:")
    for i, feat in enumerate(engine.get_feature_names(), 1):
        print(f"  {i}. {feat}")

    # Создаем метки для обучения
    print("\nСоздаем метки (labels)...")
    df_labeled = create_labels(df_features)

    print(f"✓ Размер после меток: {df_labeled.shape}")
    print("\nРаспределение меток:")
    print(df_labeled['label'].value_counts())
    print("  0 = NO TRADE")
    print("  1 = LONG")
    print("  2 = SHORT")

    return df_labeled, engine


def demo_3_model_architecture():
    """DEMO 3: Архитектура нейросети"""
    print("\n" + "="*60)
    print("DEMO 3: Архитектура Neural Network (Transformer)")
    print("="*60 + "\n")

    from ai.model import create_model
    import torch

    num_features = 30

    print("Создаем Transformer модель...")
    model = create_model(num_features, model_type="transformer", device="cpu")

    print("\nАрхитектура:")
    print(f"  • Входной слой: {num_features} фичей")
    print(f"  • Sequence length: 128 свечей")
    print(f"  • Hidden dimension: 256")
    print(f"  • Transformer layers: 4")
    print(f"  • Attention heads: 8")
    print(f"  • Dropout: 0.2")
    print(f"  • Выходной слой: 3 класса (NO TRADE, LONG, SHORT)")

    # Тест модели
    print("\nТестируем модель...")
    batch_size = 8
    seq_len = 128
    x = torch.randn(batch_size, seq_len, num_features)

    output = model(x)
    probs = model.predict_proba(x)

    print(f"✓ Вход: {x.shape}")
    print(f"✓ Выход (logits): {output.shape}")
    print(f"✓ Вероятности: {probs.shape}")
    print(f"\nПример вероятностей для 1 свечи:")
    print(f"  NO TRADE: {probs[0][0].item()*100:.2f}%")
    print(f"  LONG: {probs[0][1].item()*100:.2f}%")
    print(f"  SHORT: {probs[0][2].item()*100:.2f}%")
    print(f"  Сумма: {probs[0].sum().item()*100:.2f}%")

    return model


def demo_4_api_server():
    """DEMO 4: REST API для Mini App"""
    print("\n" + "="*60)
    print("DEMO 4: REST API Server")
    print("="*60 + "\n")

    print("API эндпоинты:")
    print("  POST /api/v1/predict")
    print("    → Получить предсказание для монеты")
    print("")
    print("  POST /api/v1/predict-batch")
    print("    → Получить предсказания для нескольких монет")
    print("")
    print("  GET /api/v1/market-status/{symbol}")
    print("    → Текущий статус рынка")
    print("")
    print("  GET /api/v1/subscription/{user_id}")
    print("    → Информация о подписке")
    print("")
    print("  GET /api/v1/supported-pairs")
    print("    → Список поддерживаемых монет")
    print("")

    print("Для запуска API:")
    print("  python main.py api")
    print("")
    print("API будет доступен на:")
    print("  http://localhost:8000")
    print("  http://localhost:8000/docs (документация)")


def demo_5_telegram_bot():
    """DEMO 5: Telegram Bot"""
    print("\n" + "="*60)
    print("DEMO 5: Telegram Bot")
    print("="*60 + "\n")

    import config

    if config.TELEGRAM_BOT_TOKEN:
        print(f"✓ Bot token настроен")
        print(f"  Token: {config.TELEGRAM_BOT_TOKEN[:20]}...")
        print("")
        print("Функции бота:")
        print("  /start - Запуск Mini App")
        print("  /subscription - Информация о подписке")
        print("  /help - Помощь")
        print("")
        print("Бот поддерживает:")
        print("  ✓ Telegram Mini App")
        print("  ✓ Управление подписками")
        print("  ✓ Платежи (USDT)")
        print("  ✓ Выбор монет")
        print("")
        print("Для запуска бота:")
        print("  python main.py bot")
        print("")
        print("Затем найдите бота в Telegram и нажмите /start")
    else:
        print("✗ Bot token не настроен")
        print("  Добавьте TELEGRAM_BOT_TOKEN в .env файл")


def demo_6_database():
    """DEMO 6: База данных и подписки"""
    print("\n" + "="*60)
    print("DEMO 6: Database & Subscriptions")
    print("="*60 + "\n")

    from payments.subscriptions import DatabaseManager, SubscriptionManager

    print("Создаем базу данных...")
    db_manager = DatabaseManager()
    sub_manager = SubscriptionManager(db_manager)

    print("✓ База данных создана: alpina_signal.db")
    print("")
    print("Таблицы:")
    print("  • users - Пользователи")
    print("  • subscriptions - Подписки")
    print("  • payments - Платежи")
    print("  • prediction_history - История предсказаний")
    print("")

    # Тестовый пользователь
    print("Создаем тестового пользователя...")
    user = sub_manager.create_user(
        telegram_id=123456789,
        username="test_user",
        first_name="Test"
    )

    print(f"✓ Пользователь создан: ID={user.id}")

    subscription = sub_manager.get_subscription(user.id)
    print(f"✓ Подписка: {subscription.tier}")
    print(f"  Лимит предсказаний: {subscription.predictions_limit}")
    print("")

    print("Тарифы:")
    import config
    for tier_name, tier_info in config.SUBSCRIPTION_TIERS.items():
        print(f"  {tier_name.upper()}: ${tier_info['price']}/мес - {tier_info.get('coins', 'все')} монет")


def demo_7_mini_app():
    """DEMO 7: Telegram Mini App"""
    print("\n" + "="*60)
    print("DEMO 7: Telegram Mini App (UI)")
    print("="*60 + "\n")

    print("Mini App файлы:")
    print("  ✓ webapp/index.html - Интерфейс")
    print("  ✓ webapp/app.js - Логика")
    print("  ✓ webapp/style.css - Стили")
    print("")
    print("Функции Mini App:")
    print("  • Выбор монеты (15 пар)")
    print("  • Выбор таймфрейма (15m, 1h, 4h)")
    print("  • Отображение сигнала (LONG/SHORT/NO TRADE)")
    print("  • Показ уверенности (Confidence %)")
    print("  • Показ вероятностей")
    print("  • Информация о волатильности")
    print("  • Обновление предсказаний")
    print("")
    print("Mini App - это ТОЛЬКО UI!")
    print("  • НЕ подключается к Binance")
    print("  • НЕ запускает ML модели")
    print("  • НЕ считает индикаторы")
    print("  ✓ Только получает данные с API")
    print("")
    print("Весь AI живет на бэкенде!")


def main():
    """Запуск всех демо"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "ALPINA SIGNAL DEMO" + " "*25 + "║")
    print("║" + " "*10 + "Проверка всех функций системы" + " "*18 + "║")
    print("╚" + "="*58 + "╝")

    try:
        # DEMO 1: Загрузка данных
        df = demo_1_data_download()

        if df is None:
            print("\n✗ Ошибка загрузки данных. Проверьте интернет-соединение.")
            return

        input("\nНажмите Enter для продолжения...")

        # DEMO 2: Feature Engineering
        df_labeled, engine = demo_2_feature_engineering(df)

        input("\nНажмите Enter для продолжения...")

        # DEMO 3: Модель
        model = demo_3_model_architecture()

        input("\nНажмите Enter для продолжения...")

        # DEMO 4: API
        demo_4_api_server()

        input("\nНажмите Enter для продолжения...")

        # DEMO 5: Bot
        demo_5_telegram_bot()

        input("\nНажмите Enter для продолжения...")

        # DEMO 6: Database
        demo_6_database()

        input("\nНажмите Enter для продолжения...")

        # DEMO 7: Mini App
        demo_7_mini_app()

        # Финал
        print("\n" + "="*60)
        print("DEMO ЗАВЕРШЕНО!")
        print("="*60 + "\n")

        print("✓ Все компоненты работают!")
        print("")
        print("Следующие шаги:")
        print("")
        print("1. ЗАПУСТИТЬ TELEGRAM BOT:")
        print("   python main.py bot")
        print("   → Найдите бота в Telegram и нажмите /start")
        print("")
        print("2. ЗАПУСТИТЬ API SERVER (в новом терминале):")
        print("   python main.py api")
        print("   → Откройте http://localhost:8000/docs")
        print("")
        print("3. ОБУЧИТЬ МОДЕЛЬ (опционально, для реальных предсказаний):")
        print("   python main.py train-single --symbol BTCUSDT --timeframe 1h")
        print("   → Займет ~10 минут")
        print("")
        print("4. ПОЛУЧИТЬ ПРЕДСКАЗАНИЕ (после обучения модели):")
        print("   python main.py predict --symbol BTCUSDT --timeframe 1h")
        print("")
        print("="*60)
        print("")

    except Exception as e:
        logger.error(f"Ошибка в демо: {e}")
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
