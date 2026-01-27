# 🚀 Установка на Windows - Пошагово

## Проблема: `pip` не работает

У вас Windows PowerShell не видит команду `pip`. Используем `python -m pip` вместо `pip`.

---

## ✅ РЕШЕНИЕ 1: Автоматическая установка (РЕКОМЕНДУЕТСЯ)

Просто дважды кликните на файл:

```
install.bat
```

Или запустите в PowerShell:

```powershell
.\install.bat
```

Подождите 2-5 минут пока установятся все пакеты.

---

## ✅ РЕШЕНИЕ 2: Ручная установка (если автоматическая не работает)

Скопируйте и вставьте в PowerShell **по одной команде**:

### Шаг 1: Обновить pip
```powershell
python -m pip install --upgrade pip
```

### Шаг 2: Установить основные пакеты
```powershell
python -m pip install torch numpy pandas scikit-learn
```

### Шаг 3: Установить веб-фреймворки
```powershell
python -m pip install fastapi uvicorn pydantic python-dotenv
```

### Шаг 4: Установить Telegram бот
```powershell
python -m pip install python-telegram-bot
```

### Шаг 5: Установить базу данных
```powershell
python -m pip install sqlalchemy aiosqlite
```

### Шаг 6: Установить утилиты
```powershell
python -m pip install loguru requests python-dateutil pytz tqdm
```

### Шаг 7: Установить дополнительные
```powershell
python -m pip install ccxt ta scipy
```

---

## ✅ РЕШЕНИЕ 3: Минимальная установка (только для запуска бота)

Если полная установка не работает, установите минимум:

```powershell
python -m pip install python-telegram-bot loguru sqlalchemy aiosqlite python-dotenv
```

Этого хватит чтобы запустить бота (но не AI предсказания).

---

## 🚀 После установки

### Проверка установки:

```powershell
python -c "import telegram; print('✓ Telegram bot OK')"
python -c "import loguru; print('✓ Loguru OK')"
python -c "import sqlalchemy; print('✓ Database OK')"
```

Если все команды вывели "OK" - можно запускать!

### Запуск бота:

```powershell
python main.py bot
```

---

## 🔧 Если всё равно ошибки

### Ошибка: "No module named 'torch'"

Это нормально для первого запуска. Установите так:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Ошибка: "externally-managed-environment"

Если видите эту ошибку, используйте:

```powershell
python -m pip install --user -r requirements.txt
```

### Ошибка: Долго устанавливается

Это нормально. PyTorch большой пакет (~200 МБ).
Просто подождите 5-10 минут.

---

## 📝 Быстрая проверка - работает ли Python?

```powershell
python --version
```

Должно показать: `Python 3.x.x`

Если показывает - значит Python установлен правильно.

---

## ✅ ГОТОВО! Теперь можно запускать:

### Минимальный запуск (только бот, без AI):

```powershell
python main.py bot
```

### С предсказаниями (нужны все пакеты):

Сначала установите ВСЕ пакеты (Решение 1 или 2), затем:

```powershell
# Обучить модель
python main.py train-single --symbol BTCUSDT --timeframe 1h

# Запустить бота
python main.py bot
```

---

## 🎯 Итоговая команда установки (одна строка):

```powershell
python -m pip install torch numpy pandas scikit-learn fastapi uvicorn pydantic python-dotenv python-telegram-bot sqlalchemy aiosqlite loguru requests python-dateutil pytz tqdm ccxt ta scipy --index-url https://download.pytorch.org/whl/cpu
```

Скопируйте всю строку, вставьте в PowerShell, нажмите Enter.
Подождите 5-10 минут.

---

## ✅ После успешной установки:

```powershell
python main.py bot
```

И найдите бота в Telegram!

---

## 💡 СОВЕТ

Если хотите просто проверить что бот работает:

1. Установите минимум (Решение 3)
2. Запустите `python main.py bot`
3. Найдите бота в Telegram
4. Протестируйте команды

Потом установите остальное для AI предсказаний.
