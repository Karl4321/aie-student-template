# S03 – eda_cli: мини-EDA для CSV

CLI-приложение для анализа CSV-файлов и HTTP-сервис для оценки качества данных. Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

---

## Часть 1: CLI-интерфейс

### 1. Краткий обзор (overview)

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

Выводит краткую статистику по датасету: количество строк/столбцов и таблицу с характеристиками каждой колонки.

### 2. Полный EDA-отчёт (report)

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

**Параметры:**

- `--title` – заголовок отчёта (по умолчанию: "Какой-то интересный отчётик про EDA CSV файла, приятного")
- `--top-k-categories` – количество топовых категорий для вывода по категориальным признакам (по умолчанию: 2)

**Как параметры влияют на отчёт:**
- `--title` задаёт пользовательский заголовок в Markdown-отчёте
- `--top-k-categories` определяет, сколько самых частых значений показывать для категориальных колонок

**Флаги качества данных в отчёте:**
- `has_high_cardinality_categoricals` – показывает, есть ли колонки с высокой кардинальностью
- `has_suspicious_id_duplicates` – проверяет, содержит ли первая колонка (предположительно user_id) все уникальные значения

**Обновлённый расчёт скор-оценки качества:**
- Базовая оценка: 1.0
- Уменьшается на долю максимальных пропусков
- Дополнительно снижается на 0.5 при наличии подозрительных дубликатов в ID
- Дополнительные штрафы за мало строк (<100) или много колонок (>100)
- Ограничена диапазоном [0.0, 1.0]

## Примеры вызова с новыми опциями

```bash
# С пользовательским заголовком и 5 топовыми категориями
uv run eda-cli report data/example.csv \
  --out-dir reports_example \
  --title "Анализ пользовательских данных" \
  --top-k-categories 5

# С параметрами по умолчанию
uv run eda-cli report data/S02-hw-dataset.csv --out-dir reports
```

## Выходные файлы отчёта

В указанном каталоге (`--out-dir`) создаются:

- `report.md` – основной отчёт в Markdown с новыми флагами качества
- `summary.csv` – таблица со статистикой по колонкам
- `missing.csv` – пропуски по колонкам
- `correlation.csv` – корреляционная матрица (если есть числовые признаки)
- `top_categories/*.csv` – top-k категорий по строковым признакам
- `hist_*.png` – гистограммы числовых колонок
- `missing_matrix.png` – визуализация пропусков
- `correlation_heatmap.png` – тепловая карта корреляций

---

## Часть 2: HTTP-сервис

### Запуск сервера

```bash
uv run uvicorn eda_cli.api:app --host 0.0.0.0 --port 8000 --reload
```

Сервис будет доступен по адресу: `http://localhost:8000`

Автоматическая документация API доступна по адресам:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Доступные эндпоинты:

#### 1. Проверка работоспособности
```bash
GET /health
```
Возвращает статус сервиса.

#### 2. Оценка качества по агрегированным признакам
```bash
POST /quality
```
Принимает JSON с параметрами датасета, возвращает эвристическую оценку качества.

Пример запроса:
```json
{
  "n_rows": 1000,
  "n_cols": 10,
  "max_missing_share": 0.1,
  "numeric_cols": 5,
  "categorical_cols": 3
}
```

#### 3. Оценка качества по CSV-файлу
```bash
POST /quality-from-csv
```
Принимает CSV-файл через multipart/form-data, запускает полный EDA-анализ и возвращает оценку качества.

#### 4. Получение полных флагов качества из CSV (новый эндпоинт)
```bash
POST /quality-flags-from-csv
```
**Новый эндпоинт**: Принимает CSV-файл и возвращает полный набор булевых флагов качества данных.

Пример использования cURL:
```bash
curl -X POST "http://localhost:8000/quality-flags-from-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

Этот эндпоинт возвращает подробные флаги качества, включая:
- `has_suspicious_id_duplicates` - подозрительные дубликаты в ID
- `has_high_cardinality_categoricals` - высокая кардинальность категориальных признаков
- `has_many_missing` - много пропущенных значений
- и другие диагностические флаги

---

## Тестирование

### Запуск тестов CLI
```bash
uv run pytest -q
```

### Тестирование API
После запуска сервера можно протестировать эндпоинты:

```bash
# Проверка health
curl http://localhost:8000/health

# Проверка качества по CSV
curl -X POST "http://localhost:8000/quality-from-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"

# Проверка нового эндпоинта флагов качества
curl -X POST "http://localhost:8000/quality-flags-from-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

## Архитектура проекта

```
eda_cli/
├── __init__.py
├── api.py              # FastAPI приложение с эндпоинтами
├── cli.py              # CLI команды (overview, report)
├── core.py             # Основная логика EDA
└── __main__.py
```

Сервис использует:
- **FastAPI** для HTTP-интерфейса
- **Pandas** для обработки данных
- **Matplotlib/Seaborn** для визуализации
- **Uvicorn** для запуска ASGI-сервера

## Лицензия

MIT