# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

---

## 1. Паспорт проекта

- **Название проекта:** VisualSearch — сервис визуального поиска одежды
- **Автор:** `Яковлев Максим Алексеевич`
- **Группа:** `БСБО-04-23`
- **Контакт:** `@Max00025`

- **Краткое описание:**  
  Проект реализует сервис визуального поиска похожих товаров одежды по изображению.  
  Используется датасет DeepFashion (~44 000 изображений), предобученная модель ResNet50 (или CLIP) для извлечения векторных эмбеддингов и библиотека FAISS для быстрого поиска ближайших соседей.  
  Результат — пайплайн, который по загруженному фото одежды возвращает топ-K визуально похожих изображений из базы.

---

## 2. Структура проекта

```
project/
├── README.md                  # этот файл
├── report.md                  # отчёт по проекту
├── self-checklist.md          # чеклист самопроверки
├── requirements.txt           # зависимости Python
├── configs/
│   └── project.yaml           # конфигурация модели, путей, индекса
├── notebooks/
│   ├── 00_download_dataset.ipynb   # скачивание датасета DeepFashion с Google Drive
│   └── 01_train_embeddings_optimized.ipynb  # извлечение эмбеддингов + FAISS-индекс + тест поиска
├── src/
│   └── service/
│       └── api.py             # FastAPI-сервис для поиска по изображению
├── data/
│   ├── dataset/
│   │   └── images/            # изображения DeepFashion (скачиваются отдельно)
│   └── downloads/             # временная папка для zip-архива
├── artifacts/
│   ├── embeddings.npy         # матрица эмбеддингов [N, 2048]
│   ├── metadata.csv           # маппинг индексов → имён файлов
│   ├── faiss_index.index      # бинарный FAISS-индекс
│   └── run_report.json        # отчёт о параметрах запуска (для воспроизводимости)
└── tests/
    └── test_search.py         # базовые проверки поиска
```

> Папка `data/dataset/images/` не хранится в репозитории — датасет скачивается отдельно (см. раздел 5).

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10`
- NVIDIA GPU (рекомендуется) или CPU
- ~7 GB свободного места для датасета (после распаковки)
- ~350 MB для матрицы эмбеддингов

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

**Зависимости (`requirements.txt`):**
```
torch
torchvision
faiss-cpu          # или faiss-gpu для GPU-поиска
pyyaml
pillow
tqdm
albumentations
pandas
numpy
matplotlib
gdown
fastapi
uvicorn
python-multipart
# Для CLIP (опционально):
# git+https://github.com/openai/CLIP.git
```

---

## 4. Как запустить проект

### 4.1. Шаг 1 — Скачать датасет

```bash
# Запустить ноутбук в Jupyter:
jupyter notebook notebooks/00_download_dataset.ipynb
```

Ноутбук скачает архив DeepFashion (~6.4 GB) с Google Drive, распакует его в `data/dataset/images/` и удалит zip после распаковки.

### 4.2. Шаг 2 — Извлечь эмбеддинги и построить индекс

```bash
jupyter notebook notebooks/01_train_embeddings_optimized.ipynb
```

Ноутбук последовательно выполняет:
1. Загрузку конфига из `configs/project.yaml`
2. Сканирование всех изображений в `data/dataset/images/`
3. Извлечение эмбеддингов через ResNet50 (batched, с GPU если доступно)
4. Построение FAISS-индекса (тип и параметры — из конфига)
5. Сохранение артефактов в `artifacts/`
6. Визуальный тест поиска

### 4.3. Шаг 3 — Запустить API-сервис

```bash
cd project
source .venv/bin/activate
uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --reload
```

Сервис поднимается на порту **8000**. Документация доступна по адресу:  
`http://localhost:8000/docs` (Swagger UI)

**Ключевые эндпоинты:**

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка работоспособности сервиса |
| POST | `/search` | Поиск похожих изображений по фото |

**Пример запроса к `/search`:**
```bash
curl -X POST "http://localhost:8000/search" \
  -F "file=@path/to/query_image.jpg" \
  -F "top_k=5"
```

**Пример ответа:**
```json
{
  "results": [
    {"rank": 1, "filename": "MEN-Denim-id_00000089-01_7.jpg", "similarity_score": 0.97},
    {"rank": 2, "filename": "MEN-Denim-id_00000089-02_7.jpg", "similarity_score": 0.94}
  ]
}
```

---

## 5. Данные

Используется открытый датасет **DeepFashion** (подмножество с изображениями одежды):

- **Источник:** Google Drive (ссылка предоставлена в ноутбуке `00_download_dataset.ipynb`)
- **Объём:** ~44 096 изображений формата JPG, ~6.4 GB в архиве
- **Структура:** все изображения лежат плоско в `data/dataset/images/`; имя файла кодирует категорию и ID товара (например, `MEN-Denim-id_00000080-01_7_additional.jpg`)
- **В репозитории:** датасет не хранится. Для скачивания используйте `notebooks/00_download_dataset.ipynb`

---

## 6. Настройка через конфиг

Все параметры вынесены в `configs/project.yaml` — изменение модели, размера батча, типа индекса и т.д. не требует правки кода:

```yaml
model:
  name: "resnet50"   # или "resnet18", "efficientnet_b0", "clip_vit_b32"
  device: "cuda"     # или "cpu"

inference:
  batch_size: 64
  top_k: 5

index:
  type: "Flat"       # или "IVF", "HNSW"
  nlist: 500
  nprobe: 5
```

---

## 7. Тесты

```bash
cd project
source .venv/bin/activate
pytest tests/
```

Тесты проверяют: загрузку артефактов, корректность размерности эмбеддингов, и то, что поиск изображения по самому себе возвращает его на первом месте (sanity-check).

---

## 8. Демонстрация на защите

На защите планируется:

1. Показать структуру проекта и конфиг (`configs/project.yaml`)
2. Запустить сервис через `uvicorn`, отправить несколько POST-запросов через Swagger UI и показать визуальные результаты поиска
3. Открыть `notebooks/01_train_embeddings_optimized.ipynb` и показать: извлечение эмбеддингов, построение FAISS-индекса, визуальную сетку с топ-5 результатами

---

## 9. Ограничения и дальнейшая работа

**Текущие ограничения:**
- Нет fine-tuning на парах похожих товаров — используются только предобученные веса ImageNet
- Поиск работает только по визуальному сходству, без учёта категорий и атрибутов
- API не имеет авторизации и rate-limiting

**Направления развития:**
- Fine-tune ResNet или CLIP на triplet loss для улучшения качества поиска
- Добавить фильтрацию по категории одежды (мужская/женская, тип одежды)
- Переключиться на `faiss-gpu` для ускорения поиска
- Добавить кеширование (Redis) и логирование запросов
