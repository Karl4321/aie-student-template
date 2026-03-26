# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет STL10 — он содержит 10 классов, изображения 96x96, подходит для сравнения простых и сложных моделей.
- Для части B выбран Pascal VOC, трек detection — классический датасет для задачи детекции объектов.
- В части A сравнивались: простая CNN, CNN с аугментациями, ResNet18 (head-only), ResNet18 (partial fine-tune). В части B — два режима порога score_threshold для детекции.

## 2. Среда и воспроизводимость

- Python: 3.x
- torch / torchvision: см. вывод первой ячейки ноутбука
- Устройство (CPU/GPU): определяется автоматически
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: STL10
- Разделение: train/val/test (val отделён из train, 20%)
- Базовые transforms: ToTensor + Normalize (ImageNet)
- Augmentation transforms: RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
- Комментарий: 10 классов, изображения 96x96, задача сложнее CIFAR10, но проще CIFAR100, классы сбалансированы.

### 3.2. Часть B: structured vision

- Датасет: Pascal VOC
- Трек: detection
- Что считается ground truth: bounding boxes из разметки VOC
- Какие предсказания использовались: выходы FasterRCNN с разными score_threshold
- Комментарий: VOC — стандарт для object detection, много классов, разная сложность объектов, IoU >= 0.5 — разумный критерий.

## 4. Часть A: модели и обучение (C1-C4)

- C1 (simple-cnn-base): простая CNN, без аугментаций
- C2 (simple-cnn-aug): та же CNN, но с аугментациями
- C3 (resnet18-head-only): ResNet18, заморожен backbone, обучается только fc
- C4 (resnet18-finetune): ResNet18, разморожены layer4+fc
- Loss: CrossEntropyLoss
- Optimizer(ы): Adam
- Batch size: 64
- Epochs (макс): 20
- Критерий выбора лучшей модели: максимальная val_accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

- Модель: FasterRCNN_ResNet50_FPN (pretrained)
- V1: score_threshold = 0.3
- V2: score_threshold = 0.7
- Как считался IoU: box_iou из torchvision.ops, greedy matching, IoU >= 0.5
- Как считались precision / recall: по сопоставленным предсказаниям (TP/FP/FN)

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/detection_examples.png`, `./artifacts/figures/detection_metrics.png`

Короткая сводка:

- Лучший эксперимент части A: (см. runs.csv)
- Лучшая val_accuracy: (см. runs.csv)
- Итоговая test_accuracy лучшего классификатора: (см. runs.csv)
- Что дали аугментации (C2 vs C1): увеличение устойчивости и точности
- Что дал transfer learning (C3/C4 vs C1/C2): значительный прирост качества
- Что оказалось лучше: partial fine-tuning (C4)
- Что показал режим V1 во второй части: выше recall, ниже precision
- Что показал режим V2 во второй части: выше precision, ниже recall
- Как интерпретируются метрики второй части: trade-off между precision и recall при разных порогах

## 7. Анализ

- Простая CNN ограничена по мощности, на STL10 даёт средние результаты.
- Аугментации стабильно улучшают качество, особенно на валидации.
- Предобученная ResNet18 даёт сильный прирост даже при head-only.
- Partial fine-tuning (C4) даёт лучший баланс между скоростью и качеством.
- Для детекции IoU >= 0.5 — разумная метрика, отражает качество локализации.
- При переходе от V1 к V2 precision растёт, recall падает — ожидаемый trade-off.
- Ошибки модели: пропуски мелких объектов, ложные срабатывания на сложных сценах.

## 8. Итоговый вывод

- В качестве базового конфига для классификации — ResNet18 с partial fine-tuning и аугментациями.
- Transfer learning — мощный инструмент для CV-задач, особенно при малых данных.
- Для detection важен баланс между precision и recall, метрики нужно подбирать под задачу.

## 9. Приложение (опционально)

- (здесь можно добавить дополнительные сценарии, confusion matrix, дополнительные графики)
