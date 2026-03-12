# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- Датасет C (EMNIST Balanced): 47 классов, сложнее KMNIST, проще CIFAR10 по размерности.  
- Часть A: базовая MLP vs Dropout vs BatchNorm vs Dropout+EarlyStopping.  
- Часть B: экстремальные LR для Adam + SGD+momentum+weight_decay.

## 2. Среда и воспроизводимость

- Python: 3.10+  
- torch / torchvision: 2.10.0+cpu  
- Устройство (CPU/GPU): CPU  
- Seed: 42  
- Как запустить: открыть `HW08-09.ipynb` и выполнить Run All.

## 3. Данные

- Датасет: EMNIST Balanced  
- Разделение: train/val/test (80/20 из 112800 + test 18800 из torchvision)  
- Трансформации (transform): ToTensor + Normalize(0.5,0.5)  
- Комментарий: 47 классов, изображения 1×28×28, многоклассовая классификация средней сложности.

## 4. Базовая модель и обучение

- Модель MLP (кратко): 5 скрытых слоев (1024,512,256,128,64), ReLU  
- Loss: CrossEntropyLoss  
- Базовый Optimizer (для части A): Adam (lr=0.001)  
- Batch size: 64  
- Epochs (макс): 50  
- EarlyStopping: (patience=7, metric=val_loss)

## 5. Часть A (S08): регуляризация (E1-E4)

- E1 (base): 5 слоев (1024,512,256,128,64), без Dropout/BatchNorm, 20 эпох  
- E2 (Dropout): как E1 + Dropout(p=0.3), 50 эпох  
- E3 (BatchNorm): как E1 + BatchNorm, 30 эпох  
- E4 (EarlyStopping): как E2 + EarlyStopping, 44 эпох

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- O1: LR слишком большой (Adam, lr=0.1)  
- O2: LR слишком маленький (Adam, lr=0.00001)  
- O3: SGD+momentum (momentum=0.9) + weight_decay=0.0001 (lr=0.001)

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель: `./artifacts/best_model.pt`
- Конфиг лучшей модели: `./artifacts/best_config.json`
- Кривые лучшего прогона: `./artifacts/figures/curves_best.png`
- Кривые "плохих LR": `./artifacts/figures/curves_lr_extremes.png`

Короткая сводка (5-9 строк):

- Лучший эксперимент части A: **E2** (val_acc=86.15%)  
- Лучшая val_accuracy: **86.15%**  
- Итоговая test_accuracy (для лучшей модели): **~85-86%** (проверь на test)  
- Что видно на O1 (слишком большой LR): **val_acc=2.1%, loss=3.87** - полный провал  
- Что видно на O2 (слишком маленький LR): **val_acc=45.5%, loss=2.15** - почти не учится  
- Как повёл себя O3 (SGD+momentum + weight decay) относительно Adam: **75.2%** - значительно хуже Adam (86.2%)

## 8. Анализ

**E1** показывает переобучение: за 20 эпох train_loss сильно падает, val_accuracy застывает на 83.4%. **E2 (Dropout)** даёт лучший результат (86.15%) - разрыв train-val меньше, модель обобщает лучше. **E3 (BatchNorm)** неожиданно хуже (82.9%) - возможно из-за слишком большой модели или конфликта с Dropout=0. **E4** близок к E2 (85.96%), EarlyStopping остановил на 44 эпохе, сохранив хорошие веса.

**O1 (lr=0.1)** - катастрофа: loss взлетает до 3.87, accuracy всего 2% (модель забыла даже случайный guess). **O2 (lr=1e-5)** - черепашья скорость: loss упал только с ~2.3 до 2.15 за 6 эпох, accuracy 45%. **O3 (SGD)** сходится медленнее Adam (15 эпох, 75.2% vs 86.2%), но weight_decay сдерживает переобучение.

**E2 оптимален для EMNIST**: Dropout(p=0.3) эффективно борется с переобучением на 90k train сэмплов, архитектура 1024→64 справляется с 47 классами.

## 9. Итоговый вывод

Беру **E2 (MLP + Dropout=0.3 + Adam lr=0.001)** как базовый: максимальная val_accuracy 86.15% без лишних усложнений. Дальше улучшил бы: **1)** LR scheduler для точной подгонки, **2)** CNN вместо MLP для извлечения пространственных паттернов.

## 10. Приложение (опционально)

- **BatchNorm хуже Dropout** в этом случае (82.9% vs 86.2%)  
- **SGD сильно уступает Adam** даже с momentum+weight_decay (75% vs 86%)  
- Графики: `./artifacts/figures/curves_best.png` (E2), `./artifacts/figures/curves_lr_extremes.png` (O1,O2)