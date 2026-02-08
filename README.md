# MiniFaceLib

## Описание

MiniFaceLib е малък Python модул за анализ на изображения с лица,
който използва машинно обучение за предсказване на емоция, пол и възраст.

## Инсталация

За да инсталиране пуснете:
```
pip install -e .
```

За да направите prediction ви трябва обект FacePredictor:

```
from minifacelib import make_predictor

predictor: FacePredictor = make_predictor("<model_type>")
```

Тук `<model_type>` е или `openai` (използва API-то на OpenAI) или `cnn` използва два малки локални CNN модела (един модел за емоции и един модел едновременно за пол и възраст).

За да пуснете CNN ще ви трябват тежестите. Процесът за получаване на тежестите е следния:

1. Изтеглете двата dataset-a - FER2013 (за емоции) и UTKFace (за пол и възраст) като пуснете `python ./training/fetch_datasets.py`.

2. Натренирайте Emotion модела, като пуснете `python ./training/train_fer.py --epoch 6` (тежестите се записват в `./src/minifacelib/models/cnn/weights/emotion.pt`).

3. Натренирайте Gender + Age модела, като пуснете `python ./training/train_utkface.py --epoch 40` (тежестите се записват в `./src/minifacelib/cnn/weights/gender_age.pt`)

4. Готово.

N.B.: можете да ми пишете на пратя наготово тежестите.

В `./reports` има резултати от валидацията при тренирането и от evaluation на тестови данни. Може да получите тези резултати чрез script-овете в `./evaluation`.

## Използване

Налично е малко HTTP демо приложение. За да го стартирате пуснете:

```
python ./demo/webapp.py --port <port> (--verbose)`

ВАЖНО: За OpenAI трябва да сте set-нали OPENAI_API_KEY.

```

## GitHub

Проектът е качен на: https://github.com/georcons/minifacelib