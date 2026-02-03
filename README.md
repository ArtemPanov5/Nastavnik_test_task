# Nastavnik_test_task
## Тестовое задание для ML-инженера проекта Nastavnik

## Описание решения

Данное решение реализует **простую систему Knowledge Tracing** для предсказания уровня владения студентом определённым навыком на основе истории его попыток.

### Архитектурные решения

Позволил себе изменить базовую архитектуру сервиса, описанную в задании, чтобы сделать код более модульным, читаемым и поддерживаемым:

1. **Разделение на части training, inference, data**: 
   - `data_preprocessing.py` — логика загрузки и обработки данных с валидацией типов
   - `src/training/train.py` — pipeline обучения модели
   - `src/inference/app.py` — FastAPI-сервис для inference

2. **Сохранение обработанных данных**: 
   - Обработанные данные сохраняются в `data/preprocessed_data/` для возможности анализа и отладки
   - Артефакты модели (`knowledge_model.joblib`, `feature_columns.json`) изолированы в `models/`

3. **Docker**: 
   - Добавлен `Dockerfile` для контейнеризации inference-сервиса
   - Образ содержит только часть необходимую для inference. (разделяю это для возможной масштабируемости)

4. **Ошибки и логирование**:
   - Pydantic валидация входных данных с ограничениями типов
   - Graceful degradation: если модель не загрузилась, `/health` сообщит `model_ready: false`
   - Логирование предсказаний для мониторинга

### Признаки

Модель использует 4 признака для предсказания:
- `num_attempts` — общее количество попыток студента
- `success_rate` — доля правильных ответов (0..1)
- `last_correct` — успешность последней попытки (1/0)
- `learning_curve` — средняя успешность последних 3 попыток

**Target**: `success_rate > 0.75` → считаем, что студент знает навык.

**Модель**: LogisticRegression (простая, интерпретируемая, быстрая для inference).

---

## Project structure:
```
Nastavnik_test_task/
├── data/
│ ├── preprocessed_data/ # Обработанные данные (features, interactions_clean)
│ └── raw_data/ # interactions.json
├── models/ # Обученная модель и метаданные
│ ├── knowledge_model.joblib
│ └── feature_columns.json
├── src/
│ ├── inference/ # FastAPI сервис
│ │ └── app.py
│ └── training/ # Пайплайн обучения
│ └── train.py
├── data_preprocessing.py # Модуль загрузки и feature engineering
├── Dockerfile # Контейнер для inference
├── requirements.txt
└── README.md
```

## Local data preprocessing and train of model:

### Run next command from root directory of project:

```shell
python -m src.training.train
```

```output in terminal
(.venv) PS C:\Users\artem\Study\Nastavnik_test_task> python -m src.training.train
Loading interactions from: C:\Users\artem\Study\Nastavnik_test_task\data\raw_data\interactions.json
Unique users: 2
Unique skills: 2
Average correctness: 0.8000
Features shape: (4, 6)
user_id  skill_id  num_attempts  success_rate  last_correct  learning_curve
     u1 functions             2      0.500000             1        0.500000
     u1     loops             3      0.666667             1        0.666667
     u2 functions             2      1.000000             1        1.000000
     u2     loops             3      1.000000             1        1.000000
Saved features to: C:\Users\artem\Study\Nastavnik_test_task\data\preprocessed_data\features.csv
Saved cleaned interactions to: C:\Users\artem\Study\Nastavnik_test_task\data\preprocessed_data\interactions_clean.csv
Train accuracy: 1.0000
Saved model to: C:\Users\artem\Study\Nastavnik_test_task\models\knowledge_model.joblib
Saved feature columns to: C:\Users\artem\Study\Nastavnik_test_task\models\feature_columns.json
```

## Deployment and checking:

### Run next command from root directory of project:

### 1. to build docker_container_of_backend_app:
```shell
docker build -t nastavnik-inference .
```

### 2. to up docker_container_of_backend_app run:
```shell
docker run -d -p 8000:8000 --name nastavnik-api nastavnik-inference
```

### 3. to test check_docker_container_of_backend_app run:
```shell
curl http://localhost:8000/health
```
```output in terminal
StatusCode        : 200
StatusDescription : OK
Content           : {"status":"ok","model_ready":true}
RawContent        : HTTP/1.1 200 OK
                    Content-Length: 34
                    Content-Type: application/json
                    Date: Tue, 03 Feb 2026 12:19:15 GMT
                    Server: uvicorn

                    {"status":"ok","model_ready":true}
Forms             : {}
Headers           : {[Content-Length, 34], [Content-Type, application/json], [Date, Tue, 03 Feb 2026 12:19:15 GMT], [Server, uvicorn]}
Images            : {}
InputFields       : {}
Links             : {}
ParsedHtml        : mshtml.HTMLDocumentClass
RawContentLength  : 34
```

### 4. to test response of ML_service run:
```shell
curl.exe -X POST "http://localhost:8000/predict_knowledge" -H "Content-Type: application/json" -d '{\"user_id\":\"u1\",\"skill_id\":\"loops\",\"num_attempts\":5,\"success_rate\":0.8,\"last_correct\":1,\"learning_curve\":0.67}'
```

```output in terminal
{"knows_skill":false,"confidence":0.42082556230991763}
```