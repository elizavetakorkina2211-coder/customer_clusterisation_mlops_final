# MLOps и сегментация клиентов

Данные о пользователях (CSV), бинарная/мультиклассовая сегментация (`whales`, `loyal`, `casual`), простой REST-сервис на FastAPI с веб-страницами, учёт экспериментов в MLflow, базовый мониторинг через Prometheus/Grafana и расчёт дрейфа по накопленным предсказаниям. Отдельно подключены DVC для пайплайна, Docker Compose для локального поднятия всего стека, манифесты под Kubernetes и пример Application для Argo CD. Манифесты k8s проверялись в minikube.

## Что лежит в репозитории

- `src/segmentation_mlops/` — код: фичи, train, дрейф, API, SQLite с предсказаниями
- `dvc.yaml`, `params.yaml` — стадии `prepare` / `train` и настройки
- `models/model.joblib` — появляется после обучения
- `data/raw`, `data/processed` — сырьё и эталонный профиль для дрейфа
- `reports/metrics.json`, `reports/drift/` — метрики для DVC и отчёты дрейфа (`latest_drift.json` для UI)
- `docker-compose.yml`, `Dockerfile`, `docker-entrypoint.sh` — приложение + MLflow + Prometheus + Grafana
- `monitoring/` — конфиг Prometheus и дашборд Grafana
- `k8s/`, `argocd/application.yaml` — деплой и пример Argo CD
- `.github/workflows/` — CI на PR (ruff, pytest, сборка образа), CD на `main` в GHCR; опционально дергается DVC
- `.gitlab-ci.yml` — то же по смыслу, если репозиторий на GitLab
- `cookiecutter-mlops/` — заготовка под другой сервис (к проекту не подключается, просто лежит рядом)

## Запуск у себя 

```bash
cd Млопс
python3 -m venv .venv
source .venv/bin/activate   
pip install -e ".[dev]"
pip install dvc

git init    # если ещё не инициализировали
dvc init

python -m segmentation_mlops.data.make_dataset
dvc repro   # или напрямую: python -m segmentation_mlops.train.train_model

uvicorn segmentation_mlops.api.main:app --reload --host 0.0.0.0 --port 8000
```

Документация API: http://127.0.0.1:8000/docs  
Интерфейс: http://127.0.0.1:8000/ (главная, инференс, таблица предсказаний, эксперименты MLflow, страница дрейфа)  
Метрики в формате Prometheus: http://127.0.0.1:8000/metrics  

MLflow по умолчанию пишет в локальную папку `mlruns/`. Для HTML-отчётов Evidently: `pip install -e ".[evidently]"`.

## Docker Compose

```bash
docker compose up --build
```

Порты: приложение 8000, MLflow 5000, Prometheus 9090, Grafana 3000 (логин/пароль по умолчанию `admin` / `admin`)

## Дрейф 

Считается по последним записям из БД предсказаний: `POST /api/v1/drift/run` или кнопка на главной. Есть PSI/KS по числовым признакам к эталону из `reference_profile.json`, проверка сдвига распределения предсказаний, при наличии `segment_truth` в батче — ещё и target drift, плюс сравнение «ранних» и «поздних» строк по `event_ts` (доля `recent_fraction` в `params.yaml`).

Автопереучение: в `params.yaml` поле `drift.auto_retrain_min_flags`. Если флагов не меньше этого числа, после расчёта дрейфа в фоне дергается train и модель перечитывается; `0` — выключено.

## CI/CD

На GitHub: в `cd.yml` образ уходит в GHCR за счёт `GITHUB_TOKEN`. Чтобы после пуша обновлять Deployment в кластере, в настройках репозитория нужны variable `K8S_DEPLOY=true` и secret `KUBE_CONFIG_B64` (kubeconfig в base64 одной строкой), плюс заранее применённые манифесты из `k8s/`.

На GitLab — см. `.gitlab-ci.yml`: образ в Container Registry, деплой по тем же идеям с `K8S_DEPLOY` и `KUBE_CONFIG_B64`.

## Kubernetes (minikube)

```bash
minikube start
eval $(minikube docker-env)
docker build -t segmentation-mlops:latest .
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl -n segmentation-mlops port-forward svc/segmentation-api 8080:80
```

Образ должен быть виден кластеру (через `minikube docker-env` или свой registry). MLflow в манифесте указан как сервис в том же namespace; если его нет — поменять `MLFLOW_TRACKING_URI` в `k8s/deployment.yaml`, например на `file:/app/mlruns`.

## Argo CD

В `argocd/application.yaml` подставить свой `repoURL` и ветку, при необходимости `path`, затем применить манифест в namespace `argocd`. Синхронизация подтянет то, что лежит в `k8s/`.

## Cookiecutter 

```bash
pip install cookiecutter
cookiecutter cookiecutter-mlops
```

## Тесты

```bash
ruff check src tests
pytest
```

Перед pytest нужен установленный пакет: `pip install -e ".[dev]"`.

## Переменные окружения

| Переменная | Зачем |
|------------|--------|
| `MLFLOW_TRACKING_URI` | куда ходит MLflow (в compose — на сервис `mlflow`) |
| `MLOPS_MODEL_PATH` | свой путь к `model.joblib`, если не `models/model.joblib` |
| `MLOPS_DB_PATH` | свой путь к SQLite с предсказаниями |
| `MLOPS_MLFLOW_TRACKING_URI` | явный override URI для приложения |

Префикс в `Settings`: `MLOPS_`, см. `src/segmentation_mlops/config.py`.

## Лицензия

Учебная работа
