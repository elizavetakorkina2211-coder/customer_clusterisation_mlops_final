# MLOps-проект: сегментация клиентов

Учебный MLOps-проект для мультиклассовой классификации клиентов по трём сегментам:

- `whales` - наиболее ценные и активные клиенты;
- `loyal` — постоянные клиенты;
- `casual` — нерегулярные клиенты.

Несмотря на название репозитория, в текущей реализации используется не классическая кластеризация без учителя, а **контролируемая мультиклассовая классификация**, поскольку в данных присутствует целевая переменная `segment_truth`.

Проект демонстрирует полный цикл работы с ML-моделью: подготовку данных, обучение, учёт экспериментов, версионирование данных и модели, публикацию REST API, мониторинг, расчёт дрейфа и развёртывание в Kubernetes по GitOps-схеме.

## Основные компоненты

- **FastAPI** — REST API и веб-интерфейс для инференса.
- **MLflow** — параметры, метрики, артефакты запусков и реестр моделей.
- **DVC** — воспроизводимый пайплайн и версионирование данных и модели.
- **Yandex Object Storage** — удалённое объектное хранилище DVC.
- **SQLite** — история предсказаний приложения.
- **Prometheus** — сбор технических метрик.
- **Grafana** — визуализация метрик.
- **Docker Compose** — локальный запуск всего стека.
- **Kubernetes / Minikube** — оркестрация контейнеров.
- **Argo CD** — GitOps-синхронизация Kubernetes с манифестами в Git.
- **GitHub Actions** — CI на Pull Request и CD после изменений в `main`.

## Структура репозитория

```text
src/segmentation_mlops/   код подготовки признаков, обучения, API, дрейфа и работы с SQLite
params.yaml               параметры обучения, дрейфа и флагов отдельных предсказаний
dvc.yaml                  стадии DVC: prepare и train
dvc.lock                  зафиксированные входы, параметры и выходы DVC-пайплайна
data/raw/                  сгенерированный датасет клиентов
data/processed/            эталонный профиль для расчёта дрейфа
models/model.joblib        сериализованная обученная модель
reports/metrics.json       метрики последнего обучения
reports/drift/             результаты расчёта дрейфа, включая latest_drift.json
Dockerfile                 образ FastAPI-приложения
docker-compose.yml         локальный стек: API, MLflow, Prometheus и Grafana
docker-entrypoint.sh       подготовка приложения при запуске контейнера
monitoring/                конфигурация Prometheus и дашборды Grafana
k8s/                       Kubernetes-манифесты приложения, MLflow, сервисов и PVC
argocd/application.yaml    описание приложения Argo CD
.github/workflows/         GitHub Actions CI/CD
.gitlab-ci.yml             альтернативный пример для GitLab; не является основным сценарием развёртывания
cookiecutter-mlops/        отдельный шаблон для создания нового MLOps-сервиса
```

## Данные и модель

В учебной версии используется синтетический датасет на 9000 клиентов: по 3000 наблюдений для каждого сегмента. Данные генерируются модулем:

```bash
python -m segmentation_mlops.data.make_dataset
```

После подготовки формируются:

```text
data/raw/customers_raw.csv
data/processed/reference_profile.json
```

`reference_profile.json` содержит эталонные статистики числовых и категориальных признаков и используется при мониторинге дрейфа.

Модель обучается как мультиклассовый классификатор на табличных признаках и сохраняется в:

```text
models/model.joblib
```

Параметры обучения задаются в `params.yaml`.

> Важно: данные синтетические и хорошо разделимы, поэтому полученные метрики нельзя напрямую переносить на реальную бизнес-задачу.

## Локальная установка

Рекомендуемая версия Python - 3.11, поскольку она используется в Docker и GitHub Actions.

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd customer_clusterisation_mlops_final

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install dvc dvc-s3
```

Для HTML-отчётов Evidently:

```bash
pip install -e ".[evidently]"
```

`git init` и `dvc init` для уже клонированного репозитория повторно выполнять не нужно.

## DVC-пайплайн

В `dvc.yaml` определены две стадии:

```text
prepare → генерация CSV и reference_profile.json
train   → обучение model.joblib и расчёт metrics.json
```

Основной запуск:

```bash
dvc repro
```

Просмотр метрик:

```bash
dvc metrics show
```

Удалённое DVC-хранилище настроено в Yandex Object Storage через S3-совместимый интерфейс. В нём версионируются:

```text
data/raw/customers_raw.csv
data/processed/reference_profile.json
models/model.joblib
```

Загрузка текущей версии в удалённое хранилище:

```bash
dvc push
```

Восстановление файлов:

```bash
dvc pull
```

Для доступа требуются корректно настроенные учётные данные Object Storage. Секретные ключи не должны сохраняться в Git.

`reports/metrics.json` объявлен как DVC-метрика с `cache: false`, поэтому он не хранится как обычный объект DVC-кэша.

## Запуск FastAPI без Docker

После подготовки данных и обучения модели:

```bash
uvicorn segmentation_mlops.api.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

Доступные адреса:

```text
http://127.0.0.1:8000/         веб-интерфейс
http://127.0.0.1:8000/docs     документация OpenAPI
http://127.0.0.1:8000/metrics  метрики Prometheus
```

В веб-интерфейсе доступны инференс, история предсказаний, информация о запусках MLflow и страница дрейфа.

## MLflow

При локальном запуске без отдельного сервера MLflow по умолчанию использует файловое хранилище `mlruns/`.

Во время обучения в MLflow записываются:

- параметры модели;
- значения метрик;
- сведения о запуске;
- модель и её версия в Model Registry.

DVC и MLflow решают разные задачи:

```text
DVC    → версии данных, модели и воспроизводимость файлового пайплайна
MLflow → эксперименты, параметры, метрики, артефакты и реестр моделей
```

## Docker Compose

Запуск локального стека:

```bash
docker compose up --build
```

Сервисы доступны по адресам:

```text
FastAPI     http://localhost:8000
MLflow      http://localhost:5001
Prometheus  http://localhost:9090
Grafana     http://localhost:3000
```

Внутри Docker-сети MLflow слушает порт `5000`, но на компьютер он опубликован как `5001`.

Стандартные учётные данные Grafana в учебной конфигурации:

```text
admin / admin
```

## Хранение предсказаний

FastAPI сохраняет историю инференса в SQLite. В Kubernetes используется путь:

```text
MLOPS_DB_PATH=/data/predictions.db
```

Каталог `/data` подключён к PVC:

```text
segmentation-db-pvc
```

Поэтому пересоздание Pod или обновление Docker-образа не удаляет накопленную историю предсказаний.

В локальном Minikube соответствующий PersistentVolume создаётся через `minikube-hostpath`. Физический путь текущего тома внутри узла Minikube имеет вид:

```text
/tmp/hostpath-provisioner/segmentation-mlops/segmentation-db-pvc
```

Там хранится файл:

```text
predictions.db
```

PVC защищает данные от пересоздания Pod, но не является внешней резервной копией. При удалении самого PVC или полном удалении Minikube данные могут быть потеряны.

## Расчёт дрейфа

Проверка запускается:

```text
POST /api/v1/drift/run
```

или кнопкой в веб-интерфейсе.

Используются следующие проверки:

- PSI по числовым признакам относительно `reference_profile.json`;
- KS-тест для числовых признаков;
- изменение распределения предсказанных классов;
- target drift, если в анализируемом наборе есть `segment_truth`;
- временное сравнение ранних и поздних наблюдений по `event_ts`.

Параметр `recent_fraction` в `params.yaml` определяет долю последних строк для временного сравнения.

Изменение распределения предсказаний является индикатором output drift. Без фактических целевых меток оно не доказывает строгий concept drift.

### Автоматическое переобучение

Порог задаётся параметром:

```text
drift.auto_retrain_min_flags
```

Если количество обнаруженных сигналов дрейфа не меньше порога, приложение может запустить обучение в фоне и перечитать модель.

```text
0 → автоматическое переобучение отключено
```

По умолчанию оно отключено, чтобы новая модель не вводилась в эксплуатацию без дополнительной проверки качества.

## Мониторинг

FastAPI публикует метрики по адресу:

```text
/metrics
```

Prometheus собирает эти метрики, а Grafana отображает их на дашборде.

Мониторинг разделён на два уровня:

```text
технический → запросы, задержки, ошибки и состояние API
модельный   → распределение предсказаний и результаты проверки дрейфа
```

## CI

Workflow CI запускается для Pull Request в основные ветки и выполняет:

1. установку проекта и dev-зависимостей;
2. проверку Ruff;
3. тестовую подготовку данных и обучение;
4. запуск `pytest`;
5. проверочную сборку Docker-образа.

CI не разворачивает приложение. Его задача — не допустить попадания неработающего изменения в `main`.

## CD и GHCR

После изменения ветки `main` CD:

1. собирает Docker-образ через Buildx;
2. публикует его в GitHub Container Registry — GHCR;
3. создаёт варианты для `linux/amd64` и `linux/arm64`;
4. присваивает образу тег Git SHA;
5. обновляет поле `image` в `k8s/deployment.yaml`;
6. создаёт служебный commit от `github-actions[bot]` с пометкой `[skip ci]`.

`[skip ci]` предотвращает бесконечный цикл, при котором служебный commit повторно запускал бы тот же CD.

Прямой деплой из GitHub Actions через `kubectl`, `K8S_DEPLOY` и `KUBE_CONFIG_B64` в текущей схеме **не используется**.

## Kubernetes и Minikube

Для локальной проверки:

```bash
minikube start
kubectl apply -f k8s/
```

Проверка ресурсов:

```bash
kubectl -n segmentation-mlops get pods,svc,pvc
```

Проброс FastAPI:

```bash
kubectl -n segmentation-mlops port-forward svc/segmentation-api 8080:80
```

После этого приложение доступно по адресу:

```text
http://127.0.0.1:8080
```

Основные объекты:

```text
Deployment → управляет Pod и обновлениями приложения
Service    → предоставляет стабильный сетевой адрес
PVC        → хранит постоянные данные
```

В Kubernetes MLflow работает как отдельный сервис в namespace `segmentation-mlops`. Приложение обращается к нему по внутреннему адресу:

```text
http://mlflow.segmentation-mlops.svc.cluster.local:5000
```

Для хранения данных MLflow используется отдельный PVC `mlflow-data`.

## Argo CD

Argo CD реализует GitOps-подход: Git является источником желаемого состояния Kubernetes.

Текущая цепочка развёртывания:

```text
Pull Request
→ CI
→ merge в main
→ CD собирает и публикует образ в GHCR
→ CD обновляет тег image в k8s/deployment.yaml
→ создаётся commit
→ Argo CD обнаруживает изменение
→ Argo CD синхронизирует Kubernetes
```

Применение описания приложения:

```bash
kubectl apply -n argocd -f argocd/application.yaml
```

Перед применением следует проверить в `argocd/application.yaml`:

- `repoURL`;
- `targetRevision`;
- `path`;
- namespace назначения.

Основные статусы Argo CD:

```text
Synced  → состояние кластера совпадает с Git
Healthy → Kubernetes-ресурсы работоспособны
```

Argo CD не собирает Docker-образ. Он применяет состояние, описанное в Git, и следит за его соответствием кластеру.

## Переменные окружения

| Переменная | Назначение |
|---|---|
| `MLFLOW_TRACKING_URI` | адрес MLflow Tracking Server |
| `MLOPS_MLFLOW_TRACKING_URI` | явный override URI MLflow для приложения |
| `MLOPS_MODEL_PATH` | путь к `model.joblib` |
| `MLOPS_DB_PATH` | путь к SQLite-базе предсказаний |
| `MLOPS_ROOT` | корневая директория приложения |

Префикс настроек приложения — `MLOPS_`, см. `src/segmentation_mlops/config.py`.

## Тесты и качество кода

```bash
ruff check src tests
pytest
```

Перед запуском пакет должен быть установлен:

```bash
pip install -e ".[dev]"
```

## Cookiecutter

Каталог `cookiecutter-mlops/` является самостоятельным учебным шаблоном и не участвует в работе текущего сервиса.

Запуск шаблона:

```bash
pip install cookiecutter
cookiecutter cookiecutter-mlops
```

## Статус проекта

Учебная работа по построению MLOps-конвейера для сегментации клиентов.
