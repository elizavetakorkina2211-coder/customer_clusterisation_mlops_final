# Changelog

Все значимые изменения в проекте документируются в этом файле.
Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/),
проект придерживается [семантического версионирования](https://semver.org/lang/ru/).

## [0.2.0] - 2026-06-07

### Added
- Развёртывание всего стека в Kubernetes (minikube) через Argo CD (GitOps)
- Манифесты k8s для MLflow, Prometheus и Grafana
- Дашборд Grafana «Segmentation MLOps» с метриками инференса и дрейфа
- Регистрация модели в MLflow Model Registry (CustomerSegmentation)
- Мультиархитектурная сборка образа (linux/amd64, linux/arm64)

### Fixed
- Обучение сохраняет модель в MLOPS_ROOT (/app/models) — путь совпадает с инференсом
- Совместимость версии сервера MLflow с клиентом (3.13.0)
- Защита от DNS rebinding в MLflow (--allowed-hosts)
- OOMKilled у MLflow (увеличена память до 2Gi)

## [0.1.0] - 2026-06-01

### Added
- Базовый пайплайн сегментации клиентов (whales/loyal/casual)
- Сервис FastAPI с OpenAPI: инференс, переобучение, дрейф, эксперименты
- Трекинг экспериментов в MLflow
- Расчёт data/target drift и генерация отчётов
- Веб-UI: инференс, таблица предсказаний, флаги аномалий, переобучение, эксперименты, дрейф
- CI/CD (ruff, pytest, сборка образа в GHCR)
- Версионирование данных через DVC (dvc.yaml: prepare, train)
- Шаблон Cookiecutter

[0.2.0]: https://github.com/elizavetakorkina2211-coder/customer_clusterisation_mlops_final/releases/tag/v0.2.0
[0.1.0]: https://github.com/elizavetakorkina2211-coder/customer_clusterisation_mlops_final/releases/tag/v0.1.0
