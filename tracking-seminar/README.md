# Семинар: трекинг ML-экспериментов

Практический семинар про четыре инструмента отслеживания экспериментов:
**TensorBoard**, **MLflow**, **Weights & Biases** и **Comet**. Один и тот же
бейзлайн (CNN на FashionMNIST) подключается к каждому из них через общий
интерфейс логгера, чтобы можно было сравнить подходы на одном сценарии.

## Что внутри

| Файл | Назначение |
|---|---|
| `seminar_tracking.ipynb` | основной ноутбук со всеми секциями и запускаемыми примерами |
| `cleanup_tracking.py` | очистка локальных и облачных артефактов трекинга |
| `images_seminar_tracking/` | иллюстрации, которые вставляет ноутбук |
| `requirements.txt` | Python-зависимости |
| `.env.example` | шаблон переменных окружения с ключами |

## Требования

- Python **3.12** (ноутбук проверялся на 3.12; подойдёт 3.10+)
- Git
- Свободный порт `5000` для локального MLflow и `6006` для TensorBoard
- ~1 ГБ на датасет, артефакты и окружение

## Установка

### 1. Окружение

Через `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Или через conda:

```bash
conda create -n tracking-seminar python=3.12 -y
conda activate tracking-seminar
```

### 2. PyTorch

Для CPU-версии (как в ноутбуке по умолчанию):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Для GPU ставьте сборку под свою CUDA по инструкции
<https://pytorch.org/get-started/locally/> — ноутбук сам выберет `cuda`,
если она доступна.

### 3. Остальные зависимости

```bash
pip install -r requirements.txt
```

`torch` и `torchvision` уже перечислены в `requirements.txt`, поэтому при
повторной установке pip только проверит версии и не будет перекачивать их
заново.

### 4. Ядро Jupyter

```bash
python -m ipykernel install --user \
  --name tracking-seminar \
  --display-name "Python (tracking-seminar)"
```

Запуск:

```bash
jupyter lab          # или: jupyter notebook
```

В ноутбуке выберите ядро **Python (tracking-seminar)** и выполняйте ячейки
сверху вниз.

> Датасет FashionMNIST (~80 МБ) скачается автоматически в `./data` при первом
> запуске. Каталог исключён из git.

## Ключи и переменные окружения

Ключи **никогда не вписывайте в ноутбук и не коммитьте**. Ноутбук читает их
из переменных окружения, а `.env` уже добавлен в `.gitignore`.

### Быстрый способ: `.env`

```bash
cp .env.example .env
# откройте .env и заполните WANDB_API_KEY / COMET_API_KEY
set -a
source .env
set +a
jupyter lab
```

`set -a` экспортирует все переменные из файла в текущую оболочку, из которой
запускается Jupyter. Если запускать ноутбук из другого терминала/IDE, переменные
нужно экспортировать там же.

### Weights & Biases

```bash
export WANDB_API_KEY="..."     # ключ: https://wandb.ai/authorize
# или интерактивно, ключ сохранится в ~/.netrc:
wandb login
```

Ячейка `wandb.login()` в ноутбуке подхватит сохранённый ключ или переменную
`WANDB_API_KEY` и не будет ничего спрашивать. Дополнительно можно задать:

```bash
export WANDB_ENTITY="your-team"      # куда писать запуски
export WANDB_MODE="offline"          # без сети; позже: wandb sync wandb/offline-run-*
```

### Comet

```bash
export COMET_API_KEY="..."           # ключ: https://www.comet.com/account-settings/apiKeys
export COMET_WORKSPACE="your-workspace"   # необязательно
```

Ключ читается библиотекой `comet_ml` автоматически. Офлайн-режим включается
константой в ноутбуке:

```python
COMET_OFFLINE = True   # архивы пишутся в runs/comet, позже: comet upload <архив>.zip
```

### MLflow

Для локального сервера ключи не нужны: ноутбук сам поднимает
`mlflow server` на `http://127.0.0.1:5000` с SQLite (`mlflow.db`) и артефактами
в `./mlartifacts`.

Если нужен удалённый сервер, замените `MLFLOW_URI` в ячейке настройки MLflow
и передайте учётные данные:

```bash
export MLFLOW_TRACKING_URI="https://mlflow.example.com"
export MLFLOW_TRACKING_USERNAME="user"
export MLFLOW_TRACKING_PASSWORD="password"
# либо токен:
export MLFLOW_TRACKING_TOKEN="..."
```

### Windows PowerShell

```powershell
$env:WANDB_API_KEY = "..."
$env:COMET_API_KEY = "..."
$env:WANDB_MODE = "online"
```

## Запуск

1. Выполните ноутбук сверху вниз.
2. TensorBoard: `tensorboard --logdir runs/tb`, затем откройте
   <http://localhost:6006>.
3. MLflow UI: <http://127.0.0.1:5000> (сервер поднимается в ноутбуке).
4. W&B и Comet: смотрите запуски проекта `seminar-tracking` в веб-интерфейсе.

Ноутбук содержит долгие ячейки обучения; для быстрого просмотра можно
уменьшать число эпох в конфигурации `CFG`.

## Очистка артефактов

Скрипт удаляет только то, что лежит внутри текущего каталога:

```bash
python cleanup_tracking.py --dry-run   # показать план
python cleanup_tracking.py             # удалить runs/, mlruns/, wandb/, БД MLflow и т.д.
python cleanup_tracking.py --data      # вместе со скачанным датасетом
python cleanup_tracking.py --remote    # ВНИМАНИЕ: удалить запуски в облаке W&B и Comet
```

`--remote` необратим и требует рабочего ключа W&B/Comet. Перед ним всегда
полезно выполнить `--dry-run --remote`.

## Частые проблемы

- **`wandb.login()` зависает** — нет `WANDB_API_KEY` и нет сохранённого ключа;
  выполните `wandb login` в терминале или задайте переменную окружения.
- **Comet не принимает ключ** — проверьте, что `COMET_API_KEY` экспортирован
  в том же процессе, из которого запущен Jupyter.
- **Порт 5000 занят** — ноутбук переиспользует уже запущенный MLflow; если это
  чужая программа, измените `MLFLOW_PORT`.
- **MLflow UI показывает старые запуски** — удалите `mlflow.db`, `mlruns/`,
  `mlartifacts/` через `cleanup_tracking.py` и запустите ячейку сервера заново.
- **Нет GPU** — это нормально: ноутбук работает на CPU, GPU-метрики просто не
  логируются.

## Безопасность

- Не коммитьте `.env`, `wandb/`, `mlruns/`, `mlflow.db`, `data/` и чекпоинты —
  всё это уже в `.gitignore`.
- Ключи, попавшие в историю git или в облачный запуск, считаются
  скомпрометированными: отзовите их и выпустите новые.
- W&B и Comet по умолчанию отправляют данные в облако. Для чувствительных
  данных используйте офлайн-режим или self-hosted инсталляцию.
