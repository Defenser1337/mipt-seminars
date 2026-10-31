# Семинар: сборка мусора в Python

Разбор того, как в CPython устроено управление памятью: подсчёт ссылок,
циклические ссылки, поколения сборщика мусора, слабые ссылки и диагностика
утечек. Ноутбук состоит из небольших запускаемых примеров со стандартной
библиотекой — внешние сервисы и ключи не нужны.

## Что внутри

| Файл | Назначение |
|---|---|
| `garbage_collector_python.ipynb` | ноутбук с примерами и экспериментами |
| `garbage_collector_python_talk.md` | тезисы доклада и вопросы для защиты |
| `images_garbage_collector_python/` | иллюстрации, которые вставляет ноутбук |
| `requirements.txt` | зависимости для запуска Jupyter (сам семинар — только stdlib) |

## Требования

- Python **3.11+** (ноутбук проверялся на CPython 3.11.16)
- Git
- Jupyter Lab или Jupyter Notebook

Поведение сборщика мусора — деталь реализации CPython. На PyPy и других
интерпретаторах часть примеров с `gc.get_count()`, `gc.get_stats()` и
`gc.is_tracked()` может вести себя иначе.

## Установка

```bash
cd py-gc-seminar

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install -r requirements.txt
```

Регистрация ядра Jupyter:

```bash
python -m ipykernel install --user \
  --name py-gc-seminar \
  --display-name "Python (py-gc-seminar)"
```

Запуск:

```bash
jupyter lab          # или: jupyter notebook
```

В ноутбуке выберите ядро **Python (py-gc-seminar)**.

Через conda:

```bash
conda create -n py-gc-seminar python=3.11 -y
conda activate py-gc-seminar
pip install -r requirements.txt
```

## Как проходить ноутбук

- Выполняйте ячейки **сверху вниз**: примеры импортируют `gc`, `sys`,
  `weakref` и `tracemalloc` по ходу повествования.
- Числа в выводе `gc.collect()` и `gc.get_count()` зависят от окружения
  и от объектов, которые создаёт сам Jupyter. Абсолютные значения не важны —
  важен порядок и сравнение «до/после».
- Ячейка с файлом создаёт `gc_demo.txt` и удаляет его в конце. Если выполнение
  прервалось, файл можно удалить вручную — он исключён из git.
- Для воспроизводимости в отчётах фиксируйте версию Python: `python --version`.

## Очистка

Ноутбук не создаёт тяжёлых артефактов. Достаточно удалить кэш и временный файл:

```bash
rm -rf __pycache__ .ipynb_checkpoints .venv
rm -f gc_demo.txt
```

## Ключи и секреты

Не требуются: все примеры работают локально и не обращаются к сети.
Если будете добавлять свои эксперименты с внешними сервисами, не коммитьте
`.env` — он уже в `.gitignore`.
