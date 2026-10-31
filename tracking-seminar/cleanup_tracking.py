"""Очистка артефактов трекинга для seminar_tracking.ipynb.

Использование в ноутбуке (первая ячейка):

    from cleanup_tracking import clean_all
    clean_all()                          # только локальные файлы, данные FashionMNIST сохраняются
    clean_all(dry_run=True)              # только показать, что будет удалено
    clean_all(remote=True)               # + удалить запуски в облаке W&B и Comet

Использование из терминала:

    python cleanup_tracking.py [--dry-run] [--data] [--remote] [--port 5000]

Что чистится
------------
Локально (всё внутри каталога, где лежит ноутбук):
    runs/                      TensorBoard, примитивный логгер, офлайн-архивы Comet
    mlflow.db, -wal, -shm      база MLflow (запуски, реестр моделей)
    mlartifacts/, mlruns/      артефакты и модели MLflow
    mlflow_server.log          лог сервера MLflow
    wandb/                     локальные и офлайн-запуски W&B
    .cometml-runs/             офлайн-архивы Comet по умолчанию
    checkpoints/               чекпоинты моделей
    cm.png, confusion-matrix.json
    data/                      датасет FashionMNIST — ТОЛЬКО с флагом include_data=True

Процессы:
    сервер MLflow на заданном порту останавливается ДО удаления базы.

Облако (только с remote=True, необратимо):
    W&B   — все запуски проекта вместе с артефактами
    Comet — все эксперименты проекта
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path.cwd().resolve()

LOCAL_DIRS = [
    "runs",
    "mlartifacts",
    "mlruns",
    "wandb",
    ".cometml-runs",
    "checkpoints",
]
LOCAL_FILES = [
    "mlflow.db",
    "mlflow.db-wal",
    "mlflow.db-shm",
    "mlflow_server.log",
    "cm.png",
    "confusion-matrix.json",
]
DATA_DIR = "data"

PROJECT = "seminar-tracking"


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #
def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def _is_inside(path: Path, base: Path) -> bool:
    """Страховка: удаляем только то, что лежит внутри рабочего каталога."""
    try:
        path.resolve().relative_to(base)
        return path.resolve() != base
    except ValueError:
        return False


def _remove(path: Path, base: Path, dry_run: bool, verbose: bool) -> bool:
    if not (path.exists() or path.is_symlink()):
        return False
    if not _is_inside(path, base):
        _log(f"  ПРОПУЩЕНО (вне рабочего каталога): {path}", verbose)
        return False
    if dry_run:
        _log(f"  [dry-run] удалил бы: {path.relative_to(base)}", verbose)
        return True
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    _log(f"  удалено: {path.relative_to(base)}", verbose)
    return True


# --------------------------------------------------------------------------- #
# MLflow: остановка сервера
# --------------------------------------------------------------------------- #
def _looks_like_mlflow(cmd: str) -> bool:
    return (
        "-m mlflow server" in cmd
        or "mlflow.server" in cmd
        or "mlflow server" in cmd
    )


def stop_mlflow_server(port: int = 5000, dry_run: bool = False, verbose: bool = True) -> list[int]:
    """Останавливает сервер MLflow, слушающий порт, вместе с дочерними процессами.

    Трогает только процессы, чья командная строка похожа на MLflow,
    чужие программы на том же порту не завершаются.
    """
    try:
        import psutil
    except ImportError:
        _log("  psutil не установлен — сервер MLflow не останавливаю", verbose)
        return []

    me = os.getpid()
    try:
        skip = {me} | {p.pid for p in psutil.Process(me).parents()}
    except psutil.Error:
        skip = {me}

    # Процессы, держащие порт
    listening: set[int] = set()
    try:
        for conn in psutil.net_connections(kind="inet"):
            if (conn.status == psutil.CONN_LISTEN and conn.laddr
                    and conn.laddr.port == port and conn.pid):
                listening.add(conn.pid)
    except (psutil.Error, PermissionError):
        pass

    # Всё дерево MLflow: сам сервер, uvicorn, job-runner, huey
    victims: dict[int, "psutil.Process"] = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.pid in skip:
            continue
        cmd = " ".join(proc.info["cmdline"] or [])
        if _looks_like_mlflow(cmd):
            victims[proc.pid] = proc
    # Чужой процесс на нашем порту НЕ трогаем, только предупреждаем
    foreign = sorted(listening - set(victims) - skip)
    if foreign:
        _log(f"  внимание: порт {port} занят не MLflow (pid {foreign}) — не завершаю", verbose)

    if not victims:
        _log("  сервер MLflow не запущен", verbose)
        return []

    pids = sorted(victims)
    if dry_run:
        _log(f"  [dry-run] остановил бы MLflow, pid: {pids}", verbose)
        return pids

    for proc in victims.values():
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(victims.values()), timeout=10)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)
    _log(f"  остановлен сервер MLflow, pid: {pids}", verbose)
    time.sleep(0.5)   # дать ОС освободить файл базы и порт
    return pids


# --------------------------------------------------------------------------- #
# Облако (необратимо!)
# --------------------------------------------------------------------------- #
def clean_wandb_remote(project: str = PROJECT, dry_run: bool = False, verbose: bool = True) -> int:
    """Удаляет все запуски проекта в W&B вместе с их артефактами."""
    try:
        import wandb
    except ImportError:
        _log("  wandb не установлен — пропуск", verbose)
        return 0
    try:
        api = wandb.Api()
        entity = api.default_entity
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as e:   # нет ключа, нет сети, проекта нет
        _log(f"  W&B недоступен или проекта нет ({type(e).__name__}: {e})", verbose)
        return 0

    for run in runs:
        if dry_run:
            _log(f"  [dry-run] удалил бы W&B run: {run.name} ({run.id})", verbose)
        else:
            try:
                run.delete(delete_artifacts=True)
                _log(f"  удалён W&B run: {run.name} ({run.id})", verbose)
            except Exception as e:
                _log(f"  не удалось удалить {run.id}: {e}", verbose)
    if not runs:
        _log(f"  в W&B проекте {entity}/{project} запусков нет", verbose)
    return len(runs)


def clean_comet_remote(project: str = PROJECT, dry_run: bool = False, verbose: bool = True) -> int:
    """Удаляет все эксперименты проекта в Comet."""
    try:
        from comet_ml.api import API
    except ImportError:
        _log("  comet_ml не установлен — пропуск", verbose)
        return 0
    try:
        api = API()
        workspace = api.get_default_workspace()
        experiments = api.get_experiments(workspace, project_name=project)
    except Exception as e:   # нет ключа, нет сети, проекта нет
        _log(f"  Comet недоступен или проекта нет ({type(e).__name__}: {e})", verbose)
        return 0

    for exp in experiments:
        key, name = exp.key, getattr(exp, "name", "")
        if dry_run:
            _log(f"  [dry-run] удалил бы Comet experiment: {name} ({key})", verbose)
        else:
            try:
                api.delete_experiment(key)
                _log(f"  удалён Comet experiment: {name} ({key})", verbose)
            except Exception as e:
                _log(f"  не удалось удалить {key}: {e}", verbose)
    if not experiments:
        _log(f"  в Comet проекте {workspace}/{project} экспериментов нет", verbose)
    return len(experiments)


# --------------------------------------------------------------------------- #
# Главная функция
# --------------------------------------------------------------------------- #
def clean_all(
    include_data: bool = False,
    remote: bool = False,
    dry_run: bool = False,
    mlflow_port: int = 5000,
    project: str = PROJECT,
    base_dir: str | os.PathLike | None = None,
    verbose: bool = True,
) -> dict:
    """Стирает артефакты TensorBoard, MLflow, W&B и Comet.

    include_data — удалить и скачанный датасет ./data (по умолчанию сохраняется).
    remote       — удалить запуски в облаке W&B и Comet (необратимо).
    dry_run      — ничего не удалять, только вывести план.
    """
    base = Path(base_dir).resolve() if base_dir else BASE_DIR
    report = {"removed": [], "mlflow_pids": [], "wandb_runs": 0, "comet_experiments": 0}
    mode = "[DRY-RUN] " if dry_run else ""
    _log(f"{mode}Очистка артефактов в {base}", verbose)

    _log("• MLflow: остановка сервера", verbose)
    report["mlflow_pids"] = stop_mlflow_server(mlflow_port, dry_run, verbose)

    _log("• Локальные файлы (TensorBoard, MLflow, W&B, Comet)", verbose)
    targets = [base / d for d in LOCAL_DIRS] + [base / f for f in LOCAL_FILES]
    if include_data:
        targets.append(base / DATA_DIR)
    else:
        _log(f"  ./{DATA_DIR} сохранён (include_data=True — удалить)", verbose)

    for path in targets:
        if _remove(path, base, dry_run, verbose):
            report["removed"].append(str(path.relative_to(base)))
    if not report["removed"]:
        _log("  чисто: удалять нечего", verbose)

    if remote:
        _log("• Облако: W&B (необратимо)", verbose)
        report["wandb_runs"] = clean_wandb_remote(project, dry_run, verbose)
        _log("• Облако: Comet (необратимо)", verbose)
        report["comet_experiments"] = clean_comet_remote(project, dry_run, verbose)

    _log(f"{mode}Готово.", verbose)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Очистка артефактов трекинга семинара")
    ap.add_argument("--dry-run", action="store_true", help="только показать, что будет удалено")
    ap.add_argument("--data", action="store_true", help="удалить и ./data (датасет)")
    ap.add_argument("--remote", action="store_true", help="удалить запуски в облаке W&B и Comet")
    ap.add_argument("--port", type=int, default=5000, help="порт сервера MLflow")
    ap.add_argument("--project", default=PROJECT, help="имя проекта в W&B/Comet")
    args = ap.parse_args()
    clean_all(include_data=args.data, remote=args.remote, dry_run=args.dry_run,
              mlflow_port=args.port, project=args.project)
    sys.exit(0)
