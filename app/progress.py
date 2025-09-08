# app/progress.py
# Purpose:
# - Read tasks/estimates from progress/roadmap.yaml
# - Compute % complete, remaining hours, and ETA from your average daily time
# - Update a progress badge in README.md
# - Render a burndown chart progress/burndown.png
#
# Commands you will use:
#   bridge progress status
#   bridge progress log --task TRN1 --minutes 90
#   bridge progress done TRN1
#   bridge progress render

from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import csv
import math
import typer
import yaml
from rich.table import Table
from rich.console import Console
import matplotlib
matplotlib.use("Agg")  # use a non-GUI backend so we can save PNG without Tk
import matplotlib.pyplot as plt
console = Console()
app = typer.Typer(no_args_is_help=True)

ROOT = Path(".")
PROG_DIR = ROOT / "progress"
YAML_PATH = PROG_DIR / "roadmap.yaml"
LOG_PATH = PROG_DIR / "daily_log.csv"   # stores (date, minutes) you worked
README = ROOT / "README.md"
BURN_PATH = PROG_DIR / "burndown.png"

def _ensure_files():
    """Create progress folder and log header if missing."""
    PROG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("date,minutes\n", encoding="utf-8")

def _load():
    """Load roadmap YAML and return (data, tasks)."""
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    return data, data.get("tasks", [])

def _save(data):
    """Persist roadmap YAML (after modifications)."""
    YAML_PATH.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def _compute_progress(tasks):
    """Return (pct_done, total_hours, remaining_hours)."""
    total = sum(t["estimate_h"] for t in tasks)
    done_h = sum((t["estimate_h"] if t["done"] else 0) for t in tasks)
    pct = 0 if total == 0 else round(done_h * 100 / total)
    remaining_h = total - sum(int(t.get("actual_h", 0)) for t in tasks)
    return pct, total, max(0, remaining_h)

def _avg_daily_minutes():
    """Average minutes per day based on daily_log.csv."""
    if not LOG_PATH.exists():
        return 0.0
    per_day = {}
    with LOG_PATH.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d = r["date"]
            try:
                m = int(r["minutes"])
            except Exception:
                continue
            per_day[d] = per_day.get(d, 0) + m
    if not per_day:
        return 0.0
    return sum(per_day.values()) / len(per_day)

def _update_readme_badge(pct: int):
    """Insert or replace the shields.io progress badge in README.md."""
    import re
    badge = f"![Progress](https://img.shields.io/badge/progress-{pct}%25-blue)"
    if README.exists():
        txt = README.read_text(encoding="utf-8")
        new, n = re.subn(
            r"!\[Progress\]\(https://img\.shields\.io/badge/progress-[0-9]+%25-[a-zA-Z]+\)",
            badge, txt, count=1
        )
        if n == 0:
            new = badge + "\n\n" + txt
        README.write_text(new, encoding="utf-8")
    else:
        README.write_text(badge + "\n", encoding="utf-8")

def _render_burndown(deadline: str | None, tasks):
    """Save burndown chart comparing ideal vs. actual remaining hours."""
    total = sum(t["estimate_h"] for t in tasks)
    start = date.today()
    end = date.fromisoformat(deadline) if deadline else start + timedelta(days=14)
    days = max(1, (end - start).days)

    # Ideal line: straight descent from total to 0 by the deadline
    ideal_x = [start + timedelta(days=i) for i in range(days + 1)]
    ideal_y = [total * (1 - i / days) for i in range(days + 1)]

    # Actual line: subtract cumulative minutes logged each day
    per_day = {}
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    per_day[date.fromisoformat(r["date"])] = per_day.get(date.fromisoformat(r["date"]), 0) + int(r["minutes"])
                except Exception:
                    pass

    actual_x, actual_y = [], []
    spent_h = 0.0
    for i in range(days + 1):
        d = start + timedelta(days=i)
        spent_h += per_day.get(d, 0) / 60.0
        actual_x.append(d)
        actual_y.append(max(0.0, total - spent_h))

    plt.figure()
    plt.plot(ideal_x, ideal_y, label="Ideal")
    plt.plot(actual_x, actual_y, label="Actual")
    plt.title("Burndown (remaining hours)")
    plt.xlabel("Date")
    plt.ylabel("Hours remaining")
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(BURN_PATH)
    plt.close()

@app.command()
def status():
    """Show table of tasks + % complete + remaining hours + ETA; update badge & burndown."""
    _ensure_files()
    data, tasks = _load()
    pct, total_h, remaining_h = _compute_progress(tasks)
    avg_h_per_day = _avg_daily_minutes() / 60.0
    if remaining_h == 0:
        eta = "Done"
    elif avg_h_per_day > 0:
        days_left = math.ceil(remaining_h / avg_h_per_day)
        eta = f"{days_left} day(s) (≈ {(date.today() + timedelta(days=days_left)).isoformat()})"
    elif data.get("deadline"):
        eta = f"by {data['deadline']} (no daily speed yet)"
    else:
        eta = "—"

    table = Table(title="Progress")
    table.add_column("Task")
    table.add_column("Est. (h)", justify="right")
    table.add_column("Actual (h)", justify="right")
    table.add_column("Status")
    for t in tasks:
        table.add_row(
            f"{t['id']} — {t['title']}",
            str(t["estimate_h"]),
            str(int(t.get("actual_h", 0))),
            "✅" if t["done"] else "…"
        )
    console.print(table)
    console.rule("[bold]Summary")
    console.print(f"Planned: [bold]{total_h} h[/] · Remaining: [bold]{remaining_h} h[/] · "
                  f"Progress: [bold]{pct}%[/] · ETA: [bold]{eta}[/]")

    _update_readme_badge(pct)
    _render_burndown(data.get("deadline"), tasks)
    console.print(f"\nREADME badge updated and burndown saved → [italic]{BURN_PATH}[/]")

@app.command()
def log(task: str = typer.Option(..., "--task", "-t"),
        minutes: int = typer.Option(..., "--minutes", "-m")):
    """Append time spent on a task (in minutes) to daily_log.csv and bump task.actual_h."""
    _ensure_files()
    data, tasks = _load()
    item = next((x for x in tasks if x["id"] == task), None)
    if not item:
        typer.secho("Task not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    item["actual_h"] = int(item.get("actual_h", 0)) + minutes // 60
    _save(data)
    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([date.today().isoformat(), minutes])
    typer.secho(f"+{minutes} min on {task}", fg=typer.colors.GREEN)

@app.command()
def done(task: str):
    """Mark a task as done; if no actual_h yet, credit the estimate."""
    data, tasks = _load()
    item = next((x for x in tasks if x["id"] == task), None)
    if not item:
        typer.secho("Task not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    item["done"] = True
    if int(item.get("actual_h", 0)) == 0:
        item["actual_h"] = int(item["estimate_h"])
    _save(data)
    typer.secho(f"Task {task} marked as done.", fg=typer.colors.GREEN)

@app.command()
def render():
    """Recreate README badge and burndown without printing the table."""
    data, tasks = _load()
    pct, _, _ = _compute_progress(tasks)
    _update_readme_badge(pct)
    _render_burndown(data.get("deadline"), tasks)
    console.print(f"OK → badge and {BURN_PATH}")
