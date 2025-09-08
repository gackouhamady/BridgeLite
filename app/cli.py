"""
app/cli.py (excerpt)

Main Typer application for BridgeLite. This file assembles sub-CLIs
(e.g., `progress`, `merchants`) under the single `bridge` command.

Make sure to:
- Define `app = typer.Typer(...)` FIRST.
- Then import submodules and mount them via `app.add_typer(...)`.
"""

# app/cli.py
# Main CLI for BridgeLite.
# Fix: define `app = typer.Typer(...)` BEFORE calling `app.add_typer(...)`.

from pathlib import Path
import typer
from rich.console import Console

# Import your sub-CLI AFTER defining `app` (see below)
# from app import progress as progress_cli  # <-- imported after app is created

console = Console()
app = typer.Typer(no_args_is_help=True)  # MUST be defined before add_typer

from app import merchants as merchants_cli
app.add_typer(merchants_cli.app, name="merchants")

# -----------------------------
# Example pipeline commands
# -----------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path("artifacts")

# Optional: your training modules (make sure these exist)
try:
    from training.train import train as train_fn
    from training.eval import evaluate as eval_fn
    from training.drift import run_drift as drift_fn
except Exception:
    # Soft import: allows CLI to work even if training files aren’t ready yet
    train_fn = lambda **kwargs: {"msg": "train stub"}
    eval_fn = lambda: {"msg": "eval stub"}
    drift_fn = lambda: {"msg": "drift stub"}

@app.command()
def ingest():
    """Create a tiny sample raw file (demo)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "sample.csv").write_text("id,value\n1,42\n", encoding="utf-8")
    console.log(f"[green]Ingested[/] → {RAW_DIR / 'sample.csv'}")

@app.command()
def preprocess():
    """Create a tiny processed file (demo)."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "sample.clean.csv").write_text("id,value\n1,42\n", encoding="utf-8")
    console.log(f"[green]Preprocessed[/] → {PROC_DIR / 'sample.clean.csv'}")

@app.command()
def train():
    """Call your training code (training/train.py)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = train_fn(artifacts_dir=ARTIFACTS_DIR)
    console.log(f"[bold]Train metrics[/]: {metrics}")

@app.command()
def evaluate():
    """Call your evaluation code (training/eval.py)."""
    report = eval_fn()
    console.log(f"[bold]Eval report[/]: {report}")

@app.command()
def drift():
    """Call your drift check (training/drift.py)."""
    out = drift_fn()
    console.log(f"[bold]Drift[/]: {out}")

@app.command()
def all():
    """Run the whole pipeline: ingest → preprocess → train → evaluate."""
    ingest()
    preprocess()
    train()
    evaluate()

# -----------------------------
# Mount the progress sub-CLI
# -----------------------------
# Import AFTER `app` is defined, then mount it:
from app import progress as progress_cli
app.add_typer(progress_cli.app, name="progress")

def main():
    app()

if __name__ == "__main__":
    main()
