# app/cli.py (add these lines)
from app import progress as progress_cli  # imports the commands defined above
app.add_typer(progress_cli.app, name="progress")  # makes `bridge progress ...` available
