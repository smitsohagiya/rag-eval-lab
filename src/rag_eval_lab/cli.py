"""Command-line entry point."""

import typer

from rag_eval_lab import __version__
from rag_eval_lab.config import get_settings
from rag_eval_lab.log import configure_logging

app = typer.Typer(help="Run RAG retrieval experiments and evaluations.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Set up logging before any command runs."""
    configure_logging(get_settings().log_level)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
