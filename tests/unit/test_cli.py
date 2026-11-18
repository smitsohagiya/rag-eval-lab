from typer.testing import CliRunner

from rag_eval_lab import __version__
from rag_eval_lab.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
