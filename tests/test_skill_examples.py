"""Type check the Python code examples embedded in agent skill files."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _python_examples() -> list[pytest.ParameterSet]:
    """Collect fenced Python code blocks from every skill file."""
    examples = []
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        blocks = re.findall(r"```python\n(.*?)```", skill_md.read_text(), flags=re.DOTALL)
        for index, block in enumerate(blocks):
            examples.append(pytest.param(block, id=f"{skill_md.parent.name}-{index}"))
    return examples


@pytest.mark.parametrize("example", _python_examples())
def test_skill_example_type_checks(example: str, tmp_path: Path) -> None:
    """Skill code examples should be self-contained and type check against the installed zeus package."""
    if shutil.which("ty") is None:
        pytest.skip("ty is not installed")

    example_file = tmp_path / "example.py"
    example_file.write_text(example)
    result = subprocess.run(
        ["ty", "check", "--python", sys.executable, str(example_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
