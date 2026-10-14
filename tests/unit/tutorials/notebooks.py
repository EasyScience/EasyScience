# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers to read and replay the tutorial notebooks without a Jupyter kernel.

The notebooks keep no stored outputs. Their exercise solutions are cells that start with the
``%%solution`` cell magic, which runs the solution and shows the code with its output. The
helpers here replay a notebook cell by cell, with a minimal stand-in for the IPython shell, and
hand back what every solution produced.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TUTORIALS_DIR = PROJECT_ROOT / 'docs' / 'docs' / 'tutorials'

MATHS_NOTEBOOKS = [
    'basics/maths-arithmetic',
    'basics/maths-exp-log',
    'basics/maths-trigonometric',
    'basics/maths-inverse-trigonometric',
    'basics/maths-hyperbolic',
]
PROPAGATION_NOTEBOOKS = [
    'basics/maths-error-propagation',
    'basics/maths-correlation',
]
FITTING_NOTEBOOKS = ['fitting-intro']
NOTEBOOKS = MATHS_NOTEBOOKS + PROPAGATION_NOTEBOOKS + FITTING_NOTEBOOKS

# These notebooks download their data, so the unit tests read them but never run them. The
# tutorial CI job executes every notebook, including these.
NETWORK_NOTEBOOKS = [
    'fitting/sans-sphere',
    'fitting/qens-lorentzian',
    'fitting/bayesian-qens',
]
STRUCTURE_NOTEBOOKS = NOTEBOOKS + NETWORK_NOTEBOOKS

# Number of exercises (and so of folded solutions) in each notebook.
EXERCISES = {
    'basics/maths-arithmetic': 4,
    'basics/maths-exp-log': 4,
    'basics/maths-trigonometric': 4,
    'basics/maths-inverse-trigonometric': 4,
    'basics/maths-hyperbolic': 4,
    'basics/maths-error-propagation': 4,
    'basics/maths-correlation': 4,
    'fitting-intro': 2,
    'fitting/sans-sphere': 4,
    'fitting/qens-lorentzian': 4,
    'fitting/bayesian-qens': 3,
}

SOLUTION_MAGIC = '%%solution'
HIDDEN_TAG = 'hide-in-docs'


def load_notebook(name: str) -> dict:
    """Read a tutorial notebook, given as its path below the tutorials folder without suffix."""
    return json.loads((TUTORIALS_DIR / f'{name}.ipynb').read_text(encoding='utf-8'))


def source_of(cell: dict) -> str:
    """Return the source of a notebook cell as one string."""
    return ''.join(cell['source'])


def is_solution_cell(cell: dict) -> bool:
    """Whether the cell is a code cell that runs a folded exercise solution."""
    return cell['cell_type'] == 'code' and source_of(cell).startswith(SOLUTION_MAGIC)


def is_hidden_helper(cell: dict) -> bool:
    """Whether the cell is the hidden code cell that defines the notebook helpers."""
    tags = cell['metadata'].get('tags', [])
    return cell['cell_type'] == 'code' and HIDDEN_TAG in tags and not is_solution_cell(cell)


class FakeShell:
    """The part of an IPython shell that the notebook helpers use."""

    def __init__(self) -> None:
        self.user_ns: dict = {}
        self.magics: dict = {}

    def register_magic_function(self, func, magic_kind='line', magic_name=None) -> None:
        self.magics[(magic_kind, magic_name or func.__name__)] = func


@dataclass
class Solution:
    """What one ``%%solution`` cell showed."""

    notebook: str
    exercise: str
    note: str
    code: str
    html: str
    output: str | None


_HIGHLIGHT = re.compile(r'<div class="highlight"><pre>(.*?)</pre></div>', re.S)
_OUTPUT = re.compile(r'<p>Output:</p><pre>(.*?)</pre>', re.S)
_TAGS = re.compile(r'<[^>]+>')


def parse_solution_html(page: str) -> tuple[str, str | None]:
    """Return the code and the output text from the HTML of a folded solution."""
    code = _HIGHLIGHT.search(page)
    output = _OUTPUT.search(page)
    return (
        html.unescape(_TAGS.sub('', code.group(1))) if code else '',
        html.unescape(output.group(1)) if output else None,
    )
