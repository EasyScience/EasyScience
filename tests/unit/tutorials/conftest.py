# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import types

import IPython
import IPython.display
import pytest

from easyscience import global_object

from .notebooks import SOLUTION_MAGIC
from .notebooks import FakeShell
from .notebooks import Solution
from .notebooks import is_hidden_helper
from .notebooks import is_solution_cell
from .notebooks import load_notebook
from .notebooks import parse_solution_html
from .notebooks import source_of


@pytest.fixture
def notebook_environment():
    """Patch IPython and plotly so that notebook cells can run without a kernel.

    Yields the fake shell and the list that collects everything the notebooks display.
    """
    shown: list = []
    shell = FakeShell()
    plotly = types.ModuleType('plotly')
    graph_objects = types.ModuleType('plotly.graph_objects')
    graph_objects.Figure = object
    plotly.graph_objects = graph_objects
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(IPython, 'get_ipython', lambda: shell)
        patch.setattr(IPython.display, 'display', shown.append)
        patch.setitem(sys.modules, 'plotly', plotly)
        patch.setitem(sys.modules, 'plotly.graph_objects', graph_objects)
        yield shell, shown


@pytest.fixture
def run_notebook(notebook_environment):
    """Return a function that replays a notebook and returns its solutions, in order.

    Cells that draw a chart are skipped, as no exercise depends on them.
    """
    shell, shown = notebook_environment

    def run(name: str) -> list[Solution]:
        global_object.map._clear()
        shell.user_ns.clear()
        solutions: list[Solution] = []
        exercise = ''
        for cell in load_notebook(name)['cells']:
            source = source_of(cell)
            if cell['cell_type'] == 'markdown':
                if source.startswith('### Exercise'):
                    exercise = source.splitlines()[0].removeprefix('### ')
                continue
            if is_solution_cell(cell):
                header, _, code = source.partition('\n')
                note = header.removeprefix(SOLUTION_MAGIC).strip()
                shown.clear()
                shell.magics[('cell', 'solution')](note, code)
                assert len(shown) == 1, 'a solution must display exactly one object'
                page = shown[0].data
                shown_code, output = parse_solution_html(page)
                solutions.append(Solution(name, exercise, note, shown_code, page, output))
            elif 'show_figure(' not in source or is_hidden_helper(cell):
                exec(source, shell.user_ns)
        return solutions

    return run


@pytest.fixture
def solutions(run_notebook):
    """Return a function that replays a notebook once and returns its solutions."""
    cache: dict[str, list[Solution]] = {}

    def get(name: str) -> list[Solution]:
        if name not in cache:
            cache[name] = run_notebook(name)
        return cache[name]

    return get
