# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Structure of the maths and fitting tutorial notebooks.

The notebooks are downloadable and are rendered by the documentation build, so they have to
stay free of stored outputs, of template delimiters and of authoring leftovers.
"""

import re

import pytest

from .notebooks import EXERCISES
from .notebooks import PROJECT_ROOT
from .notebooks import STRUCTURE_NOTEBOOKS
from .notebooks import TUTORIALS_DIR
from .notebooks import is_hidden_helper
from .notebooks import is_solution_cell
from .notebooks import load_notebook
from .notebooks import source_of

pytestmark = pytest.mark.fast


@pytest.fixture(params=STRUCTURE_NOTEBOOKS)
def name(request):
    return request.param


@pytest.fixture
def cells(name):
    return load_notebook(name)['cells']


def _exercise_blocks(cells):
    """Return the indices of the exercise headings."""
    return [
        i
        for i, cell in enumerate(cells)
        if cell['cell_type'] == 'markdown' and source_of(cell).startswith('### Exercise')
    ]


class TestNoStoredState:
    def test_notebook_is_nbformat_4(self, name):
        notebook = load_notebook(name)

        assert notebook['nbformat'] == 4
        assert notebook['nbformat_minor'] >= 5

    def test_no_outputs_are_stored(self, cells):
        code_cells = [cell for cell in cells if cell['cell_type'] == 'code']

        assert code_cells
        assert all(cell['outputs'] == [] for cell in code_cells)

    def test_cell_ids_are_unique(self, cells):
        ids = [cell['id'] for cell in cells]

        assert len(ids) == len(set(ids))


class TestNoLeftovers:
    def test_no_template_delimiters_in_the_file(self, name):
        # The documentation build runs a template engine over the raw notebook text.
        text = (TUTORIALS_DIR / f'{name}.ipynb').read_text(encoding='utf-8')

        assert not re.search(r'\{\{|\{%|\{#', text)

    def test_no_authoring_markers(self, cells):
        text = '\n'.join(source_of(cell) for cell in cells)

        assert not re.search(r'\b(NOTE|TODO|FIXME|XXX)\b\(?', text)

    def test_the_old_pasted_results_are_gone(self, cells):
        text = '\n'.join(source_of(cell) for cell in cells)

        assert 'Expected result' not in text


class TestExercises:
    def test_number_of_exercises(self, name, cells):
        assert len(_exercise_blocks(cells)) == EXERCISES[name]

    def test_every_exercise_has_givens_hints_and_a_solution_in_that_order(self, cells):
        for start in _exercise_blocks(cells):
            givens, hints, solution = cells[start + 1 : start + 4]

            assert givens['cell_type'] == 'code'
            assert not is_solution_cell(givens)
            assert hints['cell_type'] == 'markdown'
            assert is_solution_cell(solution)

    def test_hints_are_folded_markdown_and_hold_no_solution(self, cells):
        for start in _exercise_blocks(cells):
            hints = source_of(cells[start + 2])

            assert '<summary>Hint 1</summary>' in hints
            assert hints.count('<details class="question">') == hints.count('</details>')
            assert '<summary>Solution' not in hints
            assert '```' not in hints

    def test_solution_cells_are_hidden_in_the_docs_and_have_code(self, cells):
        solutions = [cell for cell in cells if is_solution_cell(cell)]

        assert solutions
        for cell in solutions:
            assert cell['metadata'].get('tags') == ['hide-in-docs']
            header, _, code = source_of(cell).partition('\n')
            assert header.startswith('%%solution')
            assert code.strip()

    def test_solution_cells_keep_their_note_on_one_line(self, cells):
        for cell in cells:
            if is_solution_cell(cell):
                assert '\n' not in source_of(cell).partition('\n')[0]


class TestHelperCell:
    def test_there_is_exactly_one_hidden_helper_cell(self, cells):
        assert sum(is_hidden_helper(cell) for cell in cells) == 1

    def test_the_helper_is_defined_before_the_first_solution(self, cells):
        helper = next(i for i, cell in enumerate(cells) if is_hidden_helper(cell))
        first = next(i for i, cell in enumerate(cells) if is_solution_cell(cell))

        assert helper < first

    def test_the_helper_defines_and_registers_the_solution_magic(self, cells):
        source = source_of(next(cell for cell in cells if is_hidden_helper(cell)))

        assert 'def solution(' in source
        assert "register_magic_function(solution, magic_kind='cell')" in source

    def test_the_helper_has_no_comments(self, cells):
        source = source_of(next(cell for cell in cells if is_hidden_helper(cell)))

        assert not [line for line in source.splitlines() if line.lstrip().startswith('#')]


class TestSite:
    @pytest.mark.parametrize('name', STRUCTURE_NOTEBOOKS)
    def test_notebook_is_in_the_navigation(self, name):
        config = (PROJECT_ROOT / 'docs' / 'mkdocs.yml').read_text(encoding='utf-8')

        assert f'{name}.ipynb' in config
