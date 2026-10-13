# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The ``%%solution`` cell magic that the tutorial notebooks define in a hidden cell."""

from types import SimpleNamespace

import pytest

from .notebooks import is_hidden_helper
from .notebooks import load_notebook
from .notebooks import parse_solution_html
from .notebooks import source_of

pytestmark = pytest.mark.fast


@pytest.fixture
def helper(notebook_environment):
    """Run the hidden helper cell of a notebook and return the magic, the shell and the output."""
    shell, shown = notebook_environment
    shell.user_ns['go'] = SimpleNamespace(Figure=object)
    cell = next(
        c for c in load_notebook('basics/maths-arithmetic')['cells'] if is_hidden_helper(c)
    )
    exec(source_of(cell), shell.user_ns)
    shown.clear()
    return shell.magics[('cell', 'solution')], shell, shown


class TestRegistration:
    def test_the_magic_is_registered_as_a_cell_magic(self, helper):
        _, shell, _ = helper

        assert ('cell', 'solution') in shell.magics


class TestOutput:
    def test_the_block_is_folded_and_titled_solution(self, helper):
        solution, _, shown = helper

        solution('', 'print(1)')

        assert shown[0].data.startswith('<details class="example"><summary>Solution</summary>')
        assert shown[0].data.endswith('</details>')

    def test_the_code_is_shown_highlighted_and_unchanged(self, helper):
        solution, _, shown = helper

        solution('', '\nvalue = 2 + 3\nprint(value)\n')

        page = shown[0].data
        code, _ = parse_solution_html(page)
        assert '<div class="highlight">' in page
        assert '<span class="' in page
        assert code.strip() == 'value = 2 + 3\nprint(value)'

    def test_the_printed_output_is_shown(self, helper):
        solution, _, shown = helper

        solution('', "print('a')\nprint('b')")

        assert parse_solution_html(shown[0].data)[1] == 'a\nb'

    def test_nothing_is_printed_so_there_is_no_output_section(self, helper):
        solution, _, shown = helper

        solution('', 'value = 1')

        assert 'Output:' not in shown[0].data

    def test_output_is_escaped(self, helper):
        solution, _, shown = helper

        solution('', "print('<b>x</b> & y')")

        assert '<b>' not in shown[0].data
        assert parse_solution_html(shown[0].data)[1] == '<b>x</b> & y'

    def test_the_note_follows_the_output_and_turns_code_spans_into_code(self, helper):
        solution, _, shown = helper

        solution('The `erf` is bounded by <1.', 'print(1)')

        page = shown[0].data
        assert page.index('Output:') < page.index('<code>erf</code>')
        assert '<p>The <code>erf</code> is bounded by &lt;1.</p>' in page

    @pytest.mark.parametrize('note', ['', '   '])
    def test_a_missing_note_adds_nothing(self, helper, note):
        solution, _, shown = helper

        solution(note, 'print(1)')

        assert shown[0].data.count('<p>') == 1


class TestExecution:
    def test_the_solution_can_use_the_names_of_the_notebook(self, helper):
        solution, shell, shown = helper
        shell.user_ns['given'] = 21

        solution('', 'print(given * 2)')

        assert parse_solution_html(shown[0].data)[1] == '42'

    def test_names_made_by_the_solution_do_not_leak_into_the_notebook(self, helper):
        solution, shell, _ = helper

        solution('', 'leaked = 1')

        assert 'leaked' not in shell.user_ns

    def test_names_of_the_notebook_are_not_overwritten(self, helper):
        solution, shell, _ = helper
        shell.user_ns['given'] = 1

        solution('', 'given = 2')

        assert shell.user_ns['given'] == 1

    def test_an_error_in_the_solution_is_raised(self, helper):
        solution, _, shown = helper

        with pytest.raises(ZeroDivisionError):
            solution('', '1 / 0')
        assert shown == []

    def test_standard_output_is_restored_after_an_error(self, helper, capsys):
        solution, _, _ = helper

        with pytest.raises(ValueError, match='boom'):
            solution('', "raise ValueError('boom')")
        print('visible')

        assert capsys.readouterr().out == 'visible\n'
