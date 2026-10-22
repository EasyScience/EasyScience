# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The hub rule of the documentation navigation.

A group of two or more pages in the navigation opens with a hub page, an ``index.md``, so
that clicking the group shows an overview. The rule is a guideline: a group that breaks it
produces a warning, never a failing test.
"""

import warnings

import pytest
import yaml

from .notebooks import PROJECT_ROOT

pytestmark = pytest.mark.fast


def _is_hub(entry) -> bool:
    if not isinstance(entry, dict) or len(entry) != 1:
        return False
    target = next(iter(entry.values()))
    return isinstance(target, str) and target.endswith('index.md')


def groups_without_hub(entries, path=()) -> list[str]:
    """Return the groups of the navigation that have several pages and no hub first."""
    missing = []
    for entry in entries:
        for title, value in entry.items():
            if not isinstance(value, list):
                continue
            here = (*path, title)
            if len(value) >= 2 and not _is_hub(value[0]):
                missing.append(' > '.join(here))
            missing += groups_without_hub(value, here)
    return missing


def warn_about_missing_hubs(entries) -> list[str]:
    """Warn once for every group that breaks the hub rule and return those groups."""
    missing = groups_without_hub(entries)
    for group in missing:
        warnings.warn(
            f'The navigation group "{group}" has several pages but no hub page first.',
            UserWarning,
            stacklevel=2,
        )
    return missing


class TestHubRule:
    def test_a_group_of_several_pages_without_a_hub_is_found(self):
        nav = [{'Basics': [{'Arithmetic': 'a.ipynb'}, {'Functions': 'b.ipynb'}]}]

        assert groups_without_hub(nav) == ['Basics']

    def test_a_group_that_opens_with_a_hub_is_accepted(self):
        nav = [{'Basics': [{'Basics': 'basics/index.md'}, {'Arithmetic': 'a.ipynb'}]}]

        assert groups_without_hub(nav) == []

    def test_a_group_with_a_single_page_needs_no_hub(self):
        nav = [{'Introduction': [{'Introduction': 'introduction/index.md'}]}]

        assert groups_without_hub(nav) == []

    def test_nested_groups_are_checked_too(self):
        nav = [
            {'Tutorials': [{'Tutorials': 'index.md'}, {'Fitting': [{'A': 'a.md'}, {'B': 'b.md'}]}]}
        ]

        assert groups_without_hub(nav) == ['Tutorials > Fitting']

    def test_a_missing_hub_is_a_warning(self):
        nav = [{'Basics': [{'Arithmetic': 'a.ipynb'}, {'Functions': 'b.ipynb'}]}]

        with pytest.warns(UserWarning, match='Basics'):
            warn_about_missing_hubs(nav)

    def test_the_site_navigation_follows_the_hub_rule(self):
        config = (PROJECT_ROOT / 'docs' / 'mkdocs.yml').read_text(encoding='utf-8')
        nav = yaml.load(config, Loader=yaml.BaseLoader)['nav']

        warn_about_missing_hubs(nav)
