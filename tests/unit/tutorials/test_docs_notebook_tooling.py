# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Docs tooling that has to cope with tutorials in sub-folders.

The maths and fitting tutorials live in ``tutorials/basics/`` and ``tutorials/fitting/``. The
Colab button of the notebook page template and the ``notebook-strip`` task both used to assume a
flat layout. These tests run the real thing: a real mkdocs build with the real template, and the
real task command from ``pixi.toml``.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from .notebooks import PROJECT_ROOT

OVERRIDES_DIR = PROJECT_ROOT / 'docs' / 'overrides'
COLAB_BASE = 'https://colab.research.google.com/github/easyscience/core/blob/gh-pages/dev/'

TUTORIALS = [
    'progress-callback.ipynb',
    'basics/maths-arithmetic.ipynb',
    'fitting/fitting-intro.ipynb',
]


def write_notebook(path, with_outputs=False):
    """Write a one-cell notebook, optionally with a stored output."""
    outputs = (
        [{'output_type': 'stream', 'name': 'stdout', 'text': ['2\n']}] if with_outputs else []
    )
    cell = {
        'cell_type': 'code',
        'execution_count': 1 if with_outputs else None,
        'metadata': {},
        'source': ['print(2)'],
        'outputs': outputs,
    }
    notebook = {'cells': [cell], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook), encoding='utf-8')


def stored_outputs(path):
    return json.loads(path.read_text(encoding='utf-8'))['cells'][0]['outputs']


def run(command, **kwargs):
    """Run a command and fail with its own output, which ``check=True`` would hide."""
    result = subprocess.run(
        command, capture_output=True, encoding='utf-8', errors='replace', **kwargs
    )
    assert result.returncode == 0, f'{command} failed:\n{result.stdout}\n{result.stderr}'


@pytest.mark.parametrize('use_directory_urls', [True, False], ids=['directory-urls', 'file-urls'])
def test_colab_link_points_at_the_published_notebook(tmp_path, use_directory_urls):
    pytest.importorskip('mkdocs_jupyter')
    pytest.importorskip('material')
    for notebook in TUTORIALS:
        write_notebook(tmp_path / 'docs' / 'tutorials' / notebook)
    (tmp_path / 'docs' / 'index.md').write_text('# Test\n', encoding='utf-8')
    (tmp_path / 'mkdocs.yml').write_text(
        f"""site_name: Test
site_url: https://easyscience.github.io/core
use_directory_urls: {str(use_directory_urls).lower()}
theme:
  name: material
  custom_dir: {json.dumps(str(OVERRIDES_DIR))}
plugins:
  - mkdocs-jupyter:
      execute: false
      include_source: true
extra:
  vars:
    github_repository: easyscience/core
    docs_version: dev
""",
        encoding='utf-8',
    )

    run([sys.executable, '-m', 'mkdocs', 'build', '-q', '-f', str(tmp_path / 'mkdocs.yml')])

    site = tmp_path / 'site'
    for notebook in TUTORIALS:
        folder, stem = Path('tutorials') / Path(notebook).parent, Path(notebook).stem
        page = (
            site / folder / stem / 'index.html'
            if use_directory_urls
            else site / folder / f'{stem}.html'
        )
        links = re.findall(r'href="(https://colab[^"]*)"', page.read_text(encoding='utf-8'))

        assert len(links) == 1, page
        assert links[0].startswith(COLAB_BASE), links[0]
        published = links[0].removeprefix(COLAB_BASE)
        assert (site / published).is_file(), f'{notebook}: Colab link {published} is not published'


def pixi_platform():
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ('arm64', 'aarch64')
    if system == 'Linux':
        return 'linux-aarch64' if arm else 'linux-64'
    if system == 'Darwin':
        return 'osx-arm64' if arm else 'osx-64'
    return 'win-64'


def test_notebook_strip_clears_outputs_at_every_depth(tmp_path):
    pixi = shutil.which('pixi')
    if pixi is None or shutil.which('nbstripout') is None:
        pytest.skip('needs pixi and nbstripout')
    manifest = (PROJECT_ROOT / 'pixi.toml').read_text(encoding='utf-8')
    command = tomllib.loads(manifest)['tasks']['notebook-strip']
    (tmp_path / 'pixi.toml').write_text(
        '[workspace]\n'
        'name = "strip"\n'
        'channels = []\n'
        f'platforms = ["{pixi_platform()}"]\n\n'
        f'[tasks]\nstrip = {json.dumps(command)}\n',
        encoding='utf-8',
    )
    tutorials = tmp_path / 'docs' / 'docs' / 'tutorials'
    stripped = [
        tutorials / name
        for name in ('top.ipynb', 'basics/nested.ipynb', 'fitting/deeper/nested.ipynb')
    ]
    elsewhere = tmp_path / 'docs' / 'docs' / 'other' / 'keep.ipynb'
    for notebook in [*stripped, elsewhere]:
        write_notebook(notebook, with_outputs=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIXI_')}
    env['PATH'] = os.pathsep.join([str(Path(shutil.which('nbstripout')).parent), env['PATH']])

    run(
        [pixi, 'run', '--manifest-path', str(tmp_path / 'pixi.toml'), 'strip'],
        env=env,
        cwd=tmp_path,
    )

    assert [stored_outputs(n) for n in stripped] == [[], [], []]
    assert stored_outputs(elsewhere) != []
