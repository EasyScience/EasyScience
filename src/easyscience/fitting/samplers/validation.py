# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Run-settings validation for the DREAM sampler."""

from __future__ import annotations


def validate_run_settings(samples: int, burn: int, thin: int) -> None:
    """Validate the DREAM run settings.

    Parameters
    ----------
    samples : int
        Number of raw samples to draw; must be a positive integer.
    burn : int
        Burn-in generations to discard; must be a non-negative integer.
    thin : int
        Thinning interval; must be a positive integer.

    Raises
    ------
    ValueError
        If any value is out of range or not an integer.
    """
    # bool is a subclass of int, so ``samples=True`` would otherwise pass as
    # ``samples=1``; these checks are strict (``10.0`` is rejected), so
    # booleans must be rejected too.
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError('samples must be a positive integer.')
    if not isinstance(burn, int) or isinstance(burn, bool) or burn < 0:
        raise ValueError('burn must be a non-negative integer.')
    if not isinstance(thin, int) or isinstance(thin, bool) or thin < 1:
        raise ValueError('thin must be a positive integer.')
