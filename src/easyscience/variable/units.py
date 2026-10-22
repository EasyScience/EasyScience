# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Helpers for how units are stored and displayed.

Scipp can display a unit in two ways that surprise a user, and they need
different remedies:

1. It prints a **numeric factor** because it has no name for a scaled unit, so
   ``m/mm`` becomes ``1000`` and ``dm*m`` becomes ``0.1m^2``. The magnitude is then
   hiding inside the unit. This module fixes that, by folding the factor back into
   the value.
2. It prints **a different name than the one that was written**, so ``angstrom``
   becomes ``Å`` and ``nm*m/s`` becomes ``nGy*s``. That is a display concern only, and
   is handled by ``UnitSpellingMixin``, which lets the descriptors remember the
   spelling a unit arrived as.

The descriptors' ``unit`` property is therefore a *display* string;
``_scalar.unit`` / ``_array.unit`` remains the source of truth for
meaning. The two never disagree about what the unit is, only about how
it is spelled.
"""

from __future__ import annotations

import re
from typing import Any
from typing import Optional
from typing import Union

import scipp as sc

# Scipp writes a scaled unit it cannot name as a numeric factor followed by the
# dimensions, in any of the forms '1000', '0.1m^2', '2.57e-44*J^2' and '3.9e+43/J^2'.

# ^: start of string
# \s*: optional whitespace
# (?P<number>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?): a number, possibly with decimal point and exponent, captured as 'number'
# \s*: optional whitespace
# (?P<separator>[*/])?: an optional separator, either '*' or '/', captured as 'separator'
# \s*: optional whitespace
# (?P<rest>.*): the rest of the string, captured as 'rest'
_NUMERIC_FACTOR = re.compile(
    r'^\s*(?P<number>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(?P<separator>[*/])?\s*(?P<rest>.*)$'
)


def has_numeric_factor(unit: Union[str, sc.Unit]) -> bool:
    """
    Report whether a unit carries a numeric factor in the way it is
    written.

    Applied to a ``sc.Unit`` this asks how scipp displays it; applied to
    a string it asks how the user wrote it, which is how a spelling such
    as '10dm^2' is rejected while 'dm*m' is accepted.

    Parameters
    ----------
    unit : Union[str, sc.Unit]
        Unit to inspect.

    Returns
    -------
    bool
        True if the unit is written with a leading multiplier.
    """
    match = _NUMERIC_FACTOR.match(str(unit))
    if match is None:
        return False
    if match.group('rest') == '' and match.group('separator') is None:
        # A pure number, such as '1000'.
        return True
    # The leading '1' of '1/meV' is a placeholder for the numerator, not a factor.
    return float(match.group('number')) != 1.0


def si_base_unit(unit: sc.Unit) -> sc.Unit:
    """
    Return the unit with the same dimensions as ``unit`` but no
    multiplier.

    The dimensions are read from ``sc.Unit.to_dict()``, which stores the
    base units and powers, which is more robust than parsing the string.

    Parameters
    ----------
    unit : sc.Unit
        Unit to reduce.

    Returns
    -------
    sc.Unit
        The corresponding SI base unit, or dimensionless if the unit has
        no dimensions.
    """
    powers = unit.to_dict().get('powers', {})
    if not powers:
        return sc.units.dimensionless
    return sc.Unit('*'.join(f'{base}^{power}' for base, power in powers.items()))


def normalisation_target(unit: sc.Unit, has_spelling: bool = False) -> Optional[str]:
    """
    Return the unit to convert to so that no magnitude is left inside
    the unit.

    A unit which reduces to a pure number, such as 'm/mm', is always
    folded away.

    Beyond that, only units with nothing better to display are
    converted. If the user's own spelling is being kept
    (``has_spelling``), then nothing numeric is on show and there is
    nothing to fix, so a parameter created with '1/meV^2' keeps both its
    spelling and its value, even though scipp would print it as
    '3.9e+43/J^2'.

    Parameters
    ----------
    unit : sc.Unit
        Unit to inspect.
    has_spelling : bool, default=False
        Whether a clean spelling for this unit is being displayed in its
        place.

    Returns
    -------
    Optional[str]
        The unit to convert to, or None if there is nothing to fix, or
        if no safe target could be determined.
    """
    if unit.to_dict().get('e_flag'):
        # Offset units such as degC cannot be converted this way.
        return None
    if not unit.to_dict().get('powers'):
        # A pure number, such as the 1000 that 'm/mm' reduces to.
        return None if unit == sc.units.dimensionless else 'dimensionless'
    if has_spelling or not has_numeric_factor(unit):
        return None
    base_unit = si_base_unit(unit)
    target = str(base_unit)
    try:
        round_trip = sc.Unit(target)
    except sc.UnitError:
        return None
    if round_trip != base_unit:
        # The target does not survive a round trip, so it is not safe to use.
        return None
    return target


def set_unit_state(obj: Any, state: tuple) -> None:
    """
    Restore a unit state on the undo stack.

    A module level function rather than a closure, so that the call
    dispatches to the ``_restore_unit_state`` of whichever class owns
    the object.

    Parameters
    ----------
    obj : Any
        Object to restore the state on.
    state : tuple
        State captured by ``_unit_state``.
    """
    obj._restore_unit_state(state)


class UnitSpellingMixin:
    """
    Remember the spelling a descriptor's unit was given with, so that it
    can be displayed that way instead of with scipp's name for it.

    The spelling is for display only. The scipp unit held by the
    descriptor remains the source of truth, and the spelling is only
    reported while it still parses to exactly that unit.
    """

    _input_unit: Optional[str] = None
    _input_unit_parsed: Optional[sc.Unit] = None

    def _remember_unit(self, unit: Union[str, sc.Unit, None]) -> None:
        """
        Remember the spelling a unit was given with, for display
        purposes.

        A ``sc.Unit`` is spelled the way scipp prints it, since scipp
        does not keep the string it was created from. The same rules
        then apply however the unit was supplied: a unit which is
        written with a numeric factor, such as '10dm^2', or which scipp
        prints with one, such as ``sc.Unit('dm*m')`` ('0.1m^2'), has no
        spelling worth keeping.

        Nothing is remembered for a dimensionless unit either: reporting
        'one' or '' instead of 'dimensionless' would break the
        comparisons against 'dimensionless' made throughout the
        descriptors.

        Parameters
        ----------
        unit : Union[str, sc.Unit, None]
            Unit as it was supplied.
        """
        self._input_unit = None
        self._input_unit_parsed = None
        if unit is None:
            return
        spelling = str(unit).strip()
        if not spelling or has_numeric_factor(spelling):
            return
        try:
            parsed_unit = sc.Unit(spelling)
        except sc.UnitError:
            return
        if parsed_unit == sc.units.dimensionless:
            return
        self._input_unit = spelling
        self._input_unit_parsed = parsed_unit

    def _spelled_unit(self, unit: sc.Unit) -> str:
        """
        Return ``unit`` as a string, using the remembered spelling while
        it still describes that unit.

        Parameters
        ----------
        unit : sc.Unit
            The unit currently held by the descriptor.

        Returns
        -------
        str
            The remembered spelling, or scipp's name for ``unit``.
        """
        if self._input_unit_parsed is not None and self._input_unit_parsed == unit:
            return self._input_unit
        return str(unit)

    def _spelling_state(self) -> tuple:
        """
        Capture the remembered spelling, for the undo stack.

        Returns
        -------
        tuple
            Opaque state, to be passed back to
            ``_restore_spelling_state``.
        """
        return (self._input_unit, self._input_unit_parsed)

    def _restore_spelling_state(self, state: tuple) -> None:
        """
        Restore state captured by ``_spelling_state``.

        Parameters
        ----------
        state : tuple
            State to restore.
        """
        self._input_unit, self._input_unit_parsed = state

    @staticmethod
    def _spelling_from_sources(unit: sc.Unit, sources: tuple) -> Union[str, sc.Unit]:
        """
        Return an operand's spelling for ``unit``, if one of them has
        the same unit.

        An operation such as an addition, or a multiplication by a plain
        number, leaves the unit untouched, and the result should be
        displayed the way its operands were rather than falling back to
        scipp's name for it. Only an exactly equal unit is used, so a
        result can never be relabelled as something it is not.

        Parameters
        ----------
        unit : sc.Unit
            Unit of the result.
        sources : tuple
            Operands of the operation. Anything which is not a
            descriptor, such as a plain number, is ignored.

        Returns
        -------
        Union[str, sc.Unit]
            The operand's spelling, or ``unit`` unchanged if no operand
            offers one.
        """
        for source in sources:
            parsed_unit = getattr(source, '_input_unit_parsed', None)
            if parsed_unit is not None and parsed_unit == unit:
                return source._input_unit
        return unit
