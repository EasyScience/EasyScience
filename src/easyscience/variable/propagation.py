# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
First-order propagation of uncertainties through calculations.

Every result of an operator or of an ``easyscience.math`` function
remembers which *source* parameters it was calculated from and how much
each of them contributes to its uncertainty. A source is a parameter
that is not itself the result of a calculation. The contribution of a
source is ``c = (d result / d source) * sigma_source``, in the unit of
the result, with its sign: it is the derivative with respect to the
source expressed in standard deviations of that source.

The variance of a result is then ``sum(c_i**2)`` plus, for sources that
share a current ``ParameterCovariance``, ``2 * c_i * c_j * rho_ij``.
Because contributions are merged by source, a source that occurs several
times (``a * a * a``, ``a + (1 - a)``, ``x * x - x**2``) is one variable
and cancels or adds up correctly.

A result is a snapshot: the contributions are those at the time it was
made. They are valid as long as the value and variance of the result are
untouched; setting either makes the result a source of its own.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Sequence

import scipp as sc

if TYPE_CHECKING:
    from .covariance import ParameterCovariance
    from .descriptor_number import DescriptorNumber

Contributions = dict[int, tuple['DescriptorNumber', float]]


class Derivation:
    """
    The contributions of the sources to a result, and the value and
    variance of the result they belong to.
    """

    __slots__ = ('contributions', 'stamp')

    def __init__(self, contributions: Contributions, stamp: tuple[float, float | None]) -> None:
        """
        Parameters
        ----------
        contributions : Contributions
            Source id mapped to the source and its signed contribution.
        stamp : tuple[float, float | None]
            Value and variance of the result when the contributions
            were made.
        """
        self.contributions = contributions
        self.stamp = stamp


def _stamp(obj: DescriptorNumber) -> tuple[float, float | None]:
    scalar = obj._scalar
    return scalar.value, scalar.variance


def contributions_of(obj: DescriptorNumber) -> Contributions:
    """
    The contributions of the sources behind ``obj`` to its uncertainty.

    Parameters
    ----------
    obj : DescriptorNumber
        A parameter or descriptor.

    Returns
    -------
    Contributions
        The contributions of the sources of a calculation if ``obj`` is
        an untouched result; otherwise ``obj`` itself with the
        contribution ``sigma``. Empty if there is no variance.
    """
    derivation = obj._derivation
    if derivation is not None and derivation.stamp == _stamp(obj):
        return derivation.contributions
    variance = obj._scalar.variance
    if variance is None:
        return {}
    return {id(obj): (obj, math.sqrt(variance))}


def variance_of(contributions: Contributions) -> float | None:
    """
    Variance from contributions.

    Parameters
    ----------
    contributions : Contributions
        Contributions as returned by ``contributions_of``.

    Returns
    -------
    float | None
        The variance, or None if there is no source with a variance.
    """
    if not contributions:
        return None
    total = sum(c * c for _, c in contributions.values())
    groups: dict[int, tuple[ParameterCovariance, list[tuple[DescriptorNumber, float]]]] = {}
    for source, c in contributions.values():
        record = getattr(source, 'covariance', None)
        if record is not None:
            groups.setdefault(id(record), (record, []))[1].append((source, c))
    for record, members in groups.values():
        for i, (first, c_first) in enumerate(members):
            for second, c_second in members[i + 1 :]:
                total += 2.0 * c_first * c_second * record.correlation_of(first, second)
    return max(total, 0.0)


def derive(
    result: DescriptorNumber, terms: Sequence[tuple[DescriptorNumber | Contributions, float]]
) -> DescriptorNumber:
    """
    Give a result the variance and the provenance of its calculation.

    Parameters
    ----------
    result : DescriptorNumber
        The new result. Its variance is replaced.
    terms : Sequence[tuple[DescriptorNumber | Contributions, float]]
        Each argument of the calculation, in the unit it had during the
        calculation, with the partial derivative of the result with
        respect to it, in the unit of the result per unit of the
        argument. An argument can be given as the ``Contributions``
        fetched while it was in that unit.

    Returns
    -------
    DescriptorNumber
        ``result``.
    """
    factor = _source_factor(result)
    merged = merge([(argument, partial * factor) for argument, partial in terms])
    variance = variance_of(merged)
    if variance is not None:
        result._scalar.variance = variance
    record(result, merged, scaled=True)
    return result


def _source_factor(result: DescriptorNumber) -> float:
    """
    The factor between the unit a result was calculated in and the unit
    it was stored in (they differ for a prefixed unit such as
    ``0.1 m^2``), and forget that unit.
    """
    source_unit, result._source_unit = result._source_unit, None
    if source_unit is None or source_unit == result._scalar.unit:
        return 1.0
    return float(sc.scalar(1.0, unit=source_unit).to(unit=result._scalar.unit).value)


def merge(
    terms: Sequence[tuple[DescriptorNumber | Contributions, float]],
) -> Contributions:
    """
    The contributions of the sources behind the arguments of a
    calculation, weighted by the partial derivatives.

    Parameters
    ----------
    terms : Sequence[tuple[DescriptorNumber | Contributions, float]]
        As for ``derive``.

    Returns
    -------
    Contributions
        The contributions to the result, merged by source.
    """
    merged: Contributions = {}
    for argument, partial in terms:
        fetched = argument if isinstance(argument, dict) else contributions_of(argument)
        for key, (source, c) in fetched.items():
            if key in merged:
                merged[key] = (source, merged[key][1] + partial * c)
            else:
                merged[key] = (source, partial * c)
    return merged


def record(result: DescriptorNumber, merged: Contributions, scaled: bool = False) -> None:
    """
    Remember the contributions of a result whose variance is already
    set.

    Parameters
    ----------
    result : DescriptorNumber
        The new result.
    merged : Contributions
        Its contributions, from ``merge``.
    scaled : bool, default=False
        Whether ``merged`` is already in the unit of the result. If not,
        it is converted from the unit the result was calculated in. By
        default, False.
    """
    if not scaled:
        factor = _source_factor(result)
        if factor != 1.0:
            merged = {key: (source, c * factor) for key, (source, c) in merged.items()}
    result._derivation = Derivation(merged, _stamp(result))


def adopt(target: DescriptorNumber, source: DescriptorNumber) -> None:
    """
    Make ``target``, which has just taken the value and variance of
    ``source``, a result of the same calculation.

    Parameters
    ----------
    target : DescriptorNumber
        The object that copied the value and variance of ``source``.
    source : DescriptorNumber
        A result of a calculation, or any parameter.
    """
    target._derivation = Derivation(dict(contributions_of(source)), _stamp(target))


def rescale(obj: DescriptorNumber, old_scalar: sc.Variable, new_scalar: sc.Variable) -> None:
    """
    Follow a unit conversion of ``obj``, whose contributions scale with
    its value.

    Parameters
    ----------
    obj : DescriptorNumber
        The converted object.
    old_scalar : sc.Variable
        Its scalar before the conversion.
    new_scalar : sc.Variable
        Its scalar after the conversion.
    """
    derivation = obj._derivation
    if derivation is None:
        return
    if derivation.stamp != (old_scalar.value, old_scalar.variance):
        obj._derivation = None
        return
    factor = float(sc.scalar(1.0, unit=old_scalar.unit).to(unit=new_scalar.unit).value)
    scaled = {key: (source, c * factor) for key, (source, c) in derivation.contributions.items()}
    obj._derivation = Derivation(scaled, (new_scalar.value, new_scalar.variance))
