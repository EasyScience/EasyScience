# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Mathematical functions for EasyScience ``Parameter`` objects.

Every public function takes one or more ``Parameter`` objects and returns a new
``Parameter`` holding

* the **value**, evaluated by Scipp (which also validates the units),
* the propagated **variance** (first-order "delta method", uncorrelated inputs), and
* the **fitting bounds**, i.e. the exact range the function can take when its
  argument stays inside the argument's own bounds.

Conventions:

* *Values are validated strictly.* A value outside the function's domain, or a
  non-finite result, raises ``ValueError``; such values would break a fit loop.
* *Bounds are permissive.* They never raise: bounds outside the domain are
  clipped to it, infinite bounds are accepted, and ``tan`` bounds that cross an
  asymptote become ``(-inf, inf)``.

Implementation: each function is described by a small, immutable ``_MathOp``
specification (which Scipp function to call, which domain it accepts, how to
derive its bounds, ...). A single dispatcher, ``_apply``, does the work for all of
them, so adding a function means adding a specification, not new control flow.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import scipp as sc

from easyscience.variable.parameter import Parameter

# A closed interval ``(lo, hi)`` of an argument, in the unit the bounds strategy works in.
_Interval = tuple[float, float]
# A bounds strategy maps one interval per argument to the ``(min, max)`` of the result.
_BoundsStrategy = Callable[..., tuple[float, float]]

_HALF_PI = np.pi / 2
_TWO_PI = 2 * np.pi

# tan: a value closer than this (in rad) to an asymptote is rejected whatever its variance;
# a larger standard deviation widens the margin.
_TAN_ASYMPTOTE_TOLERANCE = 1e-7

# sinc: the extrema are the roots of tan(x) = x. Two consecutive roots hold the largest
# maximum and the deepest minimum of an interval and a root just below ``lo`` may be
# filtered out, so four roots (one more than the three strictly needed) are refined.
_SINC_ROOT_COUNT = 4
# Newton's method converges quadratically from the asymptotic first guess; 8 steps is far
# more than the ~4 needed to reach machine precision.
_SINC_NEWTON_STEPS = 8
# Beyond this the lobes of sinc are below 1e-15 (they fall off like 1/x), and the lobe index
# would no longer fit numpy's integer range; the endpoints give the bounds to that precision.
_SINC_NEGLIGIBLE_LOBES_FROM = 1e15


# ==========================================
# Public functions: standard maths
# ==========================================


def exp(x: Parameter) -> Parameter:
    """
    Calculate the exponential, ``e**x``, of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``exp(x)`` with propagated variance and bounds ``(exp(x.min), exp(x.max))``.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the result overflows (``x`` greater than about 709.78) or, when ``x``
        has a variance, if the propagated variance overflows. The variance scales
        with ``exp(2x)``, so that happens from about 355 for a unit variance.

    Examples
    --------

    Exponential of a dimensionless parameter, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import exp
    >>> x = Parameter('x', 1.0, min=0.0, max=2.0)
    >>> result = exp(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    2.7183 in [1.0000, 7.3891]

    The variance is propagated (first order):

    >>> x = Parameter('x', 0.0, variance=0.01)
    >>> result = exp(x)
    >>> print(f'{result.value:.4f}, variance {result.variance:.4f}')
    1.0000, variance 0.0100
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_EXP, x)


def log(x: Parameter) -> Parameter:
    """
    Calculate the natural logarithm of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input. Its value must be strictly positive.

    Returns
    -------
    Parameter
        ``log(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the value of ``x`` is not strictly positive.

    Notes
    -----
    Bounds below zero are clipped to the domain, so ``x`` in ``[-1, 5]``
    gives result bounds ``[-inf, log(5)]``.

    Examples
    --------

    Natural logarithm, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import log
    >>> x = Parameter('x', 10.0, min=1.0, max=100.0)
    >>> result = log(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    2.3026 in [0.0000, 4.6052]

    Bounds below zero are clipped to the domain:

    >>> x = Parameter('x', 2.0, min=-1.0, max=5.0)
    >>> result = log(x)
    >>> print(f'bounds [{result.min:.4f}, {result.max:.4f}]')
    bounds [-inf, 1.6094]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_LOG, x)


def log10(x: Parameter) -> Parameter:
    """
    Calculate the base-10 logarithm of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input. Its value must be strictly positive.

    Returns
    -------
    Parameter
        ``log10(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the value of ``x`` is not strictly positive.

    Notes
    -----
    Bounds below zero are clipped to the domain, so ``x`` in ``[0, 100]``
    gives result bounds ``[-inf, 2]``.

    Examples
    --------

    Base-10 logarithm, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import log10
    >>> x = Parameter('x', 100.0, min=10.0, max=1000.0)
    >>> result = log10(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    2.0000 in [1.0000, 3.0000]

    Bounds touching zero are clipped to the domain:

    >>> x = Parameter('x', 10.0, min=0.0, max=100.0)
    >>> result = log10(x)
    >>> print(f'bounds [{result.min:.4f}, {result.max:.4f}]')
    bounds [-inf, 2.0000]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_LOG10, x)


def sqrt(x: Parameter) -> Parameter:
    """
    Calculate the square root of a parameter.

    Parameters
    ----------
    x : Parameter
        Input that is dimensionless, or whose unit is a perfect square (e.g. ``m**2``).
        Its value must not be negative.

    Returns
    -------
    Parameter
        ``sqrt(x)`` with propagated variance and bounds. The unit is the square root
        of the input unit (``m**2`` gives ``m``).

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If the unit of ``x`` has no square root in Scipp (e.g. ``m``).
    ValueError
        If the value of ``x`` is negative, or if the result is not finite (for example a
        non-zero variance at ``x == 0``, where the derivative is infinite).

    Notes
    -----
    Bounds below zero are clipped to the domain.
    EasyScience ``Parameter`` objects also support ``x ** 0.5``.

    Examples
    --------

    Square root of a dimensionless parameter, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import sqrt
    >>> x = Parameter('x', 4.0, min=1.0, max=9.0)
    >>> result = sqrt(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    2.0000 in [1.0000, 3.0000]

    The unit is square-rooted too:

    >>> area = Parameter('area', 9.0, unit='m**2')
    >>> result = sqrt(area)
    >>> print(result.value, result.unit)
    3.0 m
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_SQRT, x)


def erf(x: Parameter) -> Parameter:
    """
    Calculate the error function of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``erf(x)`` with propagated variance and bounds in ``[-1, 1]``.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.

    Notes
    -----
    Scipp does not propagate variances through ``erf``. EasyScience does it with
    the delta method, using ``d erf / dx = 2 / sqrt(pi) * exp(-x**2)``.

    Examples
    --------

    Error function, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import erf
    >>> x = Parameter('x', 1.0, min=0.0, max=2.0)
    >>> result = erf(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.8427 in [0.0000, 0.9953]

    The variance is propagated, although Scipp cannot do it for erf:

    >>> x = Parameter('x', 0.0, variance=0.01)
    >>> result = erf(x)
    >>> print(f'{result.value:.4f}, variance {result.variance:.4f}')
    0.0000, variance 0.0127
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_ERF, x)


def erfc(x: Parameter) -> Parameter:
    """
    Calculate the complementary error function, ``1 - erf(x)``, of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``erfc(x)`` with propagated variance and bounds in ``[0, 2]``.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.

    Notes
    -----
    Scipp does not propagate variances through ``erfc``. EasyScience does it with
    the delta method, using ``d erfc / dx = -2 / sqrt(pi) * exp(-x**2)``.
    The function is decreasing, so the bounds are swapped relative to the input.

    Examples
    --------

    Complementary error function; it is decreasing, so the bounds swap:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import erfc
    >>> x = Parameter('x', 1.0, min=0.0, max=2.0)
    >>> result = erfc(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.1573 in [0.0047, 1.0000]

    The variance is propagated, although Scipp cannot do it for erfc:

    >>> x = Parameter('x', 0.0, variance=0.01)
    >>> result = erfc(x)
    >>> print(f'{result.value:.4f}, variance {result.variance:.4f}')
    1.0000, variance 0.0127
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_ERFC, x)


# ==========================================
# Public functions: trigonometric
# ==========================================


def sin(x: Parameter) -> Parameter:
    """
    Calculate the trigonometric sine of an angle parameter.

    Parameters
    ----------
    x : Parameter
        Angle, with unit ``rad`` or ``deg``.

    Returns
    -------
    Parameter
        Dimensionless ``sin(x)`` with propagated variance and exact bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If the unit of ``x`` is not an angle.

    Notes
    -----
    The bounds are exact: the endpoints and the interior maxima and minima
    (``pi/2 + n*pi``) are evaluated, in radians whatever the input unit. Infinite
    bounds, or bounds wider than one period, give ``[-1, 1]``.

    Examples
    --------

    Angle in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import sin
    >>> import numpy as np
    >>> angle = Parameter('angle', np.pi / 2, unit='rad', min=0.0, max=np.pi)
    >>> result = sin(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    1.0000 in [0.0000, 1.0000]

    Angle in degrees; the bounds are computed in radians internally:

    >>> angle = Parameter('angle', 30.0, unit='deg', min=0.0, max=90.0)
    >>> result = sin(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.5000 in [0.0000, 1.0000]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_SIN, x)


def cos(x: Parameter) -> Parameter:
    """
    Calculate the trigonometric cosine of an angle parameter.

    Parameters
    ----------
    x : Parameter
        Angle, with unit ``rad`` or ``deg``.

    Returns
    -------
    Parameter
        Dimensionless ``cos(x)`` with propagated variance and exact bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If the unit of ``x`` is not an angle.

    Notes
    -----
    The bounds are exact: the endpoints and the interior maxima and minima
    (``n*pi``) are evaluated, in radians whatever the input unit. Infinite
    bounds, or bounds wider than one period, give ``[-1, 1]``.

    Examples
    --------

    Angle in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import cos
    >>> import numpy as np
    >>> angle = Parameter(
    ...     'angle', np.pi / 3, unit='rad', min=0.0, max=np.pi / 2
    ... )
    >>> result = cos(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.5000 in [0.0000, 1.0000]

    Angle in degrees:

    >>> angle = Parameter('angle', 60.0, unit='deg', min=0.0, max=180.0)
    >>> result = cos(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.5000 in [-1.0000, 1.0000]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_COS, x)


def tan(x: Parameter) -> Parameter:
    """
    Calculate the trigonometric tangent of an angle parameter.

    Parameters
    ----------
    x : Parameter
        Angle, with unit ``rad`` or ``deg``.

    Returns
    -------
    Parameter
        Dimensionless ``tan(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If the unit of ``x`` is not an angle.
    ValueError
        If the value of ``x`` is within 1e-7 rad, or one standard deviation if that is
        larger, of an asymptote (``pi/2 + n*pi``), where the propagated variance
        would explode.

    Notes
    -----
    The *value* is guarded against asymptotes, but the *bounds* are allowed to cross
    them: if ``x.min`` and ``x.max`` lie on different branches of ``tan``, the result
    bounds are ``(-inf, inf)``.

    Examples
    --------

    Angle in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import tan
    >>> import numpy as np
    >>> angle = Parameter(
    ...     'angle', np.pi / 4, unit='rad', min=0.0, max=np.pi / 3
    ... )
    >>> result = tan(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    1.0000 in [0.0000, 1.7321]

    Angle in degrees; bounds that cross an asymptote (90 deg) become infinite:

    >>> angle = Parameter('angle', 45.0, unit='deg', min=0.0, max=120.0)
    >>> result = tan(angle)
    >>> print(f'bounds [{result.min:.4f}, {result.max:.4f}]')
    bounds [-inf, inf]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_TAN, x)


def arcsin(x: Parameter, *, unit: str = 'rad') -> Parameter:
    """
    Calculate the inverse sine of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input; its value must be in ``[-1, 1]``.
    unit : str, default='rad'
        Unit of the returned angle, ``'rad'`` or ``'deg'``. The value, the variance and
        the bounds are all converted.

    Returns
    -------
    Parameter
        Angle in ``unit`` (``[-pi/2, pi/2]`` rad), with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless, or ``unit`` is not an angle unit.
    ValueError
        If the value of ``x`` is outside ``[-1, 1]``, or if the result is not finite
        (a non-zero variance at ``x == +-1``, where the derivative is infinite).

    Notes
    -----
    Bounds outside ``[-1, 1]`` are clipped to the domain.

    Examples
    --------

    The result is an angle in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arcsin
    >>> x = Parameter('x', 0.5, min=0.0, max=1.0)
    >>> result = arcsin(x)
    >>> result.unit
    'rad'

    Ask for degrees with ``unit``; the value, variance and bounds follow:

    >>> x = Parameter('x', 0.5, min=0.0, max=1.0)
    >>> result = arcsin(x, unit='deg')
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    30.0000 deg in [0.0000, 90.0000]
    """  # noqa: DOC502 -- raised in _apply and _angle_in
    return _angle_in(_apply(_ARCSIN, x), unit)


def arccos(x: Parameter, *, unit: str = 'rad') -> Parameter:
    """
    Calculate the inverse cosine of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input; its value must be in ``[-1, 1]``.
    unit : str, default='rad'
        Unit of the returned angle, ``'rad'`` or ``'deg'``. The value, the variance and
        the bounds are all converted.

    Returns
    -------
    Parameter
        Angle in ``unit`` (``[0, pi]`` rad), with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless, or ``unit`` is not an angle unit.
    ValueError
        If the value of ``x`` is outside ``[-1, 1]``, or if the result is not finite
        (a non-zero variance at ``x == +-1``, where the derivative is infinite).

    Notes
    -----
    Bounds outside ``[-1, 1]`` are clipped to the domain. The function is decreasing,
    so ``x.min`` maps to the result's maximum.

    Examples
    --------

    The result is an angle in radians; the function is decreasing, so the bounds swap:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arccos
    >>> x = Parameter('x', 0.5, min=0.0, max=1.0)
    >>> result = arccos(x)
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    1.0472 rad in [0.0000, 1.5708]

    Ask for degrees with ``unit``; the value, variance and bounds follow:

    >>> x = Parameter('x', 0.5, min=0.0, max=1.0)
    >>> result = arccos(x, unit='deg')
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    60.0000 deg in [0.0000, 90.0000]
    """  # noqa: DOC502 -- raised in _apply and _angle_in
    return _angle_in(_apply(_ARCCOS, x), unit)


def arctan(x: Parameter, *, unit: str = 'rad') -> Parameter:
    """
    Calculate the inverse tangent of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.
    unit : str, default='rad'
        Unit of the returned angle, ``'rad'`` or ``'deg'``. The value, the variance and
        the bounds are all converted.

    Returns
    -------
    Parameter
        Angle in ``unit`` (``[-pi/2, pi/2]`` rad), with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless, or ``unit`` is not an angle unit.

    Examples
    --------

    The result is an angle in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arctan
    >>> x = Parameter('x', 1.0, min=0.0, max=10.0)
    >>> result = arctan(x)
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    0.7854 rad in [0.0000, 1.4711]

    Ask for degrees with ``unit``; the value, variance and bounds follow:

    >>> x = Parameter('x', 1.0, min=0.0, max=10.0)
    >>> result = arctan(x, unit='deg')
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    45.0000 deg in [0.0000, 84.2894]
    """  # noqa: DOC502 -- raised in _apply and _angle_in
    return _angle_in(_apply(_ARCTAN, x), unit)


def arctan2(y: Parameter, x: Parameter, *, unit: str = 'rad') -> Parameter:
    """
    Calculate the angle of the point ``(x, y)``, using both signs to find the quadrant.

    Parameters
    ----------
    y : Parameter
        Cartesian y-coordinate.
    x : Parameter
        Cartesian x-coordinate. Must have the same unit as ``y``.
    unit : str, default='rad'
        Unit of the returned angle, ``'rad'`` or ``'deg'``. The value, the variance and
        the bounds are all converted.

    Returns
    -------
    Parameter
        Angle in ``unit`` (``[-pi, pi]`` rad), with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``y`` or ``x`` is not a ``Parameter``.
    UnitError
        If ``y`` and ``x`` have different units, or ``unit`` is not an angle unit.
    ValueError
        If an argument has a variance and the point ``(x, y)`` lies within the larger of
        the two standard deviations of the origin, where the angle is undefined, or if
        the result is not finite.

    Warns
    -----
    UserWarning
        If all four bounds are finite and the box they span touches the branch cut.

    Notes
    -----
    Scipp does not propagate variances through ``atan2``. EasyScience does it with the
    delta method, assuming ``y`` and ``x`` are uncorrelated.

    The bounds are the exact range of the angle over the box ``[y.min, y.max] x
    [x.min, x.max]``. If the box touches the branch cut (``y = 0`` with ``x < 0``) the
    angle jumps between ``-pi`` and ``pi``; the bounds are then ``[-pi, pi]``, and a
    ``UserWarning`` is raised when all four bounds are finite, since gradient-based
    minimisers behave badly across the jump.

    Examples
    --------

    The angle of the point (x, y) = (1, 1) m, in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arctan2
    >>> y = Parameter('y', 1.0, unit='m', min=0.5, max=2.0)
    >>> x = Parameter('x', 1.0, unit='m', min=1.0, max=3.0)
    >>> result = arctan2(y, x)
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    0.7854 rad in [0.1651, 1.1071]

    Ask for degrees with ``unit``; the value, variance and bounds follow:

    >>> y = Parameter('y', 1.0, unit='m', min=0.5, max=2.0)
    >>> x = Parameter('x', 1.0, unit='m', min=1.0, max=3.0)
    >>> result = arctan2(y, x, unit='deg')
    >>> print(
    ...     f'{result.value:.4f} {result.unit} in [{result.min:.4f}, {result.max:.4f}]'
    ... )
    45.0000 deg in [9.4623, 63.4349]
    """  # noqa: DOC502 -- raised in _apply and _angle_in
    return _angle_in(_apply(_ARCTAN2, y, x), unit)


def sinc(x: Parameter) -> Parameter:
    """
    Calculate the unnormalised sinc function, ``sin(x) / x``, of a parameter.

    Parameters
    ----------
    x : Parameter
        Input, with unit ``rad`` or dimensionless.

    Returns
    -------
    Parameter
        Dimensionless ``sinc(x)`` (equal to 1 at ``x = 0``) with propagated variance
        and exact bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If the unit of ``x`` is neither ``rad`` nor dimensionless.

    Notes
    -----
    This matches Scipp and the physics convention, not the normalised
    ``sin(pi x) / (pi x)`` of ``numpy.sinc``.

    The bounds are exact. The extrema of sinc lie at ``x = 0`` and at the roots of
    ``tan(x) = x``; the roots inside the interval are found numerically.
    The global minimum, about -0.2172, is at ``x`` of about 4.4934.

    Examples
    --------

    Angle in radians:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import sinc
    >>> import numpy as np
    >>> angle = Parameter('angle', np.pi / 2, unit='rad', min=0.0, max=10.0)
    >>> result = sinc(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.6366 in [-0.2172, 1.0000]

    sinc does not accept degrees; convert an angle in degrees first:

    >>> angle = Parameter('angle', 90.0, unit='deg')
    >>> angle.convert_unit('rad')
    >>> result = sinc(angle)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.6366 in [-0.2172, 1.0000]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_SINC, x)


# ==========================================
# Public functions: hyperbolic
# ==========================================


def sinh(x: Parameter) -> Parameter:
    """
    Calculate the hyperbolic sine of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``sinh(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the result overflows (``|x|`` greater than about 710) or, when ``x``
        has a variance, if the propagated variance overflows (from ``|x|`` of
        about 355 for a unit variance).

    Examples
    --------

    Hyperbolic sine, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import sinh
    >>> x = Parameter('x', 1.0, min=0.0, max=2.0)
    >>> result = sinh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    1.1752 in [0.0000, 3.6269]

    The variance is propagated (first order):

    >>> x = Parameter('x', 0.0, variance=0.01)
    >>> result = sinh(x)
    >>> print(f'{result.value:.4f}, variance {result.variance:.4f}')
    0.0000, variance 0.0100
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_SINH, x)


def cosh(x: Parameter) -> Parameter:
    """
    Calculate the hyperbolic cosine of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``cosh(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the result overflows (``|x|`` greater than about 710) or, when ``x``
        has a variance, if the propagated variance overflows (from ``|x|`` of
        about 355 for a unit variance).

    Notes
    -----
    ``cosh`` is not monotonic: if the bounds contain 0, the lower result bound is 1.

    Examples
    --------

    Hyperbolic cosine; the minimum, 1, is inside the bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import cosh
    >>> x = Parameter('x', 0.5, min=-1.0, max=2.0)
    >>> result = cosh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    1.1276 in [1.0000, 3.7622]

    Bounds on one side of zero: cosh is monotonic there:

    >>> x = Parameter('x', 1.0, min=0.5, max=2.0)
    >>> result = cosh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    1.5431 in [1.1276, 3.7622]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_COSH, x)


def tanh(x: Parameter) -> Parameter:
    """
    Calculate the hyperbolic tangent of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``tanh(x)`` with propagated variance and bounds in ``[-1, 1]``.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.

    Examples
    --------

    Hyperbolic tangent, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import tanh
    >>> x = Parameter('x', 0.5, min=0.0, max=2.0)
    >>> result = tanh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.4621 in [0.0000, 0.9640]

    Without bounds the result is limited to [-1, 1]:

    >>> x = Parameter('x', 0.5)
    >>> result = tanh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.4621 in [-1.0000, 1.0000]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_TANH, x)


def arcsinh(x: Parameter) -> Parameter:
    """
    Calculate the inverse hyperbolic sine of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input.

    Returns
    -------
    Parameter
        ``arcsinh(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.

    Examples
    --------

    Inverse hyperbolic sine, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arcsinh
    >>> x = Parameter('x', 1.0, min=0.0, max=10.0)
    >>> result = arcsinh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.8814 in [0.0000, 2.9982]

    The variance is propagated (first order):

    >>> x = Parameter('x', 0.0, variance=0.01)
    >>> result = arcsinh(x)
    >>> print(f'{result.value:.4f}, variance {result.variance:.4f}')
    0.0000, variance 0.0100
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_ARCSINH, x)


def arccosh(x: Parameter) -> Parameter:
    """
    Calculate the inverse hyperbolic cosine of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input; its value must be at least 1.

    Returns
    -------
    Parameter
        ``arccosh(x)`` (non-negative) with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the value of ``x`` is below 1, or if the result is not finite (a non-zero
        variance at ``x == 1``, where the derivative is infinite).

    Notes
    -----
    Bounds below 1 are clipped to the domain.

    Examples
    --------

    Inverse hyperbolic cosine, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arccosh
    >>> x = Parameter('x', 2.0, min=1.0, max=10.0)
    >>> result = arccosh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    1.3170 in [0.0000, 2.9932]

    Bounds below 1 are clipped to the domain:

    >>> x = Parameter('x', 2.0, min=-5.0, max=10.0)
    >>> result = arccosh(x)
    >>> print(f'bounds [{result.min:.4f}, {result.max:.4f}]')
    bounds [0.0000, 2.9932]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_ARCCOSH, x)


def arctanh(x: Parameter) -> Parameter:
    """
    Calculate the inverse hyperbolic tangent of a dimensionless parameter.

    Parameters
    ----------
    x : Parameter
        Dimensionless input; its value must be strictly between -1 and 1.

    Returns
    -------
    Parameter
        ``arctanh(x)`` with propagated variance and bounds.

    Raises
    ------
    NotImplementedError
        If ``x`` is not a ``Parameter``.
    UnitError
        If ``x`` is not dimensionless.
    ValueError
        If the value of ``x`` is not strictly inside ``(-1, 1)``.

    Notes
    -----
    Bounds beyond +-1 are clipped to the domain, which makes the result bound infinite.

    Examples
    --------

    Inverse hyperbolic tangent, with its bounds:

    >>> from easyscience import Parameter
    >>> from easyscience.variable import arctanh
    >>> x = Parameter('x', 0.5, min=0.0, max=0.9)
    >>> result = arctanh(x)
    >>> print(f'{result.value:.4f} in [{result.min:.4f}, {result.max:.4f}]')
    0.5493 in [0.0000, 1.4722]

    Bounds beyond +-1 are clipped, which makes the result bounds infinite:

    >>> x = Parameter('x', 0.5, min=-2.0, max=2.0)
    >>> result = arctanh(x)
    >>> print(f'bounds [{result.min:.4f}, {result.max:.4f}]')
    bounds [-inf, inf]
    """  # noqa: DOC502 -- raised in _apply
    return _apply(_ARCTANH, x)


# ==========================================
# Aliases
# ==========================================
# The C / Scipp spellings of the inverse functions. They are the same function objects, so
# they share the docstring, and the parameters they return are named after the canonical
# function (``asin(x)`` returns a parameter called ``arcsin(x)``).

asin = arcsin
acos = arccos
atan = arctan
atan2 = arctan2
asinh = arcsinh
acosh = arccosh
atanh = arctanh


# ==========================================
# Specification of a maths operation
# ==========================================


@dataclass(frozen=True)
class _Domain:
    """The set of argument values a function accepts: an interval, open or closed at each end."""

    lo: float = -np.inf
    hi: float = np.inf
    open_lo: bool = False
    open_hi: bool = False

    def contains(self, value: float) -> bool:
        above = value > self.lo if self.open_lo else value >= self.lo
        below = value < self.hi if self.open_hi else value <= self.hi
        return above and below

    def clip(self, lo: float, hi: float) -> _Interval:
        """Restrict ``[lo, hi]`` to the domain (open ends clip to the edge: the limit)."""
        return max(lo, self.lo), min(hi, self.hi)

    def __str__(self) -> str:
        left = '(' if self.open_lo else '['
        right = ')' if self.open_hi else ']'
        return f'{left}{self.lo}, {self.hi}{right}'


@dataclass(frozen=True)
class _MathOp:
    """
    Everything that makes one maths function different from another.

    Attributes
    ----------
    name : str
        Name used in error messages and in the name of the resulting ``Parameter``.
    sc_func : Callable
        The Scipp function (it validates units and propagates variance, if it can).
    bounds : _BoundsStrategy
        Maps one ``(lo, hi)`` interval per argument to the ``(min, max)`` of the result.
    domain : _Domain, default=_Domain()
        Accepted argument values. Applies to every argument.
    native_unit : str | None, default=None
        If set, bounds are converted to this unit before ``bounds`` runs
        (``'rad'`` for the trigonometric functions, whose numpy versions assume radians).
    jacobian : Callable | None, default=None
        Partial derivatives of the function, ``jacobian(*values) -> (df/dx1, df/dx2, ...)``.
        Set it for functions whose variance Scipp cannot propagate; EasyScience then
        applies the delta method itself.
    guard : Callable[..., None] | None, default=None
        Extra check of the arguments' values, ``guard(*params)``, raising ``ValueError``
        where the delta method breaks down (tan asymptotes, the arctan2 origin).
    """

    name: str
    sc_func: Callable
    bounds: _BoundsStrategy
    domain: _Domain = _Domain()
    native_unit: str | None = None
    jacobian: Callable | None = None
    guard: Callable[..., None] | None = None


# ==========================================
# The dispatcher
# ==========================================


def _apply(op: _MathOp, *params: Parameter) -> Parameter:
    """
    Evaluate ``op`` on ``params``: validate, call Scipp, derive bounds, repack.

    Scipp validates the units, so a wrong unit raises its ``UnitError`` before any of
    the checks below.

    Parameters
    ----------
    op : _MathOp
        Specification of the function.
    *params : Parameter
        The arguments.

    Returns
    -------
    Parameter
        A new ``Parameter`` named ``'<op>(<arg names>)'``.

    Raises
    ------
    NotImplementedError
        If an argument is not a ``Parameter``.
    ValueError
        If an argument's value is outside the domain, or the result is not finite.
    """
    for param in params:
        if not isinstance(param, Parameter):
            raise NotImplementedError(
                f'EasyScience math functions currently only support `Parameter` objects. '
                f'Raw numbers and arrays are not yet supported. Found: {type(param)}'
            )

    # 1. Scipp first: it owns the unit rules, so a unit error is reported before anything else.
    manual_variance = op.jacobian is not None
    sc_args = [_to_scipp_scalar(param, strip_variance=manual_variance) for param in params]
    sc_result = op.sc_func(*sc_args)

    # 2. Values are validated strictly: a bad value would break a fit loop.
    for param in params:
        if not op.domain.contains(param.value):
            raise ValueError(
                f"The value of '{param.name}' ({param.value}) is outside the domain "
                f'{op.domain} of {op.name}().'
            )
    if op.guard is not None:
        op.guard(*params)

    # 3. Variance for functions Scipp cannot handle: the delta method.
    if manual_variance:
        variance = _delta_method_variance(op, params)
        if variance is not None:
            sc_result.variance = variance

    label = ', '.join(param.name for param in params)
    if not _is_finite(sc_result.value) or not _is_finite(sc_result.variance):
        raise ValueError(
            f'{op.name}({label}) is not finite: value={sc_result.value}, '
            f'variance={sc_result.variance}. Check the value and variance of the argument(s).'
        )

    # 4. Bounds are permissive: clip to the domain, convert, and let the strategy decide.
    intervals = [_native_interval(op, param) for param in params]
    with np.errstate(all='ignore'):
        new_min, new_max = op.bounds(*intervals)
    # Guard against round-off at an extremum: the value is always inside its own bounds.
    new_min, new_max = min(new_min, sc_result.value), max(new_max, sc_result.value)

    return Parameter.from_scipp(
        name=f'{op.name}({label})',
        full_value=sc_result,
        min=new_min,
        max=new_max,
    )


def _angle_in(result: Parameter, unit: str) -> Parameter:
    """
    Express an angle ``result`` (in radians, as Scipp returns it) in ``unit``.

    The value, the variance and the bounds are converted together by
    ``Parameter.convert_unit``.

    Parameters
    ----------
    result : Parameter
        An angle in radians. It is converted in place.
    unit : str
        Target unit, ``'rad'`` or ``'deg'``.

    Returns
    -------
    Parameter
        ``result``, in ``unit``.

    Raises
    ------
    UnitError
        If ``unit`` is not an angle unit.
    """  # noqa: DOC502 -- raised by Parameter.convert_unit
    if unit != result.unit:
        result.convert_unit(unit)
    return result


def _variance(param: Parameter) -> float:
    """Variance of a parameter; a missing (``None``) or zero variance counts as 0.0."""
    return float(param.variance) if param.variance else 0.0


def _to_scipp_scalar(param: Parameter, strip_variance: bool = False) -> sc.Variable:
    """Convert a Parameter to a Scipp scalar; a zero or missing variance is dropped."""
    kwargs = {'value': float(param.value), 'unit': param.unit}
    variance = _variance(param)
    if not strip_variance and variance > 0:
        kwargs['variance'] = variance
    return sc.scalar(**kwargs)


def _delta_method_variance(op: _MathOp, params: tuple[Parameter, ...]) -> float | None:
    """
    First-order error propagation, ``var(y) = sum_i (dy/dx_i)**2 * var(x_i)``.

    Only arguments that have a variance contribute (this avoids ``0 * inf``).
    Returns ``None`` if no argument has a variance.
    """
    variances = [_variance(p) for p in params]
    if not any(variances):
        return None
    with np.errstate(all='ignore'):
        partials = op.jacobian(*(np.float64(p.value) for p in params))
        return float(sum(j**2 * v for j, v in zip(partials, variances, strict=True) if v > 0))


def _is_finite(value: float | None) -> bool:
    """Whether ``value`` is finite; a missing value (no variance) counts as fine."""
    return value is None or bool(np.isfinite(value))


@cache
def _native_factor(unit: str, native_unit: str | None) -> float:
    """Factor converting ``unit`` to ``native_unit`` (1.0 if none is needed), asked of Scipp once."""
    if native_unit is None or unit == native_unit:
        return 1.0
    return float(sc.scalar(1.0, unit=unit).to(unit=native_unit).value)


def _native_interval(op: _MathOp, param: Parameter) -> _Interval:
    """The parameter's bounds, clipped to the domain and in the unit the strategy expects."""
    lo, hi = op.domain.clip(float(param.min), float(param.max))
    factor = _native_factor(str(param.unit), op.native_unit)
    return lo * factor, hi * factor


# ==========================================
# Bounds strategies
# ==========================================
# Each takes one (lo, hi) interval per argument and returns the exact (min, max) of the
# function over that interval. They run inside ``np.errstate(all='ignore')``, so an edge
# such as log(0) = -inf is returned instead of warning.


def _monotonic(func: Callable) -> _BoundsStrategy:
    """Monotonic function (increasing or decreasing): the extremes sit at the endpoints."""

    def bounds(interval: _Interval) -> tuple[float, float]:
        lo, hi = interval
        at_lo, at_hi = float(func(lo)), float(func(hi))
        return min(at_lo, at_hi), max(at_lo, at_hi)

    return bounds


def _extrema(
    func: Callable, critical_points: Callable[[float, float], np.ndarray]
) -> _BoundsStrategy:
    """
    Non-monotonic function: the extremes sit at the endpoints or at critical points.

    ``critical_points(lo, hi)`` returns the points inside ``[lo, hi]`` where the function
    may have a local extremum. ``nan`` candidates, such as ``sin(inf)``, are ignored.
    """

    def bounds(interval: _Interval) -> tuple[float, float]:
        lo, hi = interval
        candidates = np.concatenate(([lo, hi], critical_points(lo, hi)))
        values = np.asarray(func(candidates), dtype=float)
        return float(np.nanmin(values)), float(np.nanmax(values))

    return bounds


def _periodic_critical_points(offset: float) -> Callable[[float, float], np.ndarray]:
    """
    Critical points ``offset + n*pi`` (n an integer) of sin (``pi/2``) or cos (``0``).

    A whole period, or an infinite interval, holds a maximum and a minimum, so those two
    are returned without enumerating.
    """

    def points(lo: float, hi: float) -> np.ndarray:
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo >= _TWO_PI:
            return np.array([offset, offset + np.pi])
        n = np.arange(np.ceil((lo - offset) / np.pi), np.floor((hi - offset) / np.pi) + 1)
        return offset + n * np.pi

    return points


def _cosh_critical_points(lo: float, hi: float) -> np.ndarray:
    """The single minimum of cosh, at 0."""
    return np.array([0.0]) if lo <= 0.0 <= hi else np.array([])


def _unnormalized_sinc(x) -> np.ndarray:
    """
    sin(x)/x, with sinc(0) = 1 and the limit sinc(+-inf) = 0.

    ``np.sinc`` is the *normalised* sin(pi x)/(pi x), so the argument is scaled by 1/pi.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(all='ignore'):
        return np.where(np.isinf(x), 0.0, np.sinc(x / np.pi))


def _sinc_positive_extrema(lo: float, hi: float, count: int = _SINC_ROOT_COUNT) -> np.ndarray:
    """
    Stationary points of sinc on the positive axis inside ``[lo, hi]``, nearest to ``lo``.

    They are the positive roots of ``tan(x) = x``, one in each ``(k*pi, k*pi + pi/2)``.
    The heights of the lobes fall off like 1/x, so the largest maximum and the deepest
    minimum in the interval are among the first few roots at or after ``lo``.

    Beyond ``_SINC_NEGLIGIBLE_LOBES_FROM`` the lobes are negligible and none are returned.

    All ``count`` roots are refined at once with Newton's method on
    ``g(x) = x cos(x) - sin(x)``, ``g'(x) = -x sin(x)``, with the vectorised update
    ``x <- x - g(x) / g'(x)`` (no Python loop over the roots).
    """
    if hi <= 0.0 or lo > _SINC_NEGLIGIBLE_LOBES_FROM:
        return np.array([])
    first = max(1, int(np.floor(max(lo, 0.0) / np.pi)))
    k = np.arange(first, first + count)
    x = (k + 0.5) * np.pi
    x = x - 1.0 / x  # asymptotic first guess for the root of tan(x) = x
    for _ in range(_SINC_NEWTON_STEPS):
        x = x - (x * np.cos(x) - np.sin(x)) / (-x * np.sin(x))
    return x[(x >= lo) & (x <= hi)]


def _sinc_critical_points(lo: float, hi: float) -> np.ndarray:
    """Stationary points of sinc: 0 (the global maximum) and the roots of tan(x) = x on both sides."""
    points = [_sinc_positive_extrema(lo, hi), -_sinc_positive_extrema(-hi, -lo)]
    if lo <= 0.0 <= hi:
        points.append(np.array([0.0]))
    return np.concatenate(points)


def _tan_bounds(interval: _Interval) -> tuple[float, float]:
    """
    Bounds of tan(x) for ``x`` in radians. tan is increasing between asymptotes
    (``pi/2 + n*pi``); an interval that spans an asymptote covers all real numbers.
    """
    lo, hi = interval
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return -np.inf, np.inf
    branch_lo = np.floor((lo - _HALF_PI) / np.pi)
    branch_hi = np.floor((hi - _HALF_PI) / np.pi)
    if branch_lo != branch_hi:
        return -np.inf, np.inf
    return float(np.tan(lo)), float(np.tan(hi))


def _std(param: Parameter) -> float:
    """Standard deviation of a parameter; 0 if it has no variance."""
    return math.sqrt(_variance(param))


def _tan_guard(param: Parameter) -> None:
    """
    The value must keep clear of a tan asymptote, where the variance would explode.

    The delta method is only meaningful if the asymptote lies more than one standard
    deviation away, so that distance (never less than ``_TAN_ASYMPTOTE_TOLERANCE``) is
    the limit.
    """
    factor = _native_factor(str(param.unit), 'rad')
    value = float(param.value) * factor
    margin = max(_TAN_ASYMPTOTE_TOLERANCE, _std(param) * factor)
    distance = abs(value % np.pi - _HALF_PI)
    if distance <= margin:
        raise ValueError(
            f"The value of '{param.name}' ({param.value} {param.unit}) is within "
            f'{margin:.3g} rad of a tan asymptote (pi/2 + n*pi); the propagated variance '
            f'would explode.'
        )


def _arctan2_guard(y: Parameter, x: Parameter) -> None:
    """
    The point ``(x, y)`` must keep clear of the origin, where the angle is undefined.

    Only checked if an argument has a variance: the delta method then divides by
    ``x**2 + y**2``. The point must lie more than one standard deviation from the origin.
    """
    margin = max(_std(y), _std(x))
    if margin > 0 and np.hypot(float(y.value), float(x.value)) <= margin:
        raise ValueError(
            f"The point ('{x.name}', '{y.name}') = ({x.value}, {y.value}) is within one standard "
            f'deviation ({margin:.3g}) of the origin, where the angle of arctan2() is undefined '
            f'and its propagated variance would explode.'
        )


def _arctan2_bounds(y_interval: _Interval, x_interval: _Interval) -> tuple[float, float]:
    """
    Bounds of arctan2(y, x) over the box ``y_interval x x_interval``.

    The angle is a harmonic function, so away from the branch cut its extremes sit on the
    corners of the box. If the box touches the cut (``y = 0`` with ``x < 0``) the angle
    jumps between -pi and pi and the range is the full ``[-pi, pi]``.
    """
    (y_lo, y_hi), (x_lo, x_hi) = y_interval, x_interval
    if y_lo < 0.0 <= y_hi and x_lo < 0.0:
        if np.isfinite([y_lo, y_hi, x_lo, x_hi]).all():
            # stacklevel=4: this function -> _apply -> arctan2 -> the user's call.
            warnings.warn(
                'The bounds of the arguments touch the arctan2 branch cut (y = 0, x < 0), where '
                'the angle jumps between -pi and pi. Gradient-based minimisers behave badly '
                'across such a jump.',
                UserWarning,
                stacklevel=4,
            )
        return -np.pi, np.pi
    corners = [np.arctan2(y, x) for y in (y_lo, y_hi) for x in (x_lo, x_hi)]
    return float(min(corners)), float(max(corners))


# ==========================================
# Jacobians (for variance Scipp cannot propagate)
# ==========================================


# NumPy scalars (not Python floats) are passed in: 0/0 then gives nan under ``np.errstate``
# and is caught by the finiteness check, instead of raising ``ZeroDivisionError``.


def _arctan2_jacobian(y: np.float64, x: np.float64) -> tuple[np.float64, np.float64]:
    r2 = x**2 + y**2
    return x / r2, -y / r2


def _erf_jacobian(x: np.float64) -> tuple[np.float64]:
    return (2.0 / np.sqrt(np.pi) * np.exp(-(x**2)),)


def _erfc_jacobian(x: np.float64) -> tuple[np.float64]:
    return (-2.0 / np.sqrt(np.pi) * np.exp(-(x**2)),)


# ==========================================
# The specifications
# ==========================================

_POSITIVE = _Domain(lo=0.0, open_lo=True)
_NON_NEGATIVE = _Domain(lo=0.0)
_AT_LEAST_ONE = _Domain(lo=1.0)
_CLOSED_UNIT_INTERVAL = _Domain(lo=-1.0, hi=1.0)
_OPEN_UNIT_INTERVAL = _Domain(lo=-1.0, hi=1.0, open_lo=True, open_hi=True)

_EXP = _MathOp('exp', sc.exp, _monotonic(np.exp))
_LOG = _MathOp('log', sc.log, _monotonic(np.log), domain=_POSITIVE)
_LOG10 = _MathOp('log10', sc.log10, _monotonic(np.log10), domain=_POSITIVE)
_SQRT = _MathOp('sqrt', sc.sqrt, _monotonic(np.sqrt), domain=_NON_NEGATIVE)
_ERF = _MathOp('erf', sc.erf, _monotonic(math.erf), jacobian=_erf_jacobian)
_ERFC = _MathOp('erfc', sc.erfc, _monotonic(math.erfc), jacobian=_erfc_jacobian)

_SIN = _MathOp(
    'sin', sc.sin, _extrema(np.sin, _periodic_critical_points(_HALF_PI)), native_unit='rad'
)
_COS = _MathOp('cos', sc.cos, _extrema(np.cos, _periodic_critical_points(0.0)), native_unit='rad')
_TAN = _MathOp('tan', sc.tan, _tan_bounds, native_unit='rad', guard=_tan_guard)
_ARCSIN = _MathOp('arcsin', sc.asin, _monotonic(np.arcsin), domain=_CLOSED_UNIT_INTERVAL)
_ARCCOS = _MathOp('arccos', sc.acos, _monotonic(np.arccos), domain=_CLOSED_UNIT_INTERVAL)
_ARCTAN = _MathOp('arctan', sc.atan, _monotonic(np.arctan))
_ARCTAN2 = _MathOp(
    'arctan2',
    lambda y, x: sc.atan2(y=y, x=x),  # Scipp's atan2 takes keyword-only arguments
    _arctan2_bounds,
    jacobian=_arctan2_jacobian,
    guard=_arctan2_guard,
)
_SINC = _MathOp('sinc', sc.sinc, _extrema(_unnormalized_sinc, _sinc_critical_points))

_SINH = _MathOp('sinh', sc.sinh, _monotonic(np.sinh))
_COSH = _MathOp('cosh', sc.cosh, _extrema(np.cosh, _cosh_critical_points))
_TANH = _MathOp('tanh', sc.tanh, _monotonic(np.tanh))
_ARCSINH = _MathOp('arcsinh', sc.asinh, _monotonic(np.arcsinh))
_ARCCOSH = _MathOp('arccosh', sc.acosh, _monotonic(np.arccosh), domain=_AT_LEAST_ONE)
_ARCTANH = _MathOp('arctanh', sc.atanh, _monotonic(np.arctanh), domain=_OPEN_UNIT_INTERVAL)
