import numbers
import warnings

import numpy as np
import scipp as sc

# from scipp import UnitError
# from scipp import Variable
# from scipp.typing import VariableLike
# from scipy.stats import norm
from easyscience.variable.descriptor_array import DescriptorArray
from easyscience.variable.descriptor_number import DescriptorNumber
from easyscience.variable.parameter import Parameter


def _validate_parameter(argument):
    """Enforces type and finite bound requirements."""
    if not isinstance(argument, Parameter):
        raise TypeError(f'Argument must be a `Parameter`. Found: {type(argument)}')

    if np.isinf(argument.min) or np.isinf(argument.max):
        raise ValueError(
            f'Argument bounds cannot be infinite. Found min: {argument.min}, max: {argument.max}'
        )


def _derive_grid_bounds(argument, np_func, grid_size=100):
    """Derives bounds for non-monotonic functions (sin, cos) using interval evaluation."""
    grid = np.linspace(argument.min, argument.max, grid_size)
    mapped_grid = np_func(grid)
    return float(np.min(mapped_grid)), float(np.max(mapped_grid))


def _derive_tan_bounds(argument, np_func, num_tolerance=1e-7, sigma_threshold=3.0):
    """
    Derives bounds for tangent, enforcing both a numerical keep-out zone
    and a statistical keep-out zone around asymptotes.
    """
    unit_str = str(argument.unit) if argument.unit else 'rad'

    # Standardize to radians
    min_rad = argument.min if unit_str == 'rad' else np.deg2rad(argument.min)
    max_rad = argument.max if unit_str == 'rad' else np.deg2rad(argument.max)
    val_rad = argument.value if unit_str == 'rad' else np.deg2rad(argument.value)

    def _dist_to_asymptote(rad_val):
        dist = (rad_val - np.pi / 2) % np.pi
        return min(dist, np.pi - dist)

    # 1. Numerical check (prevents floating point explosions)
    if _dist_to_asymptote(val_rad) < num_tolerance:
        raise ValueError(
            f'Parameter value {argument.value} {unit_str} is too close to a tangent asymptote.'
        )

    # 2. Statistical check (prevents Delta Method breakdown)
    if argument.variance is not None and argument.variance > 0:
        std_dev = np.sqrt(argument.variance)
        std_rad = std_dev if unit_str == 'rad' else np.deg2rad(std_dev)

        # If the asymptote is within N standard deviations of the value
        if _dist_to_asymptote(val_rad) < (sigma_threshold * std_rad):
            warnings.warn(
                f"The parameter '{argument.name}' is {argument.value}±{std_dev:.2f} {unit_str}. "
                f'This means the {sigma_threshold}-sigma confidence interval crosses a tangent asymptote. '
                'Linear error propagation (variance) reported by Scipp will be highly inaccurate.',
                UserWarning,
            )

    # 3. Bounds check (same as before)
    if _dist_to_asymptote(min_rad) < num_tolerance or _dist_to_asymptote(max_rad) < num_tolerance:
        raise ValueError('Tangent bounds touch or encroach on an asymptote.')

    branch_min = np.floor((min_rad - np.pi / 2) / np.pi)
    branch_max = np.floor((max_rad - np.pi / 2) / np.pi)

    if branch_min != branch_max:
        raise ValueError('Tangent bounds cross an asymptote. Fitting is unstable.')

    return float(np_func(min_rad)), float(np_func(max_rad))


def _apply_math_operation(argument, sc_func, np_func, operation_name):
    """
    Universal wrapper that routes bounds logic internally
    and handles Scipp execution.
    """
    _validate_parameter(argument)

    # Route to the correct bounds derivation logic
    if operation_name == 'tan':
        new_min, new_max = _derive_tan_bounds(argument, np_func)
    else:
        new_min, new_max = _derive_grid_bounds(argument, np_func)

    # Pass values to Scipp for strict execution and error propagation
    scipp_object = sc_func(
        sc.scalar(value=argument.value, variance=argument.variance, unit=argument.unit)
    )

    return Parameter.from_scipp(
        name=f'{operation_name}({argument.name})',
        full_value=scipp_object,
        min=new_min,
        max=new_max,
    )


def sin(argument: Parameter | numbers.Number):
    """EasyScience sine wrapper."""
    return _apply_math_operation(argument, sc.sin, np.sin, 'sin')


def cos(argument: Parameter | numbers.Number):
    """EasyScience cosine wrapper."""
    return _apply_math_operation(argument, sc.cos, np.cos, 'cos')


def tan(argument: Parameter | numbers.Number):
    """EasyScience tangent wrapper."""

    # TODO(): check these cases:
    # 1. if `value` is very close to pi/2 -> tan goes to inf.
    # 2. if `value` is close to pi/2 AND `value +/- sigma` crosses
    #    the point pi/2 -> tan AND propegated sigma go to inf.
    # 3. if `value` AND either bounds are very close to pi/2
    #    -> tan AND propegated bounds go to inf.
    # 4. if the `variance` places `value` closer to pi/2 than, e.g.,
    #    a 3-sigma CI  -> either bounds might go to inf.

    return _apply_math_operation(argument, sc.tan, np.tan, 'tan')
