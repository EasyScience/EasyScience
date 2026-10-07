import warnings
from collections.abc import Callable

import numpy as np
import scipp as sc

# from scipp import UnitError
# from scipp import Variable
# from scipp.typing import VariableLike
# from scipy.stats import norm
from easyscience.variable.descriptor_array import DescriptorArray
from easyscience.variable.descriptor_number import DescriptorNumber
from easyscience.variable.parameter import Parameter

# Define type aliases for readability if preferred, or use inline Callable
# np_func takes a NumPy array and returns a NumPy array
NumPyMathFunc = Callable[[np.ndarray], np.ndarray]
# sc_func takes a Scipp Variable and returns a Scipp Variable
ScippMathFunc = Callable[[sc.Variable], sc.Variable]


###############################################################################
## Standard Maths Operations
###############################################################################

# list: exp, log, log10, erf, erfc


###############################################################################
## Trigonometric Operations
###############################################################################


def sin(argument: Parameter) -> Parameter:
    """Calculates the trigonometric sine of the given parameter.

    This function safely wraps the Scipp sine execution to guarantee correct
    error propagation and unit conversions. It automatically calculates new
    fitting bounds using a grid-based evaluation to properly account for the
    non-monotonic nature of the sine function.

    Parameters
    ----------
    argument : Parameter
        The input parameter or numerical value to be evaluated. It should
        have angular units (e.g., 'rad', 'deg').

    Returns
    -------
    : Same type as input
        A new parameter object containing the calculated sine value,
        propagated variance, mapped bounds, and a dimensionless unit.

    Raises
    ------
    TypeError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the fitting bounds of the input parameter are infinite.

    Examples
    --------
    Compute sine of angles in radians:

      >>> import easyscience as es
      >>> import numpy as np
      >>> angle = es.Parameter(
      ...     name='angle',
      ...     value=np.pi / 4,
      ...     min=0.0,
      ...     max=1.0,
      ...     fixed=False,
      ...     unit='rad',
      ... )
      >>> es.sin(angle)
     <Parameter 'sin(angle)': 0.7071, bounds=[0.0:0.8414709848078965]>

    The input must have angle units (rad or deg):

      >>> angle_deg = es.Parameter(
      ...     name='angle_deg', value=45, min=0, max=90, fixed=False, unit='deg'
      ... )
      >>> es.sin(angle_deg)
      <Parameter 'sin(angle_deg)': 0.7071, bounds=[-0.999981805600717:0.9998362543788631]>
    """
    return _apply_math_operation(argument, sc.sin, np.sin, 'sin')


def cos(argument: Parameter) -> Parameter:
    """
    Calculates the trigonometric cosine of the given parameter.

    This function safely wraps the Scipp cosine execution to guarantee correct
    error propagation and unit conversions. It automatically calculates new
    fitting bounds using a grid-based evaluation to properly account for the
    non-monotonic nature of the cosine function.

    Parameters
    ----------
    argument : Parameter
        The input parameter or numerical value to be evaluated. It should
        have angular units (e.g., 'rad', 'deg').

    Returns
    -------
    Parameter
        A new parameter object containing the calculated cosine value,
        propagated variance, mapped bounds, and a dimensionless unit.

    Raises
    ------
    TypeError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the fitting bounds of the input parameter are infinite.
    """
    return _apply_math_operation(argument, sc.cos, np.cos, 'cos')


def tan(argument: Parameter) -> Parameter:
    """
    Calculates the trigonometric tangent of the given parameter.

    This function safely wraps the Scipp tangent execution to guarantee correct
    error propagation and unit conversions. It implements strict numerical
    and statistical guardrails around asymptotes (pi/2 + n*pi) to prevent
    optimization crashes and physically invalid variances during fitting.

    Parameters
    ----------
    argument : Parameter
        The input parameter or numerical value to be evaluated. It should
        have angular units (e.g., 'rad', 'deg').

    Returns
    -------
    Parameter
        A new parameter object containing the calculated tangent value,
        propagated variance, mapped bounds, and a dimensionless unit.

    Raises
    ------
    TypeError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the fitting bounds are infinite, if the value touches an asymptote,
        or if the bounds straddle an asymptote.

    Notes
    -----
    Because Scipp uses the Delta Method (first-order Taylor expansion) for
    error propagation, a ``UserWarning`` will be raised if the parameter's
    variance places its 3-sigma confidence interval across an asymptote. In
    this scenario, the linear error propagation reported is highly inaccurate.
    """

    # TODO(): check these cases:
    # 1. if `value` is very close to pi/2 -> tan goes to inf.
    # 2. if `value` is close to pi/2 AND `value +/- sigma` crosses
    #    the point pi/2 -> tan AND propegated sigma go to inf.
    # 3. if `value` AND either bounds are very close to pi/2
    #    -> tan AND propegated bounds go to inf.
    # 4. if the `variance` places `value` closer to pi/2 than, e.g.,
    #    a 3-sigma CI  -> either bounds might go to inf.
    #
    # and make a ruling:
    # a. use EasyScience guardrails and raise an error to prevent the user from doing something bad.
    # b. let Scipp handle everything and allow the user to do something bad (maybe show a warning).

    return _apply_math_operation(argument, sc.tan, np.tan, 'tan')


def arcsin(argument: Parameter) -> Parameter:
    """
    Calculates the inverse trigonometric sine (arcsine) of the given parameter.

    This function safely wraps the Scipp arcsine execution to guarantee correct
    error propagation. It automatically calculates new fitting bounds using
    a grid-based evaluation.

    Parameters
    ----------
    argument : Parameter
        The input parameter or numerical value to be evaluated. It should
        be a dimensionless ratio strictly within the domain [-1.0, 1.0].

    Returns
    -------
    Parameter
        A new parameter object containing the calculated angle (typically in radians),
        propagated variance, and mapped bounds.

    Raises
    ------
    TypeError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the value or fitting bounds of the input parameter exceed the
        domain [-1.0, 1.0], or if the bounds are infinite.

    Notes
    -----
    The output parameter's range is restricted to [-pi/2, pi/2].
    """

    # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
    return _apply_math_operation(argument, sc.asin, np.arcsin, 'arcsin')


def arccos(argument: Parameter) -> Parameter:
    """
    Calculates the inverse trigonometric cosine (arccosine) of the given parameter.

    This function safely wraps the Scipp arccosine execution to guarantee correct
    error propagation. It automatically calculates new fitting bounds using
    a grid-based evaluation.

    Parameters
    ----------
    argument : Parameter
        The input parameter or numerical value to be evaluated. It should
        be a dimensionless ratio strictly within the domain [-1.0, 1.0].

    Returns
    -------
    Parameter
        A new parameter object containing the calculated angle (typically in radians),
        propagated variance, and mapped bounds.

    Raises
    ------
    TypeError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the value or fitting bounds of the input parameter exceed the
        domain [-1.0, 1.0], or if the bounds are infinite.

    Notes
    -----
    The output parameter's range is restricted to [0, pi]. As arccosine is
    a monotonically decreasing function, the minimum bound of the input will
    map to the maximum bound of the output.
    """
    # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
    return _apply_math_operation(argument, sc.acos, np.arccos, 'arccos')


def arctan(argument: Parameter) -> Parameter:
    """
    Calculates the inverse trigonometric tangent (arctangent) of the given parameter.

    This function safely wraps the Scipp arctangent execution to guarantee correct
    error propagation. It automatically calculates new fitting bounds using
    a grid-based evaluation.

    Parameters
    ----------
    argument : Parameter
        The input parameter or numerical value to be evaluated. It should
        be a dimensionless number.

    Returns
    -------
    Parameter
        A new parameter object containing the calculated angle (typically in radians),
        propagated variance, and mapped bounds.

    Raises
    ------
    TypeError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the fitting bounds of the input parameter are infinite.

    Notes
    -----
    The output parameter's range is restricted to [-pi/2, pi/2].
    """
    # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
    return _apply_math_operation(argument, sc.atan, np.arctan, 'arctan')


def _arctan2(y: Parameter, x: Parameter) -> Parameter:
    """
    Calculates the element-wise inverse tangent of y/x determining the correct quadrant.

    Unlike ``arctan(y/x)``, this function uses the signs of both arguments
    to determine the correct quadrant of the result, returning an angle
    in the range [-pi, pi]. This safely avoids divide-by-zero errors when x=0.

    Parameters
    ----------
    y : Parameter
        The input y value (Cartesian y-coordinate).
    x : Parameter
        The input x value (Cartesian x-coordinate).
        At least one of the inputs (y or x) must be an EasyScience ``Parameter``.

    Returns
    -------
    Parameter
        A new parameter object containing the calculated angle (in radians),
        jointly propagated variance from both inputs, and mapped 2D bounds.

    Raises
    ------
    TypeError
        If the inputs are not of type ``Parameter``.
    ValueError
        If the fitting bounds of any input parameter are infinite.

    Notes
    -----
    Fitting bounds are mapped non-monotonically using a 2D meshgrid across
    the bounding box of both parameters.

    A ``UserWarning`` will be raised if the fitting space crosses the negative
    x-axis (y=0, x<0). This is the branch cut of the arctan2 function where
    values discontinue between -pi and pi, which can severely destabilize
    gradient-based optimization engines.
    """
    # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
    # TODO(): hash out issue with Scipp returning `VariancesError: Variances in argument 0 not supported.`
    return _apply_binary_math_operation(y, x, sc.atan2, np.arctan2, 'arctan2')


###############################################################################
## Hyperbolic Trigonometric Operations
###############################################################################

# list: sinh, cosh, tanh, asinh, acosh, atanh

###############################################################################
## Helpers
###############################################################################


def _apply_math_operation(
    argument: Parameter,
    sc_func: ScippMathFunc,
    np_func: NumPyMathFunc,
    operation_name: str,
) -> Parameter:
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

    parameter = Parameter.from_scipp(
        name=f'{operation_name}({argument.name})',
        full_value=scipp_object,
        min=new_min,
        max=new_max,
    )

    # parameter.name = parameter.unique_name

    return parameter


def _apply_binary_math_operation(y, x, sc_func, np_func, operation_name):
    """
    Universal wrapper for binary mathematical operations (y, x).
    """
    if not isinstance(y, Parameter):
        raise TypeError(f'Argument `y` must be a Parameter. Found: {type(y)}')
    if not isinstance(x, Parameter):
        raise TypeError(f'Argument `x` must be a Parameter. Found: {type(x)}')

    if not isinstance(y, Parameter) and not isinstance(x, Parameter):
        raise TypeError('At least one argument must be an EasyScience Parameter.')

    for arg in (y, x):
        if isinstance(arg, Parameter) and (np.isinf(arg.min) or np.isinf(arg.max)):
            raise ValueError('Argument bounds cannot be infinite.')

    # 1. Evaluate 2D bounds
    new_min, new_max = _derive_binary_grid_bounds(y, x, np_func)

    # 2. Package to Scipp and execute
    sc_y = _to_scipp_scalar(y)
    sc_x = _to_scipp_scalar(x)

    scipp_object = sc_func(y=sc_y, x=sc_x)

    # 3. Generate combined name
    y_name = y.name if isinstance(y, Parameter) else str(y)
    x_name = x.name if isinstance(x, Parameter) else str(x)

    return Parameter.from_scipp(
        name=f'{operation_name}({y_name}, {x_name})',
        full_value=scipp_object,
        min=new_min,
        max=new_max,
    )


###############################################################################


def _validate_parameter(argument):
    """Enforces type and finite bound requirements."""
    if not isinstance(argument, Parameter):
        raise TypeError(f'Argument must be a `Parameter`. Found: {type(argument)}')

    # Without the check below, defaylt inf bounds with result in numpy returning:
    #
    # `RuntimeWarning: invalid value encountered in multiply y *= step`
    # `RuntimeWarning: invalid value encountered in add y += start`
    #
    # and calculated bounds will be NaNs

    # if np.isinf(argument.min) or np.isinf(argument.max):
    #     raise ValueError(
    #         f'Argument bounds cannot be infinite. Found min: {argument.min}, max: {argument.max}'
    #     )


def _derive_grid_bounds(
    argument: Parameter, np_func: NumPyMathFunc, grid_size: int = 100
) -> tuple[float, float]:
    """Derives bounds for non-monotonic functions (sin, cos) using interval evaluation."""
    grid = np.linspace(argument.min, argument.max, grid_size)
    mapped_grid = np_func(grid)
    return float(np.min(mapped_grid)), float(np.max(mapped_grid))


def _derive_tan_bounds(
    argument: Parameter,
    np_func: NumPyMathFunc,
    num_tolerance: float = 1e-7,
    sigma_threshold: float = 3.0,
) -> tuple[float, float]:
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


def _derive_binary_grid_bounds(y, x, np_func, grid_size=50):
    """
    Derives bounds for 2D parameter spaces using interval evaluation.
    Also checks for the atan2 branch cut along the negative x-axis.
    """

    def _get_range(arg):
        if isinstance(arg, Parameter):
            return np.linspace(arg.min, arg.max, grid_size), arg.min, arg.max
        val = float(arg)
        return np.array([val]), val, val

    y_grid, y_min, y_max = _get_range(y)
    x_grid, x_min, x_max = _get_range(x)

    # Branch cut warning: crosses y=0 while x is negative
    if (y_min < 0 < y_max) and (x_min < 0):
        warnings.warn(
            'The 2D parameter bounds cross the negative x-axis (y=0, x<0). '
            'This is the atan2 branch cut where the angle jumps between -pi and pi. '
            'Optimization derivatives (Jacobian) will be highly unstable here.',
            UserWarning,
        )

    # Create a 2D plane of all possible (y, x) combinations in the bounds
    Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')
    mapped_grid = np_func(Y, X)

    return float(np.min(mapped_grid)), float(np.max(mapped_grid))


def _to_scipp_scalar(arg):
    """Safely converts a Parameter to a Scipp scalar."""
    if isinstance(arg, Parameter):
        variance = arg.variance if arg.variance is not None else 0.0
        return sc.scalar(value=arg.value, variance=variance, unit=arg.unit)
    return sc.scalar(value=float(arg), unit='')
