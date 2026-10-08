import warnings
from collections.abc import Callable

import numpy as np
import scipp as sc

from easyscience.variable.descriptor_array import DescriptorArray
from easyscience.variable.descriptor_number import DescriptorNumber
from easyscience.variable.parameter import Parameter

# Define type aliases for readability if preferred, or use inline Callable
# np_func takes a NumPy array and returns a NumPy array
NumPyMathFunc = Callable[[np.ndarray], np.ndarray]
# sc_func takes a Scipp Variable and returns a Scipp Variable
ScippMathFunc = Callable[[sc.Variable], sc.Variable]


# ==========================================
# Standard Maths Operations
# ==========================================


def exp(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.exp, np.exp, _derive_monotonic_bounds, 'exp', x)


def log(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.log, np.log, _derive_monotonic_bounds, 'log', x)


# ==========================================
# Trigonometric Operations
# ==========================================


def sin(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.sin, np.sin, _derive_grid_bounds, 'sin', x)


def cos(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.cos, np.cos, _derive_grid_bounds, 'cos', x)


def tan(x: Parameter) -> Parameter:
    """ """
    return _call_scipp_func(sc.tan, np.tan, _derive_tan_bounds, 'tan', x)


def arcsin(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.asin, np.arcsin, _derive_grid_bounds, 'arcsin', x)


def arccos(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.acos, np.arccos, _derive_grid_bounds, 'arccos', x)


def arctan(x: Parameter) -> Parameter:
    return _call_scipp_func(sc.atan, np.arctan, _derive_grid_bounds, 'arctan', x)


def arctan2(y: Parameter, x: Parameter) -> Parameter:
    return _call_scipp_func(sc.atan2, np.arctan2, _derive_2d_bounds, 'arctan2', y, x)


def sinc(argument: Parameter) -> Parameter:
    """
    Calculates the unnormalized sinc function (sin(x)/x) of the given parameter.

    This function safely wraps the Scipp sinc execution to guarantee correct
    error propagation. It evaluates new fitting bounds using a grid-based
    evaluation, accounting for the oscillating nature of the sinc curve.

    Parameters
    ----------
    argument : Parameter
        The input parameter to be evaluated. It should have angular units
        (e.g., 'rad').

    Returns
    -------
    Parameter
        A new parameter object containing the calculated sinc value,
        propagated variance, and mapped bounds. The output is dimensionless.

    Raises
    ------
    NotImplementedError
        If the input argument is not of type ``Parameter``.
    ValueError
        If the fitting bounds of the input parameter are infinite.

    Notes
    -----
    This function computes the unnormalized sinc function, matching the
    behavior of Scipp and standard mathematical definitions in physics,
    rather than the normalized signal-processing sinc function used by NumPy.
    The peak value is 1.0 at x=0.

    Examples
    --------
    Compute the unnormalized sinc of a Parameter at the origin (x=0):

      >>> import easyscience as es
      >>> angle = es.Parameter(
      ...     name='angle',
      ...     value=0.0,
      ...     variance=0.1,
      ...     min=-np.pi,
      ...     max=np.pi,
      ...     fixed=False,
      ...     unit='rad',
      ... )
      >>> es.sinc(angle)
      <Parameter 'sinc(angle)': 1.0 ± 0.0, bounds=[-0.21723362821122166:1.0]>

    Compute the unnormalized sinc at pi/2:

      >>> angle_half_pi = es.Parameter(
      ...     name='angle_half_pi',
      ...     value=np.pi / 2,
      ...     min=0.0,
      ...     max=np.pi,
      ...     fixed=False,
      ...     unit='rad',
      ... )
      >>> es.sinc(angle_half_pi)
      <Parameter 'sinc(angle_half_pi)': 0.6366197723675814 ± 0.0, bounds=[0.0:1.0]>
    """
    return _call_scipp_func(sc.sinc, _unnormalized_np_sinc, _derive_grid_bounds, 'sinc', argument)


# ==========================================
# Helpers
# ==========================================


def _to_scipp_scalar(arg: Parameter, strip_variance: bool = False) -> sc.Variable:
    """Converts a Parameter to a Scipp scalar, optionally stripping variance."""
    kwargs = {'value': arg.value, 'unit': str(arg.unit) if arg.unit else ''}
    if not strip_variance and arg.variance is not None and arg.variance > 0:
        kwargs['variance'] = float(arg.variance)
    return sc.scalar(**kwargs)


def _call_scipp_func(sc_func, np_func, bounds_deriver, op_name, *args) -> Parameter:
    """
    Central dispatcher for EasyScience math operations.
    Strips EasyScience types, evaluates bounds, calls Scipp, and repacks.
    """
    # 1. Strict EasyScience Caveats Enforced Here
    for arg in args:
        if not isinstance(arg, Parameter):
            raise NotImplementedError(
                f'EasyScience math functions currently only support `Parameter` objects. '
                f'Raw numbers and arrays are not yet supported. Found: {type(arg)}'
            )
        if op_name != 'arctan' and (np.isinf(arg.min) or np.isinf(arg.max)):
            raise ValueError(f"Bounds for '{arg.name}' cannot be infinite.")

    # 2. Derive new bounds based on the injected logic (Grid, Monotonic, 2D, etc.)
    new_min, new_max = bounds_deriver(np_func, *args)

    # 3. Convert to Scipp (Strip variances for atan2 due to missing C++ support)
    is_atan2 = sc_func is sc.atan2
    sc_args = {
        f'arg{i}': _to_scipp_scalar(arg, strip_variance=is_atan2) for i, arg in enumerate(args)
    }

    # 4. Execute Scipp Operation
    if is_atan2:
        scipp_object = sc_func(y=sc_args['arg0'], x=sc_args['arg1'])
        # Manual Delta Method for atan2
        y_var, x_var = getattr(args[0], 'variance', 0) or 0, getattr(args[1], 'variance', 0) or 0
        if y_var > 0 or x_var > 0:
            y_val, x_val = args[0].value, args[1].value
            scipp_object.variance = ((x_val**2) * y_var + (y_val**2) * x_var) / (
                (x_val**2 + y_val**2) ** 2
            )
    else:
        scipp_object = sc_func(sc_args['arg0'])

    # 5. Build dynamic name and return
    arg_names = ', '.join(arg.name for arg in args)

    return Parameter.from_scipp(
        name=f'{op_name}({arg_names})',
        full_value=scipp_object,
        min=new_min,
        max=new_max,
    )


def _derive_monotonic_bounds(np_func, x):
    """For monotonically increasing functions like exp."""
    return float(np_func(x.min)), float(np_func(x.max))


def _derive_grid_bounds(np_func, x, grid_size=100):
    """For non-monotonic unary functions like sin, cos."""
    grid = np.linspace(x.min, x.max, grid_size)
    mapped_grid = np_func(grid)
    return float(np.min(mapped_grid)), float(np.max(mapped_grid))


def _derive_2d_bounds(np_func, y, x, grid_size=50):
    """For binary functions like arctan2."""
    Y, X = np.meshgrid(
        np.linspace(y.min, y.max, grid_size), np.linspace(x.min, x.max, grid_size), indexing='ij'
    )
    mapped_grid = np_func(Y, X)
    return float(np.min(mapped_grid)), float(np.max(mapped_grid))


def _derive_tan_bounds(np_func, x):
    """
    Derives bounds for tangent.
    Guards the parameter value against asymptotes to prevent variance explosion,
    but allows the fitting bounds to blow up to infinity if they cross an asymptote.
    """
    unit_str = str(x.unit) if x.unit else 'rad'

    # 1. Guard the VALUE from blowing up
    val_rad = x.value if unit_str == 'rad' else np.deg2rad(x.value)

    # Asymptotes occur at pi/2 + n*pi. Modulo pi maps everything to [0, pi]
    # We check if the value is dangerously close to pi/2
    if np.isclose(val_rad % np.pi, np.pi / 2, atol=1e-7):
        raise ValueError(
            f'Parameter value {x.value} {unit_str} is dangerously close to a '
            'tangent asymptote. This will cause the propagated variance to explode.'
        )

    # 2. Allow BOUNDS to blow up mathematically
    if np.isinf(x.min) or np.isinf(x.max):
        return -np.inf, np.inf

    min_rad = x.min if unit_str == 'rad' else np.deg2rad(x.min)
    max_rad = x.max if unit_str == 'rad' else np.deg2rad(x.max)

    branch_min = np.floor((min_rad - np.pi / 2) / np.pi)
    branch_max = np.floor((max_rad - np.pi / 2) / np.pi)

    # If the min and max are on different branches, they crossed an asymptote.
    # The bounds mathematically blow up to encompass all real numbers.
    if branch_min != branch_max:
        return -np.inf, np.inf

    # Otherwise, map them normally
    return float(np_func(min_rad)), float(np_func(max_rad))


def _unnormalized_np_sinc(x):
    """
    Evaluates the unnormalized sinc function (sin(x)/x).
    Uses np.sinc (which is normalized) and scales by pi to safely
    avoid divide-by-zero singularities at x=0.
    """
    return np.sinc(x / np.pi)


# def sin(argument: Parameter) -> Parameter:
#     """Calculates the trigonometric sine of the given parameter.

#     This function safely wraps the Scipp sine execution to guarantee correct
#     error propagation and unit conversions. It automatically calculates new
#     fitting bounds using a grid-based evaluation to properly account for the
#     non-monotonic nature of the sine function.

#     Parameters
#     ----------
#     argument : Parameter
#         The input parameter or numerical value to be evaluated. It should
#         have angular units (e.g., 'rad', 'deg').

#     Returns
#     -------
#     : Same type as input
#         A new parameter object containing the calculated sine value,
#         propagated variance, mapped bounds, and a dimensionless unit.

#     Raises
#     ------
#     TypeError
#         If the input argument is not of type ``Parameter``.
#     ValueError
#         If the fitting bounds of the input parameter are infinite.

#     Examples
#     --------
#     Compute sine of angles in radians:

#       >>> import easyscience as es
#       >>> import numpy as np
#       >>> angle = es.Parameter(
#       ...     name='angle',
#       ...     value=np.pi / 4,
#       ...     min=0.0,
#       ...     max=1.0,
#       ...     fixed=False,
#       ...     unit='rad',
#       ... )
#       >>> es.sin(angle)
#      <Parameter 'sin(angle)': 0.7071, bounds=[0.0:0.8414709848078965]>

#     The input must have angle units (rad or deg):

#       >>> angle_deg = es.Parameter(
#       ...     name='angle_deg', value=45, min=0, max=90, fixed=False, unit='deg'
#       ... )
#       >>> es.sin(angle_deg)
#       <Parameter 'sin(angle_deg)': 0.7071, bounds=[-0.999981805600717:0.9998362543788631]>
#     """
#     return _apply_math_operation(argument, sc.sin, np.sin, 'sin')


# def cos(argument: Parameter) -> Parameter:
#     """
#     Calculates the trigonometric cosine of the given parameter.

#     This function safely wraps the Scipp cosine execution to guarantee correct
#     error propagation and unit conversions. It automatically calculates new
#     fitting bounds using a grid-based evaluation to properly account for the
#     non-monotonic nature of the cosine function.

#     Parameters
#     ----------
#     argument : Parameter
#         The input parameter or numerical value to be evaluated. It should
#         have angular units (e.g., 'rad', 'deg').

#     Returns
#     -------
#     Parameter
#         A new parameter object containing the calculated cosine value,
#         propagated variance, mapped bounds, and a dimensionless unit.

#     Raises
#     ------
#     TypeError
#         If the input argument is not of type ``Parameter``.
#     ValueError
#         If the fitting bounds of the input parameter are infinite.
#     """
#     return _apply_math_operation(argument, sc.cos, np.cos, 'cos')


# def tan(argument: Parameter) -> Parameter:
#     """
#     Calculates the trigonometric tangent of the given parameter.

#     This function safely wraps the Scipp tangent execution to guarantee correct
#     error propagation and unit conversions. It implements strict numerical
#     and statistical guardrails around asymptotes (pi/2 + n*pi) to prevent
#     optimization crashes and physically invalid variances during fitting.

#     Parameters
#     ----------
#     argument : Parameter
#         The input parameter or numerical value to be evaluated. It should
#         have angular units (e.g., 'rad', 'deg').

#     Returns
#     -------
#     Parameter
#         A new parameter object containing the calculated tangent value,
#         propagated variance, mapped bounds, and a dimensionless unit.

#     Raises
#     ------
#     TypeError
#         If the input argument is not of type ``Parameter``.
#     ValueError
#         If the fitting bounds are infinite, if the value touches an asymptote,
#         or if the bounds straddle an asymptote.

#     Notes
#     -----
#     Because Scipp uses the Delta Method (first-order Taylor expansion) for
#     error propagation, a ``UserWarning`` will be raised if the parameter's
#     variance places its 3-sigma confidence interval across an asymptote. In
#     this scenario, the linear error propagation reported is highly inaccurate.
#     """

#     # TODO(): check these cases:
#     # 1. if `value` is very close to pi/2 -> tan goes to inf.
#     # 2. if `value` is close to pi/2 AND `value +/- sigma` crosses
#     #    the point pi/2 -> tan AND propegated sigma go to inf.
#     # 3. if `value` AND either bounds are very close to pi/2
#     #    -> tan AND propegated bounds go to inf.
#     # 4. if the `variance` places `value` closer to pi/2 than, e.g.,
#     #    a 3-sigma CI  -> either bounds might go to inf.
#     #
#     # and make a ruling:
#     # a. use EasyScience guardrails and raise an error to prevent the user from doing something bad.
#     # b. let Scipp handle everything and allow the user to do something bad (maybe show a warning).

#     return _apply_math_operation(argument, sc.tan, np.tan, 'tan')


# def arcsin(argument: Parameter) -> Parameter:
#     """
#     Calculates the inverse trigonometric sine (arcsine) of the given parameter.

#     This function safely wraps the Scipp arcsine execution to guarantee correct
#     error propagation. It automatically calculates new fitting bounds using
#     a grid-based evaluation.

#     Parameters
#     ----------
#     argument : Parameter
#         The input parameter or numerical value to be evaluated. It should
#         be a dimensionless ratio strictly within the domain [-1.0, 1.0].

#     Returns
#     -------
#     Parameter
#         A new parameter object containing the calculated angle (typically in radians),
#         propagated variance, and mapped bounds.

#     Raises
#     ------
#     TypeError
#         If the input argument is not of type ``Parameter``.
#     ValueError
#         If the value or fitting bounds of the input parameter exceed the
#         domain [-1.0, 1.0], or if the bounds are infinite.

#     Notes
#     -----
#     The output parameter's range is restricted to [-pi/2, pi/2].
#     """

#     # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
#     return _apply_math_operation(argument, sc.asin, np.arcsin, 'arcsin')


# def arccos(argument: Parameter) -> Parameter:
#     """
#     Calculates the inverse trigonometric cosine (arccosine) of the given parameter.

#     This function safely wraps the Scipp arccosine execution to guarantee correct
#     error propagation. It automatically calculates new fitting bounds using
#     a grid-based evaluation.

#     Parameters
#     ----------
#     argument : Parameter
#         The input parameter or numerical value to be evaluated. It should
#         be a dimensionless ratio strictly within the domain [-1.0, 1.0].

#     Returns
#     -------
#     Parameter
#         A new parameter object containing the calculated angle (typically in radians),
#         propagated variance, and mapped bounds.

#     Raises
#     ------
#     TypeError
#         If the input argument is not of type ``Parameter``.
#     ValueError
#         If the value or fitting bounds of the input parameter exceed the
#         domain [-1.0, 1.0], or if the bounds are infinite.

#     Notes
#     -----
#     The output parameter's range is restricted to [0, pi]. As arccosine is
#     a monotonically decreasing function, the minimum bound of the input will
#     map to the maximum bound of the output.
#     """
#     # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
#     return _apply_math_operation(argument, sc.acos, np.arccos, 'arccos')


# def arctan(argument: Parameter) -> Parameter:
#     """
#     Calculates the inverse trigonometric tangent (arctangent) of the given parameter.

#     This function safely wraps the Scipp arctangent execution to guarantee correct
#     error propagation. It automatically calculates new fitting bounds using
#     a grid-based evaluation.

#     Parameters
#     ----------
#     argument : Parameter
#         The input parameter or numerical value to be evaluated. It should
#         be a dimensionless number.

#     Returns
#     -------
#     Parameter
#         A new parameter object containing the calculated angle (typically in radians),
#         propagated variance, and mapped bounds.

#     Raises
#     ------
#     TypeError
#         If the input argument is not of type ``Parameter``.
#     ValueError
#         If the fitting bounds of the input parameter are infinite.

#     Notes
#     -----
#     The output parameter's range is restricted to [-pi/2, pi/2].
#     """
#     # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
#     return _apply_math_operation(argument, sc.atan, np.arctan, 'arctan')


# def arctan2(y: Parameter, x: Parameter) -> Parameter:
#     """
#     Calculates the element-wise inverse tangent of y/x determining the correct quadrant.

#     Unlike ``arctan(y/x)``, this function uses the signs of both arguments
#     to determine the correct quadrant of the result, returning an angle
#     in the range [-pi, pi]. This safely avoids divide-by-zero errors when x=0.

#     Parameters
#     ----------
#     y : Parameter
#         The input y value (Cartesian y-coordinate).
#     x : Parameter
#         The input x value (Cartesian x-coordinate).
#         At least one of the inputs (y or x) must be an EasyScience ``Parameter``.

#     Returns
#     -------
#     Parameter
#         A new parameter object containing the calculated angle (in radians),
#         jointly propagated variance from both inputs, and mapped 2D bounds.

#     Raises
#     ------
#     TypeError
#         If the inputs are not of type ``Parameter``.
#     ValueError
#         If the fitting bounds of any input parameter are infinite.

#     Notes
#     -----
#     Fitting bounds are mapped non-monotonically using a 2D meshgrid across
#     the bounding box of both parameters.

#     A ``UserWarning`` will be raised if the fitting space crosses the negative
#     x-axis (y=0, x<0). This is the branch cut of the arctan2 function where
#     values discontinue between -pi and pi, which can severely destabilize
#     gradient-based optimization engines.
#     """
#     # TODO(): inverse trig functions are not hashed out; handling of domain and range needs work
#     # TODO(): hash out issue with Scipp returning `VariancesError: Variances in argument 0 not supported.`
#     return _apply_binary_math_operation(y, x, sc.atan2, np.arctan2, 'arctan2')


# ###############################################################################
# ## Hyperbolic Trigonometric Operations
# ###############################################################################

# # list: sinh, cosh, tanh, asinh, acosh, atanh

# ###############################################################################
# ## Helpers
# ###############################################################################


# def _apply_math_operation(
#     argument: Parameter,
#     sc_func: ScippMathFunc,
#     np_func: NumPyMathFunc,
#     operation_name: str,
# ) -> Parameter:
#     """
#     Universal wrapper that routes bounds logic internally
#     and handles Scipp execution.
#     """
#     _validate_parameter(argument)

#     # Route to the correct bounds derivation logic
#     if operation_name == 'tan':
#         new_min, new_max = _derive_tan_bounds(argument, np_func)
#     else:
#         new_min, new_max = _derive_grid_bounds(argument, np_func)

#     # Pass values to Scipp for strict execution and error propagation
#     scipp_object = sc_func(
#         sc.scalar(value=argument.value, variance=argument.variance, unit=argument.unit)
#     )

#     parameter = Parameter.from_scipp(
#         name=f'{operation_name}({argument.name})',
#         full_value=scipp_object,
#         min=new_min,
#         max=new_max,
#     )

#     # parameter.name = parameter.unique_name

#     return parameter


# def _apply_binary_math_operation(y, x, sc_func, np_func, operation_name):
#     """
#     Universal wrapper for binary mathematical operations (y, x).
#     """
#     if not isinstance(y, Parameter):
#         raise TypeError(f'Argument `y` must be a Parameter. Found: {type(y)}')
#     if not isinstance(x, Parameter):
#         raise TypeError(f'Argument `x` must be a Parameter. Found: {type(x)}')

#     if not isinstance(y, Parameter) and not isinstance(x, Parameter):
#         raise TypeError('At least one argument must be an EasyScience Parameter.')

#     # for arg in (y, x):
#     #     if isinstance(arg, Parameter) and (np.isinf(arg.min) or np.isinf(arg.max)):
#     #         raise ValueError('Argument bounds cannot be infinite.')

#     # 1. Evaluate 2D bounds
#     new_min, new_max = _derive_binary_grid_bounds(y, x, np_func)

#     # 2. Package to Scipp and execute
#     sc_y = _to_scipp_scalar(y)
#     sc_x = _to_scipp_scalar(x)

#     scipp_object = sc_func(y=sc_y, x=sc_x)

#     # 3. Generate combined name
#     y_name = y.name if isinstance(y, Parameter) else str(y)
#     x_name = x.name if isinstance(x, Parameter) else str(x)

#     return Parameter.from_scipp(
#         name=f'{operation_name}({y_name}, {x_name})',
#         full_value=scipp_object,
#         min=new_min,
#         max=new_max,
#     )


# ###############################################################################


# def _validate_parameter(argument):
#     """Enforces type and finite bound requirements."""
#     if not isinstance(argument, Parameter):
#         raise TypeError(f'Argument must be a `Parameter`. Found: {type(argument)}')

#     # Without the check below, defaylt inf bounds with result in numpy returning:
#     #
#     # `RuntimeWarning: invalid value encountered in multiply y *= step`
#     # `RuntimeWarning: invalid value encountered in add y += start`
#     #
#     # and calculated bounds will be NaNs

#     # if np.isinf(argument.min) or np.isinf(argument.max):
#     #     raise ValueError(
#     #         f'Argument bounds cannot be infinite. Found min: {argument.min}, max: {argument.max}'
#     #     )


# def _derive_grid_bounds(
#     argument: Parameter, np_func: NumPyMathFunc, grid_size: int = 100
# ) -> tuple[float, float]:
#     """Derives bounds for non-monotonic functions (sin, cos) using interval evaluation."""
#     grid = np.linspace(argument.min, argument.max, grid_size)
#     mapped_grid = np_func(grid)
#     return float(np.min(mapped_grid)), float(np.max(mapped_grid))


# def _derive_tan_bounds(
#     argument: Parameter,
#     np_func: NumPyMathFunc,
#     num_tolerance: float = 1e-7,
#     sigma_threshold: float = 3.0,
# ) -> tuple[float, float]:
#     """
#     Derives bounds for tangent, enforcing both a numerical keep-out zone
#     and a statistical keep-out zone around asymptotes.
#     """
#     unit_str = str(argument.unit) if argument.unit else 'rad'

#     # Standardize to radians
#     min_rad = argument.min if unit_str == 'rad' else np.deg2rad(argument.min)
#     max_rad = argument.max if unit_str == 'rad' else np.deg2rad(argument.max)
#     val_rad = argument.value if unit_str == 'rad' else np.deg2rad(argument.value)

#     def _dist_to_asymptote(rad_val):
#         dist = (rad_val - np.pi / 2) % np.pi
#         return min(dist, np.pi - dist)

#     # 1. Numerical check (prevents floating point explosions)
#     if _dist_to_asymptote(val_rad) < num_tolerance:
#         raise ValueError(
#             f'Parameter value {argument.value} {unit_str} is too close to a tangent asymptote.'
#         )

#     # 2. Statistical check (prevents Delta Method breakdown)
#     if argument.variance is not None and argument.variance > 0:
#         std_dev = np.sqrt(argument.variance)
#         std_rad = std_dev if unit_str == 'rad' else np.deg2rad(std_dev)

#         # If the asymptote is within N standard deviations of the value
#         if _dist_to_asymptote(val_rad) < (sigma_threshold * std_rad):
#             warnings.warn(
#                 f"The parameter '{argument.name}' is {argument.value}±{std_dev:.2f} {unit_str}. "
#                 f'This means the {sigma_threshold}-sigma confidence interval crosses a tangent asymptote. '
#                 'Linear error propagation (variance) reported by Scipp will be highly inaccurate.',
#                 UserWarning,
#             )

#     # 3. Bounds check (same as before)
#     if _dist_to_asymptote(min_rad) < num_tolerance or _dist_to_asymptote(max_rad) < num_tolerance:
#         raise ValueError('Tangent bounds touch or encroach on an asymptote.')

#     branch_min = np.floor((min_rad - np.pi / 2) / np.pi)
#     branch_max = np.floor((max_rad - np.pi / 2) / np.pi)

#     if branch_min != branch_max:
#         raise ValueError('Tangent bounds cross an asymptote. Fitting is unstable.')

#     return float(np_func(min_rad)), float(np_func(max_rad))


# def _derive_binary_grid_bounds(y, x, np_func, grid_size=50):
#     """
#     Derives bounds for 2D parameter spaces using interval evaluation.
#     Also checks for the atan2 branch cut along the negative x-axis.
#     """

#     def _get_range(arg):
#         if isinstance(arg, Parameter):
#             return np.linspace(arg.min, arg.max, grid_size), arg.min, arg.max
#         val = float(arg)
#         return np.array([val]), val, val

#     y_grid, y_min, y_max = _get_range(y)
#     x_grid, x_min, x_max = _get_range(x)

#     # Branch cut warning: crosses y=0 while x is negative
#     if (y_min < 0 < y_max) and (x_min < 0):
#         warnings.warn(
#             'The 2D parameter bounds cross the negative x-axis (y=0, x<0). '
#             'This is the atan2 branch cut where the angle jumps between -pi and pi. '
#             'Optimization derivatives (Jacobian) will be highly unstable here.',
#             UserWarning,
#         )

#     # Create a 2D plane of all possible (y, x) combinations in the bounds
#     Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')
#     mapped_grid = np_func(Y, X)

#     return float(np.min(mapped_grid)), float(np.max(mapped_grid))


# def _to_scipp_scalar(arg):
#     """Safely converts a Parameter to a Scipp scalar."""
#     if isinstance(arg, Parameter):
#         variance = arg.variance if arg.variance is not None or 0.0
#         return sc.scalar(value=arg.value, variance=variance, unit=arg.unit)
#     return sc.scalar(value=float(arg), unit='')
