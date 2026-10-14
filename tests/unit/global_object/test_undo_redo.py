# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest

from easyscience import DescriptorNumber
from easyscience import Fitter
from easyscience import ModelBase
from easyscience import Parameter
from easyscience.variable import DescriptorBool
from easyscience.variable import DescriptorStr


class Line(ModelBase):
    def __init__(self, m: Parameter, c: Parameter):
        super().__init__()
        self._m = m
        self._c = c

    @property
    def m(self) -> Parameter:
        return self._m

    @m.setter
    def m(self, value: float) -> None:
        self._m.value = value

    @property
    def c(self) -> Parameter:
        return self._c

    @c.setter
    def c(self, value: float) -> None:
        self._c.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.m.value * x + self.c.value


def createSingleObjs(idx):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    reps = math.floor(idx / len(alphabet)) + 1
    name = alphabet[idx % len(alphabet)] * reps
    if idx % 2:
        return Parameter(idx, unit='m/s', display_name=name)
    else:
        return DescriptorNumber(idx, unit='m/s', display_name=name)


def createParam(option):
    return pytest.param(option, id=option[0])


def doUndoRedo(obj, attr, future, additional=''):
    from easyscience import global_object

    global_object.stack.enabled = True
    e = False

    def getter(_obj, _attr):
        value = getattr(_obj, _attr)
        if additional:
            value = getattr(value, additional)
        return value

    try:
        previous = getter(obj, attr)
        if attr == 'unit' and hasattr(obj, 'convert_unit'):
            obj.convert_unit(future)
        else:
            setattr(obj, attr, future)
        assert getter(obj, attr) == future
        assert global_object.stack.canUndo()
        global_object.stack.undo()
        assert getter(obj, attr) == previous
        assert global_object.stack.canRedo()
        global_object.stack.redo()
        assert getter(obj, attr) == future
    except Exception as err:
        e = err
    finally:
        global_object.stack.enabled = False
    return e


@pytest.mark.parametrize(
    'test',
    [
        createParam(option)
        for option in [
            ('value', 500),
            ('error', 5),
            ('unit', 'km/s'),
            ('display_name', 'boom'),
        ]
    ],
)
def test_DescriptorNumberUndoRedo(test):
    obj = DescriptorNumber(1, unit='m/s', display_name='DescriptorNumber')
    attr = test[0]
    value = test[1]

    e = doUndoRedo(obj, attr, value)
    assert not e


def test_DescriptorBoolUndoRedo():
    obj = DescriptorBool(False, display_name='DescriptorBool')
    attr = 'value'
    value = True

    e = doUndoRedo(obj, attr, value)
    assert not e

    obj = DescriptorBool(True, display_name='DescriptorBool')
    attr = 'value'
    value = False

    e = doUndoRedo(obj, attr, value)
    assert not e


def test_DescriptorStrUndoRedo():
    obj = DescriptorStr('Foo', display_name='DescriptorStr')
    attr = 'value'
    value = 'Bar'

    e = doUndoRedo(obj, attr, value)
    assert not e


@pytest.mark.parametrize(
    'test',
    [
        createParam(option)
        for option in [
            ('value', 500),
            ('error', 5),
            ('unit', 'km/s'),
            ('display_name', 'boom'),
            ('fixed', False),
            ('max', 505),
            ('min', -1),
        ]
    ],
)
def test_ParameterUndoRedo(test):
    obj = Parameter(1, unit='m/s', display_name='Parameter')
    attr = test[0]
    value = test[1]

    e = doUndoRedo(obj, attr, value)
    assert not e


def test_Parameter_Bounds_UndoRedo():
    from easyscience import global_object

    global_object.stack.enabled = True
    parameter = Parameter(1, display_name='test')
    assert parameter.min == -np.inf
    assert parameter.max == np.inf

    parameter.min = 0
    parameter.max = 2
    assert parameter.min == 0
    assert parameter.max == 2

    global_object.stack.undo()
    global_object.stack.undo()
    assert parameter.min == -np.inf
    assert parameter.max == np.inf


def test_ModelBaseUndoRedo():
    line = Line(m=Parameter(1.0, display_name='m'), c=Parameter(2.0, display_name='c'))

    # Test setting value through the model's property setters
    for attr in ('m', 'c'):
        e = doUndoRedo(line, attr, getattr(line, attr).value + 1, 'value')
        assert not e


def test_UndoRedoMacros():
    items = [createSingleObjs(idx) for idx in range(5)]
    offset = 5
    undo_text = 'test_macro'
    from easyscience import global_object

    global_object.stack.enabled = True
    global_object.stack.beginMacro(undo_text)
    values = [item.value for item in items]

    for item, value in zip(items, values):
        item.value = value + offset
    global_object.stack.endMacro()

    for item, old_value in zip(items, values):
        assert item.value == old_value + offset
    assert global_object.stack.undoText() == undo_text

    global_object.stack.undo()

    for item, old_value in zip(items, values):
        assert item.value == old_value
    assert global_object.stack.redoText() == undo_text

    global_object.stack.redo()
    for item, old_value in zip(items, values):
        assert item.value == old_value + offset


@pytest.mark.parametrize('fit_engine', ['LMFit', 'Bumps', 'DFO'])
def test_fittingUndoRedo(fit_engine):
    m_value = 6
    c_value = 2
    x = np.linspace(-5, 5, 100)
    weights = np.ones_like(x)
    dy = np.random.rand(*x.shape)

    l1 = Line(m=Parameter(m_value, display_name='m'), c=Parameter(c_value, display_name='c'))
    m_sp = 4
    c_sp = -3

    l2 = Line(m=Parameter(m_sp, display_name='m'), c=Parameter(c_sp, display_name='c'))
    l2.m.fixed = False
    l2.c.fixed = False

    y = l1(x) + 0.125 * (dy - 0.5)

    f = Fitter(l2, l2)
    try:
        f.switch_minimizer(fit_engine)
    except AttributeError:
        pytest.skip(f'{fit_engine} is not installed')

    from easyscience import global_object

    global_object.stack.enabled = True
    res = f.fit(x, y, weights=weights)

    # assert l1.c.value == pytest.approx(l2.c.value, rel=l2.c.error * 3)
    # assert l1.m.value == pytest.approx(l2.m.value, rel=l2.m.error * 3)
    assert global_object.stack.undoText() == 'Fitting routine'

    global_object.stack.undo()
    assert l2.m.value == m_sp
    assert l2.c.value == c_sp
    assert global_object.stack.redoText() == 'Fitting routine'

    global_object.stack.redo()
    assert l2.m.value == res.p[f'p{l2.m.unique_name}']
    assert l2.c.value == res.p[f'p{l2.c.unique_name}']


# TODO: Check if this test is needed
# @pytest.mark.parametrize('math_funcs', [pytest.param([Parameter.__add__, float.__add__], id='Addition'),
#                                         pytest.param([Parameter.__sub__, float.__sub__], id='Subtraction')])
# def test_parameter_maths_basic(math_funcs):
#     a = 1.0
#     b = 2.0
#     sa = 0.1
#     sb = 0.2

#     p_fun = math_funcs[0]
#     f_fun = math_funcs[1]

#     result_value = f_fun(a, b)
#     result_error = (sa ** 2 + sb ** 2) ** 0.5

#     from easyscience import global_object
#     global_object.stack.enabled = True

#     # Perform basic test
#     p1 = Parameter('a', a)
#     p2 = Parameter('b', b)

#     p1 = p_fun(p1, p2)

#     assert p1.value == result_value
#     global_object.stack.undo()
#     assert p1.value == a
#     global_object.stack.redo()
#     assert p1.value == result_value

# # Perform basic + error
# p1 = Parameter('a', a, error=sa)
# p2 = Parameter('b', b, error=sb)
# p1 = p_fun(p1, p2)
# assert p1.value == result_value
# assert p1.error == result_error
# global_object.stack.undo()
# assert p1.value == a
# assert p1.error == sa
# global_object.stack.redo()
# assert p1.value == result_value
# assert p1.error == result_error

# # Perform basic + units
# p1 = Parameter('a', a, error=sa, units='m/s')
# p2 = Parameter('b', b, error=sb, units='m/s')
# p1 = p_fun(p1, p2)
# assert p1.value == result_value
# assert p1.error == result_error
# assert str(p1.unit) == 'meter / second'
# global_object.stack.undo()
# assert p1.value == a
# assert p1.error == sa
# assert str(p1.unit) == 'meter / second'
# global_object.stack.redo()
# assert p1.value == result_value
# assert p1.error == result_error
# assert str(p1.unit) == 'meter / second'


# @pytest.mark.parametrize('math_funcs', [pytest.param([Parameter.__imul__, float.__mul__,
#                                                       'meter ** 2 / second ** 2'], id='Multiplication'),
#                                         pytest.param([Parameter.__itruediv__, float.__truediv__,
#                                                       'dimensionless'], id='Division')])
# def test_parameter_maths_advanced(math_funcs):
#     a = 4.0
#     b = 2.0
#     sa = 0.1
#     sb = 0.2
#     unit = 'meter / second'

#     p_fun = math_funcs[0]
#     f_fun = math_funcs[1]
#     u_str = math_funcs[2]

#     result_value = f_fun(a, b)
#     result_error = ((sa / a) ** 2 + (sb / b) ** 2) ** 0.5 * result_value

#     from easyscience import global_object
#     global_object.stack.enabled = True

#     # Perform basic test
#     p1 = Parameter('a', a)
#     p2 = Parameter('b', b)

#     p1 = p_fun(p1, p2)
#     assert p1.value == result_value
#     global_object.stack.undo()
#     assert p1.value == a
#     global_object.stack.redo()
#     assert p1.value == result_value

#     # Perform basic + error
#     p1 = Parameter('a', a, error=sa)
#     p2 = Parameter('b', b, error=sb)
#     p1 = p_fun(p1, p2)
#     assert p1.value == result_value
#     assert p1.error == result_error
#     global_object.stack.undo()
#     assert p1.value == a
#     assert p1.error == sa
#     global_object.stack.redo()
#     assert p1.value == result_value
#     assert p1.error == result_error

#     # Perform basic + units
#     p1 = Parameter('a', a, error=sa, units=unit)
#     p2 = Parameter('b', b, error=sb, units=unit)
#     p1 = p_fun(p1, p2)
#     assert p1.value == result_value
#     assert p1.error == result_error
#     assert str(p1.unit) == u_str
#     global_object.stack.undo()
#     assert p1.value == a
#     assert p1.error == sa
#     assert str(p1.unit) == unit
#     global_object.stack.redo()
#     assert p1.value == result_value
#     assert p1.error == result_error
#     assert str(p1.unit) == u_str
