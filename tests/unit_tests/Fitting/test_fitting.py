#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = "github.com/wardsimon"
__version__ = "0.0.1"

import pytest

import numpy as np
from easyscience.Fitting.Constraints import ObjConstraint
from easyscience.Fitting.Fitting import Fitter
from easyscience.Fitting.Fitting import MultiFitter
from easyscience.Fitting.minimizers import FitError
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter



class Line(BaseObj):
    m: Parameter
    c: Parameter

    def __init__(self, m_val: float, c_val: float):
        m = Parameter("m", m_val)
        c = Parameter("c", c_val)
        super(Line, self).__init__("line", m=m, c=c)

    def __call__(self, x):
        return self.m.raw_value * x + self.c.raw_value
    

class AbsSin(BaseObj):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter("offset", offset_val)
        phase = Parameter("phase", phase_val)
        super().__init__("sin", offset=offset, phase=phase)

    def __call__(self, x):
        return np.abs(np.sin(self.phase.raw_value * x + self.offset.raw_value))


class AbsSin2D(BaseObj):
    phase: Parameter
    offset: Parameter

    def __init__(self, offset_val: float, phase_val: float):
        offset = Parameter("offset", offset_val)
        phase = Parameter("phase", phase_val)
        super().__init__("sin2D", offset=offset, phase=phase)

    def __call__(self, x):
        X = x[:, :, 0]   # x is a 2D array
        Y = x[:, :, 1]
        return np.abs(
            np.sin(self.phase.raw_value * X + self.offset.raw_value)
        ) * np.abs(np.sin(self.phase.raw_value * Y + self.offset.raw_value))


class AbsSin2DL(AbsSin2D):
    def __call__(self, x):
        X = x[:, 0]   # x is a 1D array
        Y = x[:, 1]
        return np.abs(
            np.sin(self.phase.raw_value * X + self.offset.raw_value)
        ) * np.abs(np.sin(self.phase.raw_value * Y + self.offset.raw_value))


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_basic_fit(fit_engine, with_errors):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    if fit_engine is not None:
        try:
            f.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    args = [x, y]
    kwargs = {}
    if with_errors:
        kwargs["weights"] = 1 / np.sqrt(y)
    result = f.fit(*args, **kwargs)

    if fit_engine is not None:
        assert result.fitting_engine.name == fit_engine
    assert sp_sin.phase.raw_value == pytest.approx(ref_sin.phase.raw_value, rel=1e-3)
    assert sp_sin.offset.raw_value == pytest.approx(ref_sin.offset.raw_value, rel=1e-3)


def check_fit_results(result, sp_sin, ref_sin, x, **kwargs):
    assert result.n_pars == len(sp_sin.get_fit_parameters())
    assert result.chi2 == pytest.approx(0, abs=1.5e-3 * (len(result.x) - result.n_pars))
    assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
    assert result.success
    if "sp_ref1" in kwargs.keys():
        sp_ref1 = kwargs["sp_ref1"]
        for key, value in sp_ref1.items():
            assert key in result.p.keys()
            assert key in result.p0.keys()
            assert result.p0[key] == pytest.approx(
                value
            )  # Bumps does something strange here
    assert np.all(result.x == x)
    for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values()):
        # assert item.error > 0 % This does not work as some methods don't calculate error
        assert item1.error == pytest.approx(0, abs=1e-1)
        assert item1.raw_value == pytest.approx(item2.raw_value, abs=5e-3)
    y_calc_ref = ref_sin(x)
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(sp_sin(x) - y_calc_ref, abs=1e-2)


@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_fit_result(fit_engine):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    sp_ref1 = {
        f"p{sp_sin._borg.map.convert_id_to_key(item1)}": item1.raw_value
        for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values())
    }
    sp_ref2 = {
        f"p{sp_sin._borg.map.convert_id_to_key(item1)}": item2.raw_value
        for item1, item2 in zip(sp_sin._kwargs.values(), ref_sin._kwargs.values())
    }

    f = Fitter(sp_sin, sp_sin)

    if fit_engine is not None:
        try:
            f.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    result = f.fit(x, y)
    check_fit_results(result, sp_sin, ref_sin, x, sp_ref1=sp_ref1, sp_ref2=sp_ref2)


@pytest.mark.parametrize("fit_method", ["leastsq", "powell", "cobyla"])
def test_lmfit_methods(fit_method):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    assert fit_method in f.available_methods()
    result = f.fit(x, y, method=fit_method)
    check_fit_results(result, sp_sin, ref_sin, x)


@pytest.mark.xfail(reason="known bumps issue")
@pytest.mark.parametrize("fit_method", ["newton", "lm"])
def test_bumps_methods(fit_method):
    ref_sin = AbsSin(0.2, np.pi)
    sp_sin = AbsSin(0.354, 3.05)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.offset.fixed = False
    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)
    f.switch_engine("bumps")
    assert fit_method in f.available_methods()
    result = f.fit(x, y, method=fit_method)
    check_fit_results(result, sp_sin, ref_sin, x)


@pytest.mark.parametrize("fit_engine", ["lmfit", "bumps", "DFO_LS"])
def test_fit_constraints(fit_engine):
    ref_sin = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin = AbsSin(1, 0.5)

    x = np.linspace(0, 5, 200)
    y = ref_sin(x)

    sp_sin.phase.fixed = False

    f = Fitter(sp_sin, sp_sin)

    assert len(f.fit_constraints()) == 0
    c = ObjConstraint(sp_sin.offset, "2*", sp_sin.phase)
    f.add_fit_constraint(c)

    if fit_engine is not None:
        try:
            f.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    result = f.fit(x, y)
    check_fit_results(result, sp_sin, ref_sin, x)
    assert len(f.fit_constraints()) == 1
    f.remove_fit_constraint(0)
    assert len(f.fit_constraints()) == 0



@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_multi_fit(fit_engine, with_errors):
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)

    ref_sin_1.offset.user_constraints["ref_sin2"] = ObjConstraint(
        ref_sin_2.offset, "", ref_sin_1.offset
    )
    ref_sin_1.offset.user_constraints["ref_sin2"]()

    sp_sin_1.offset.user_constraints["sp_sin2"] = ObjConstraint(
        sp_sin_2.offset, "", sp_sin_1.offset
    )
    sp_sin_1.offset.user_constraints["sp_sin2"]()

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x2 = np.copy(x1)
    y2 = ref_sin_2(x2)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False

    f = MultiFitter([sp_sin_1, sp_sin_2], [sp_sin_1, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    args = [[x1, x2], [y1, y2]]
    kwargs = {}
    if with_errors:
        kwargs["weights"] = [1 / np.sqrt(y1), 1 / np.sqrt(y2)]

    results = f.fit(*args, **kwargs)
    X = [x1, x2]
    Y = [y1, y2]
    F_ref = [ref_sin_1, ref_sin_2]
    F_real = [sp_sin_1, sp_sin_2]
    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin_1.get_fit_parameters()) + len(
            sp_sin_2.get_fit_parameters()
        )
        assert result.chi2 == pytest.approx(
            0, abs=1.5e-3 * (len(result.x) - result.n_pars)
        )
        assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_ref[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(
            F_real[idx](X[idx]) - F_ref[idx](X[idx]), abs=1e-2
        )


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_multi_fit2(fit_engine, with_errors):
    ref_sin_1 = AbsSin(0.2, np.pi)
    sp_sin_1 = AbsSin(0.354, 3.05)
    ref_sin_2 = AbsSin(np.pi * 0.45, 0.45 * np.pi * 0.5)
    sp_sin_2 = AbsSin(1, 0.5)#    ref_sin_1_obj = genObjs[0]
    ref_line_obj = Line(1, 4.6)

    ref_sin_1.offset.user_constraints["ref_sin2"] = ObjConstraint(
        ref_sin_2.offset, "", ref_sin_1.offset
    )
    ref_sin_1.offset.user_constraints["ref_line"] = ObjConstraint(
        ref_line_obj.m, "", ref_sin_1.offset
    )
    ref_sin_1.offset.user_constraints["ref_sin2"]()
    ref_sin_1.offset.user_constraints["ref_line"]()

    sp_line = Line(0.43, 6.1)

    sp_sin_1.offset.user_constraints["sp_sin2"] = ObjConstraint(
        sp_sin_2.offset, "", sp_sin_1.offset
    )
    sp_sin_1.offset.user_constraints["sp_line"] = ObjConstraint(
        sp_line.m, "", sp_sin_1.offset
    )
    sp_sin_1.offset.user_constraints["sp_sin2"]()
    sp_sin_1.offset.user_constraints["sp_line"]()

    x1 = np.linspace(0, 5, 200)
    y1 = ref_sin_1(x1)
    x3 = np.copy(x1)
    y3 = ref_sin_2(x3)
    x2 = np.copy(x1)
    y2 = ref_line_obj(x2)

    sp_sin_1.offset.fixed = False
    sp_sin_1.phase.fixed = False
    sp_sin_2.phase.fixed = False
    sp_line.c.fixed = False

    f = MultiFitter([sp_sin_1, sp_line, sp_sin_2], [sp_sin_1, sp_line, sp_sin_2])
    if fit_engine is not None:
        try:
            f.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    args = [[x1, x2, x3], [y1, y2, y3]]
    kwargs = {}
    if with_errors:
        kwargs["weights"] = [1 / np.sqrt(y1), 1 / np.sqrt(y2), 1 / np.sqrt(y3)]

    results = f.fit(*args, **kwargs)
    X = [x1, x2, x3]
    Y = [y1, y2, y3]
    F_ref = [ref_sin_1, ref_line_obj, ref_sin_2]
    F_real = [sp_sin_1, sp_line, sp_sin_2]

    assert len(results) == len(X)

    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin_1.get_fit_parameters()) + len(
            sp_sin_2.get_fit_parameters()
        ) + len(sp_line.get_fit_parameters())
        assert result.chi2 == pytest.approx(
            0, abs=1.5e-3 * (len(result.x) - result.n_pars)
        )
        assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_real[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(
            F_ref[idx](X[idx]) - F_real[idx](X[idx]), abs=1e-2
        )


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_2D_vectorized(fit_engine, with_errors):
    x = np.linspace(0, 5, 200)
    mm = AbsSin2D(0.3, 1.6)
    m2 = AbsSin2D(
        0.1, 1.8
    )  # The fit is quite sensitive to the initial values :-(
    X, Y = np.meshgrid(x, x)
    XY = np.stack((X, Y), axis=2)
    ff = Fitter(m2, m2)
    if fit_engine is not None:
        try:
            ff.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    try:
        args = [XY, mm(XY)]
        kwargs = {"vectorized": True}
        if with_errors:
            kwargs["weights"] = 1 / np.sqrt(args[1])
        result = ff.fit(*args, **kwargs)
    except FitError as e:
        if "Unable to allocate" in str(e):
            pytest.skip(msg="MemoryError - Matrix too large")
        else:
            raise e
    assert result.n_pars == len(m2.get_fit_parameters())
    assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
    assert result.success
    assert np.all(result.x == XY)
    y_calc_ref = m2(XY)
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(mm(XY) - y_calc_ref, abs=1e-2)


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_2D_non_vectorized(fit_engine, with_errors):
    x = np.linspace(0, 5, 200)
    mm = AbsSin2DL(0.3, 1.6)
    m2 = AbsSin2DL(
        0.1, 1.8
    )  # The fit is quite sensitive to the initial values :-(
    X, Y = np.meshgrid(x, x)
    XY = np.stack((X, Y), axis=2)
    ff = Fitter(m2, m2)
    if fit_engine is not None:
        try:
            ff.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    try:
        args = [XY, mm(XY.reshape(-1, 2))]
        kwargs = {"vectorized": False}
        if with_errors:
            kwargs["weights"] = 1 / np.sqrt(args[1])
        result = ff.fit(*args, **kwargs)
    except FitError as e:
        if "Unable to allocate" in str(e):
            pytest.skip(msg="MemoryError - Matrix too large")
        else:
            raise e
    assert result.n_pars == len(m2.get_fit_parameters())
    assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
    assert result.success
    assert np.all(result.x == XY)
    y_calc_ref = m2(XY.reshape(-1, 2))
    assert result.y_calc == pytest.approx(y_calc_ref, abs=1e-2)
    assert result.residual == pytest.approx(
        mm(XY.reshape(-1, 2)) - y_calc_ref, abs=1e-2
    )


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("fit_engine", [None, "lmfit", "bumps", "DFO_LS"])
def test_multi_fit_1D_2D(fit_engine, with_errors):
    # Generate fit and reference objects
    ref_sin1D = AbsSin(0.2, np.pi)
    sp_sin1D = AbsSin(0.354, 3.05)

    ref_sin2D = AbsSin2D(0.3, 1.6)
    sp_sin2D = AbsSin2D(
        0.1, 1.75
    )  # The fit is VERY sensitive to the initial values :-(

    # Link the parameters
    ref_sin1D.offset.user_constraints["ref_sin2"] = ObjConstraint(
        ref_sin2D.offset, "", ref_sin1D.offset
    )
    ref_sin1D.offset.user_constraints["ref_sin2"]()

    sp_sin1D.offset.user_constraints["sp_sin2"] = ObjConstraint(
        sp_sin2D.offset, "", sp_sin1D.offset
    )
    sp_sin1D.offset.user_constraints["sp_sin2"]()

    # Generate data
    x1D = np.linspace(0.2, 3.8, 400)
    y1D = ref_sin1D(x1D)

    x = np.linspace(0, 5, 200)
    X, Y = np.meshgrid(x, x)
    x2D = np.stack((X, Y), axis=2)
    y2D = ref_sin2D(x2D)

    ff = MultiFitter([sp_sin1D, sp_sin2D], [sp_sin1D, sp_sin2D])
    if fit_engine is not None:
        try:
            ff.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")

    sp_sin1D.offset.fixed = False
    sp_sin1D.phase.fixed = False
    sp_sin2D.phase.fixed = False

    f = MultiFitter([sp_sin1D, sp_sin2D], [sp_sin1D, sp_sin2D])
    if fit_engine is not None:
        try:
            f.switch_engine(fit_engine)
        except AttributeError:
            pytest.skip(msg=f"{fit_engine} is not installed")
    try:
        args = [[x1D, x2D], [y1D, y2D]]
        kwargs = {"vectorized": True}
        if with_errors:
            kwargs["weights"] = [1 / np.sqrt(y1D), 1 / np.sqrt(y2D)]
        results = f.fit(*args, **kwargs)
    except FitError as e:
        if "Unable to allocate" in str(e):
            pytest.skip(msg="MemoryError - Matrix too large")
        else:
            raise e

    X = [x1D, x2D]
    Y = [y1D, y2D]
    F_ref = [ref_sin1D, ref_sin2D]
    F_real = [sp_sin1D, sp_sin2D]
    for idx, result in enumerate(results):
        assert result.n_pars == len(sp_sin1D.get_fit_parameters()) + len(
            sp_sin2D.get_fit_parameters()
        )
        assert result.chi2 == pytest.approx(
            0, abs=1.5e-3 * (len(result.x) - result.n_pars)
        )
        assert result.reduced_chi == pytest.approx(0, abs=1.5e-3)
        assert result.success
        assert np.all(result.x == X[idx])
        assert np.all(result.y_obs == Y[idx])
        assert result.y_calc == pytest.approx(F_ref[idx](X[idx]), abs=1e-2)
        assert result.residual == pytest.approx(
            F_real[idx](X[idx]) - F_ref[idx](X[idx]), abs=1e-2
        )
