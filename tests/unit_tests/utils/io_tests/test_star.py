#  SPDX-FileCopyrightText: 2022 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2022 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience>

__author__ = "github.com/wardsimon"
__version__ = "0.0.1"

import pytest

from easyscience.models.polynomial import Line
from easyscience.Objects.Groups import BaseCollection
from easyscience.Objects.Variable import Descriptor
from easyscience.Objects.Variable import Parameter
from easyscience.Utils.io.star import ItemHolder
from easyscience.Utils.io.star import StarLoop
from easyscience.Utils.io.star import StarSection


@pytest.mark.parametrize(
    "value, error, precision, expected",
    (
        (1.234560e05, 1.230000e02, 1, "123500(100)"),
        (1.234567e01, 1.230000e-03, 2, "12.3457(12)"),
        (1.234560e-01, 1.230000e-04, 3, "0.123456(123)"),
        (1.234560e-03, 1.234500e-08, 4, "0.00123456000(1234)"),
        (1.234560e-05, 1.234000e-07, 1, "0.0000123(1)"),
    ),
    ids=[
        "1.234560e+05 +- 1.230000e+02 @1",
        "1.234567e+01 +- 1.230000e-03 @2",
        "1.234560e-01 +- 1.230000e-04 @3",
        "1.234560e-03 +- 1.234500e-08 @4",
        "1.234560e-05 +- 1.234000e-07 @1",
    ],
)
def test_ItemHolder_with_error(value, error, precision, expected):
    p = Parameter("p", value, error=error)
    s = ItemHolder(p, decimal_places=precision)
    assert str(s) == expected


@pytest.mark.parametrize("fixed", (True, False), ids=["fixed", "not fixed"])
@pytest.mark.parametrize(
    "value, precision, expected",
    (
        (1.234560e05, 1, "123456.0"),
        (1.234567e01, 2, "12.35"),
        (1.234560e-01, 3, "0.123"),
        (1.234560e-03, 4, "0.0012"),
        (1.234560e-05, 1, "0.0"),
        (1.234560e-05, 5, "0.00001"),
    ),
    ids=[
        "1.234560e+05 @1",
        "1.234567e+01 @2",
        "1.234560e-01 @3",
        "1.234560e-03 @4",
        "1.234560e-05 @1",
        "1.234560e-05 @5",
    ],
)
def test_ItemHolder_fixed(fixed, value, precision, expected):
    p = Parameter("p", value, fixed=fixed)
    s = ItemHolder(p, decimal_places=precision)
    if not p.fixed:
        expected += "()"
    assert str(s) == expected


@pytest.mark.parametrize("cls", [Descriptor])
def test_ItemHolder_str(cls):
    v = cls("v", "fooooooooo")
    s = ItemHolder(v)
    assert str(s) == "fooooooooo"


def test_StarSection():
    l = Line(2, 3)
    s = StarSection(l)
    expected = "_m   2.00000000()\n_c   3.00000000()\n"
    assert str(s) == expected


def test_StarLoop():
    l1 = Line(2, 3)
    l2 = Line(4, 5)

    ps = BaseCollection("LineCollection", l1, l2)
    s = StarLoop(ps)

    expected = (
        "loop_\n _m\n _c\n  2.00000000()  3.00000000()\n  4.00000000()  5.00000000()"
    )

    assert str(s) == expected
