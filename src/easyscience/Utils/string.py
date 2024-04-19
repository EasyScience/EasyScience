#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from fractions import Fraction

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'


"""
This module provides utility classes for string operations.
"""


def transformation_to_string(matrix, translation_vec=(0, 0, 0), components=('x', 'y', 'z'), c='', delim=','):
    """
    Convenience method. Given matrix returns string, e.g. x+2y+1/4
    :param matrix
    :param translation_vec
    :param components: either ('x', 'y', 'z') or ('a', 'b', 'c')
    :param c: optional additional character to print (used for magmoms)
    :param delim: delimiter
    :return: xyz string
    """
    parts = []
    for i in range(3):
        s = ''
        m = matrix[i]
        t = translation_vec[i]
        for j, dim in enumerate(components):
            if m[j] != 0:
                f = Fraction(m[j]).limit_denominator()
                if s != '' and f >= 0:
                    s += '+'
                if abs(f.numerator) != 1:
                    s += str(f.numerator)
                elif f < 0:
                    s += '-'
                s += c + dim
                if f.denominator != 1:
                    s += '/' + str(f.denominator)
        if t != 0:
            s += ('+' if (t > 0 and s != '') else '') + str(Fraction(t).limit_denominator())
        if s == '':
            s += '0'
        parts.append(s)
    return delim.join(parts)
