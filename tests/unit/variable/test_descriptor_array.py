# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError
from scipp.testing import assert_identical

from easyscience import DescriptorNumber
from easyscience import global_object
from easyscience.variable import DescriptorArray


class TestDescriptorArray:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorArray(
            value=[[1.0, 2.0], [3.0, 4.0]],
            unit='m',
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description='description',
            url='url',
            display_name='display_name',
        )
        return descriptor

    @pytest.fixture
    def descriptor_dimensionless(self):
        descriptor = DescriptorArray(
            value=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            unit='dimensionless',
            variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            description='description',
            url='url',
            display_name='display_name',
        )
        return descriptor

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor._array.values, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert descriptor._array.unit == 'm'
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))

        # From super
        assert descriptor._description == 'description'
        assert descriptor._url == 'url'
        assert descriptor._display_name == 'display_name'

    def test_init_sc_unit(self):
        # When Then
        descriptor = DescriptorArray(
            value=[[1.0, 2.0], [3.0, 4.0]],
            unit=sc.units.Unit('m'),
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description='description',
            url='url',
            display_name='display_name',
        )

        # Expect
        assert np.array_equal(descriptor._array.values, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert descriptor._array.unit == 'm'
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_init_sc_unit_unknown(self):
        # When Then Expect
        with pytest.raises(UnitError):
            DescriptorArray(
                value=[[1.0, 2.0], [3.0, 4.0]],
                unit='unknown',
                variance=[[0.1, 0.2], [0.3, 0.4]],
                description='description',
                url='url',
                display_name='display_name',
            )

    @pytest.mark.parametrize('value', [True, 'string'])
    def test_init_value_type_exception(self, value):
        # When

        # Then Expect
        with pytest.raises(TypeError):
            DescriptorArray(
                value=value,
                unit='m',
                variance=[[0.1, 0.2], [0.3, 0.4]],
                description='description',
                url='url',
                display_name='display_name',
            )

    def test_init_variance_exception(self):
        # When
        variance = [[-0.1, -0.2], [-0.3, -0.4]]
        # Then Expect
        with pytest.raises(ValueError):
            DescriptorArray(
                value=[[1.0, 2.0], [3.0, 4.0]],
                unit='m',
                variance=variance,
                description='description',
                url='url',
                display_name='display_name',
            )

    # test from_scipp
    def test_from_scipp(self):
        # When
        full_value = sc.array(dims=['row', 'column'], values=[[1, 2], [3, 4]], unit='m')
        # Then
        descriptor = DescriptorArray.from_scipp(display_name='name', full_value=full_value)

        # Expect
        assert np.array_equal(descriptor._array.values, [[1, 2], [3, 4]])
        assert descriptor._array.unit == 'm'
        assert descriptor._array.variances == None

    # @pytest.mark.parametrize("full_value", [sc.array(values=[1,2], dimensions=["x"]), sc.array(values=[[1], [2]], dims=["x","y"]), object(), 1, "string"], ids=["1D", "2D", "object", "int", "string"])
    # def test_from_scipp_type_exception(self, full_value):
    #     # When Then Expect
    #     with pytest.raises(TypeError):
    #         DescriptorArray.from_scipp(name="name", full_value=full_value)

    def test_get_full_value(self, descriptor: DescriptorArray):
        # When Then Expect
        other = sc.array(
            dims=('dim0', 'dim1'),
            values=[[1.0, 2.0], [3.0, 4.0]],
            unit='m',
            variances=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert_identical(descriptor.full_value, other)

    def test_set_full_value(self, descriptor: DescriptorArray):
        with pytest.raises(AttributeError):
            descriptor.full_value = sc.array(
                dims=['row', 'column'], values=[[1, 2], [3, 4]], unit='s'
            )

    def test_unit(self, descriptor: DescriptorArray):
        # When Then Expect
        assert descriptor.unit == 'm'

    def test_set_unit(self, descriptor: DescriptorArray):
        with pytest.raises(AttributeError):
            descriptor.unit = 's'

    def test_convert_unit(self, descriptor: DescriptorArray):
        # When  Then
        descriptor.convert_unit('mm')

        # Expect
        assert descriptor._array.unit == 'mm'
        assert np.array_equal(descriptor._array.values, [[1000, 2000], [3000, 4000]])
        assert np.array_equal(descriptor._array.variances, [[100000, 200000], [300000, 400000]])

    def test_variance(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_set_variance(self, descriptor: DescriptorArray):
        # When Then
        descriptor.variance = [[0.2, 0.3], [0.4, 0.5]]

        # Expect
        assert np.array_equal(descriptor.variance, np.array([[0.2, 0.3], [0.4, 0.5]]))
        assert np.array_equal(descriptor.error, np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]])))

    def test_error(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor.error, np.sqrt(np.array([[0.1, 0.2], [0.3, 0.4]])))

    def test_set_error(self, descriptor: DescriptorArray):
        # When Then
        descriptor.error = np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]]))
        # Expect
        assert np.allclose(descriptor.error, np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]])))
        assert np.allclose(descriptor.variance, np.array([[0.2, 0.3], [0.4, 0.5]]))

    def test_value(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor.value, np.array([[1, 2], [3, 4]]))

    def test_set_value(self, descriptor: DescriptorArray):
        # When Then
        descriptor.value = [[0.2, 0.3], [0.4, 0.5]]
        # Expect
        assert np.array_equal(descriptor._array.values, np.array([[0.2, 0.3], [0.4, 0.5]]))

    def test_repr(self, descriptor: DescriptorArray):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert (
            repr_str
            == "<DescriptorArray 'display_name': values=[[1. 2.], [3. 4.]], errors=[[0.3162 0.4472], [0.5477 0.6325]], unit=m>"
        )

    def test_copy(self, descriptor: DescriptorArray):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorArray
        assert np.array_equal(descriptor_copy._array.values, descriptor._array.values)
        assert descriptor_copy._array.unit == descriptor._array.unit

    @pytest.mark.parametrize(
        'unit_string, expected',
        [('1e+9', 'dimensionless'), ('1000', 'dimensionless'), ('10dm^2', 'm^2')],
        ids=['scientific_notation', 'numbers', 'unit_prefix'],
    )
    def test_base_unit(self, unit_string, expected):
        # When
        descriptor = DescriptorArray(
            display_name='name', value=[[1.0, 2.0], [3.0, 4.0]], unit=unit_string
        )

        # Then
        base_unit = descriptor._base_unit()

        # Expect
        assert base_unit == expected

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[3.0, 4.0], [5.0, 6.0]],
                    unit='m',
                    variance=[[0.11, 0.21], [0.31, 0.41]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[1.01, 2.01], [3.01, 4.01]],
                    unit='m',
                    variance=[[0.1010, 0.2010], [0.3010, 0.4010]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[1.02, 2.03], [3.04, 3.95]],
                    unit='m',
                    variance=[[0.1001, 0.2002], [0.3003, 0.4004]],
                    display_name='test + name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm', display_name='test'),
                DescriptorArray(
                    [[1.02, 2.03], [3.04, 3.95]],
                    unit='m',
                    variance=[[0.1, 0.2], [0.3, 0.4]],
                    display_name='test + name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'array_conversion',
            'array_conversion_integer',
        ],
    )
    def test_addition(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = descriptor + test
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = descriptor + test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[3.0, 5.0], [7.0, -1.0], [11.0, -2.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
            (
                1,
                DescriptorArray(
                    [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_addition_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = descriptor_dimensionless + test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[3.0, 4.0], [5.0, 6.0]],
                    unit='m',
                    variance=[[0.11, 0.21], [0.31, 0.41]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[101.0, 201.0], [301.0, 401.0]],
                    unit='cm',
                    variance=[[1010.0, 2010.0], [3010.0, 4010.0]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[102.0, 203.0], [304.0, 395.0]],
                    unit='cm',
                    variance=[[1001.0, 2002.0], [3003.0, 4004.0]],
                    display_name='test + name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm', display_name='test'),
                DescriptorArray(
                    [[102.0, 203.0], [304.0, 395.0]],
                    unit='cm',
                    variance=[[1000.0, 2000.0], [3000.0, 4000.0]],
                    display_name='test + name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'array_conversion',
            'array_conversion_integer',
        ],
    )
    def test_reverse_addition(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test + descriptor
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test + descriptor
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[3.0, 5.0], [7.0, -1.0], [11.0, -2.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
            (
                1,
                DescriptorArray(
                    [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_reverse_addition_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = test + descriptor_dimensionless
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[-1.0, 0.0], [1.0, 2.0]],
                    unit='m',
                    variance=[[0.11, 0.21], [0.31, 0.41]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[0.99, 1.99], [2.99, 3.99]],
                    unit='m',
                    variance=[[0.1010, 0.2010], [0.3010, 0.4010]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[0.98, 1.97], [2.96, 4.05]],
                    unit='m',
                    variance=[[0.1001, 0.2002], [0.3003, 0.4004]],
                    display_name='test + name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm', display_name='test'),
                DescriptorArray(
                    [[0.98, 1.97], [2.96, 4.05]],
                    unit='m',
                    variance=[[0.100, 0.200], [0.300, 0.400]],
                    display_name='test + name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'array_conversion',
            'array_conversion_integer',
        ],
    )
    def test_subtraction(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = descriptor - test
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = descriptor - test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[-1.0, -1.0], [-1.0, 9.0], [-1, 14.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
            (
                1,
                DescriptorArray(
                    [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_subtraction_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = descriptor_dimensionless - test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[1.0, 0.0], [-1.0, -2.0]],
                    unit='m',
                    variance=[[0.11, 0.21], [0.31, 0.41]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[-99.0, -199.0], [-299.0, -399.0]],
                    unit='cm',
                    variance=[[1010.0, 2010.0], [3010.0, 4010.0]],
                    display_name='test + name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[-98.0, -197.0], [-296.0, -405.0]],
                    unit='cm',
                    variance=[[1001.0, 2002.0], [3003.0, 4004.0]],
                    display_name='test + name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm', display_name='test'),
                DescriptorArray(
                    [[-98.0, -197.0], [-296.0, -405.0]],
                    unit='cm',
                    variance=[[1000.0, 2000.0], [3000.0, 4000.0]],
                    display_name='test + name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'array_conversion',
            'array_conversion_integer',
        ],
    )
    def test_reverse_subtraction(
        self, descriptor: DescriptorArray, test, expected, raises_warning
    ):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test - descriptor
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test - descriptor
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[1.0, 1.0], [1.0, -9.0], [1.0, -14.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
            (
                1,
                DescriptorArray(
                    [[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0]],
                    unit='dimensionless',
                    variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_reverse_subtraction_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = test - descriptor_dimensionless
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[2.0, 4.0], [6.0, 8.0]],
                    unit='m^2',
                    variance=[[0.41, 0.84], [1.29, 1.76]],
                    display_name='test * name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[0.01, 0.02], [0.03, 0.04]],
                    unit='m^2',
                    variance=[[0.00101, 0.00402], [0.00903, 0.01604]],
                    display_name='test * name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='kg', variance=10, display_name='test'),
                DescriptorArray(
                    [[1.0, 2.0], [3.0, 4.0]],
                    unit='kg*m',
                    variance=[[10.1, 40.2], [90.3, 160.4]],
                    display_name='test * name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[0.02, 0.06], [0.12, -0.2]],
                    unit='m^2',
                    variance=[[0.00014, 0.00098], [0.00318, 0.0074]],
                    display_name='test * name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm', display_name='test'),
                DescriptorArray(
                    [[0.02, 0.06], [0.12, -0.2]],
                    unit='m^2',
                    variance=[
                        [0.1 * 2**2 * 1e-4, 0.2 * 3**2 * 1e-4],
                        [0.3 * 4**2 * 1e-4, 0.4 * 5**2 * 1e-4],
                    ],
                    display_name='test * name',
                ),
                False,
            ),
            (
                [[2.0, 3.0], [4.0, -5.0]],
                DescriptorArray(
                    [[2.0, 6.0], [12.0, -20.0]],
                    unit='m',
                    variance=[[0.1 * 2**2, 0.2 * 3**2], [0.3 * 4**2, 0.4 * 5**2]],
                    display_name='test * name',
                ),
                False,
            ),
            (
                2.0,
                DescriptorArray(
                    [[2.0, 4.0], [6.0, 8.0]],
                    unit='m',
                    variance=[[0.1 * 2**2, 0.2 * 2**2], [0.3 * 2**2, 0.4 * 2**2]],
                    display_name='test * name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'descriptor_number_different_units',
            'array_conversion',
            'array_conversion_integer',
            'list',
            'number',
        ],
    )
    def test_multiplication(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = descriptor * test
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = descriptor * test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[2.0, 6.0], [12.0, -20.0], [30.0, -48.0]],
                    unit='dimensionless',
                    variance=[[0.4, 1.8], [4.8, 10.0], [18.0, 38.4]],
                    display_name='test',
                ),
            ),
            (
                1.5,
                DescriptorArray(
                    [[1.5, 3.0], [4.5, 6.0], [7.5, 9.0]],
                    unit='dimensionless',
                    variance=[[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_multiplication_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = descriptor_dimensionless * test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[2.0, 4.0], [6.0, 8.0]],
                    unit='m^2',
                    variance=[[0.41, 0.84], [1.29, 1.76]],
                    display_name='test * name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[100.0, 200.0], [300.0, 400.0]],
                    unit='cm^2',
                    variance=[[101000.0, 402000.0], [903000.0, 1604000.0]],
                    display_name='test * name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='kg', variance=10, display_name='test'),
                DescriptorArray(
                    [[1.0, 2.0], [3.0, 4.0]],
                    unit='kg*m',
                    variance=[[10.1, 40.2], [90.3, 160.4]],
                    display_name='test * name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[200.0, 600.0], [1200.0, -2000.0]],
                    unit='cm^2',
                    variance=[[14000.0, 98000.0], [318000.0, 740000.0]],
                    display_name='test * name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm', display_name='test'),
                DescriptorArray(
                    [[200.0, 600.0], [1200.0, -2000.0]],
                    unit='cm^2',
                    variance=[
                        [0.1 * 2**2 * 1e4, 0.2 * 3**2 * 1e4],
                        [0.3 * 4**2 * 1e4, 0.4 * 5**2 * 1e4],
                    ],
                    display_name='test * name',
                ),
                False,
            ),
            (
                [[2.0, 3.0], [4.0, -5.0]],
                DescriptorArray(
                    [[2.0, 6.0], [12.0, -20.0]],
                    unit='m',
                    variance=[[0.1 * 2**2, 0.2 * 3**2], [0.3 * 4**2, 0.4 * 5**2]],
                    display_name='test * name',
                ),
                False,
            ),
            (
                2.0,
                DescriptorArray(
                    [[2.0, 4.0], [6.0, 8.0]],
                    unit='m',
                    variance=[[0.1 * 2**2, 0.2 * 2**2], [0.3 * 2**2, 0.4 * 2**2]],
                    display_name='test * name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'descriptor_number_different_units',
            'array_conversion',
            'array_conversion_integer',
            'list',
            'number',
        ],
    )
    def test_reverse_multiplication(
        self, descriptor: DescriptorArray, test, expected, raises_warning
    ):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test * descriptor
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test * descriptor
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[2.0, 6.0], [12.0, -20.0], [30.0, -48.0]],
                    unit='dimensionless',
                    variance=[[0.4, 1.8], [4.8, 10.0], [18.0, 38.4]],
                    display_name='test',
                ),
            ),
            (
                1.5,
                DescriptorArray(
                    [[1.5, 3.0], [4.5, 6.0], [7.5, 9.0]],
                    unit='dimensionless',
                    variance=[[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_reverse_multiplication_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = test * descriptor_dimensionless
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[1.0 / 2.0, 2.0 / 2.0], [3.0 / 2.0, 4.0 / 2.0]],
                    unit='dimensionless',
                    variance=[
                        [
                            (0.1 + 0.01 * 1.0**2 / 2.0**2) / 2.0**2,
                            (0.2 + 0.01 * 2.0**2 / 2.0**2) / 2.0**2,
                        ],
                        [
                            (0.3 + 0.01 * 3.0**2 / 2.0**2) / 2.0**2,
                            (0.4 + 0.01 * 4.0**2 / 2.0**2) / 2.0**2,
                        ],
                    ],
                    display_name='name / test',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[100.0, 200.0], [300.0, 400.0]],
                    unit='dimensionless',
                    variance=[
                        [
                            (0.1 + 10 * 1.0**2 / 1.0**2) / 1.0**2 * 1e4,
                            (0.2 + 10 * 2.0**2 / 1.0**2) / 1.0**2 * 1e4,
                        ],
                        [
                            (0.3 + 10 * 3.0**2 / 1.0**2) / 1.0**2 * 1e4,
                            (0.4 + 10 * 4.0**2 / 1.0**2) / 1.0**2 * 1e4,
                        ],
                    ],
                    display_name='name / test',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='kg', variance=10, display_name='test'),
                DescriptorArray(
                    [[1.0, 2.0], [3.0, 4.0]],
                    unit='m/kg',
                    variance=[
                        [
                            (0.1 + 10 * 1.0**2 / 1.0**2) / 1.0**2,
                            (0.2 + 10 * 2.0**2 / 1.0**2) / 1.0**2,
                        ],
                        [
                            (0.3 + 10 * 3.0**2 / 1.0**2) / 1.0**2,
                            (0.4 + 10 * 4.0**2 / 1.0**2) / 1.0**2,
                        ],
                    ],
                    display_name='name / test',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm^2',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[1 / 2 * 1e4, 2 / 3 * 1e4], [3.0 / 4.0 * 1e4, -4.0 / 5.0 * 1e4]],
                    unit='1/m',
                    variance=[
                        [
                            (0.1 + 1.0 * 1.0**2 / 2.0**2) / 2.0**2 * 1e8,
                            (0.2 + 2.0 * 2.0**2 / 3.0**2) / 3.0**2 * 1e8,
                        ],
                        [
                            (0.3 + 3.0 * 3.0**2 / 4.0**2) / 4.0**2 * 1e8,
                            (0.4 + 4.0 * 4.0**2 / 5.0**2) / 5.0**2 * 1e8,
                        ],
                    ],
                    display_name='name / test',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm^2', display_name='test'),
                DescriptorArray(
                    [[1 / 2 * 1e4, 2 / 3 * 1e4], [3.0 / 4.0 * 1e4, -4.0 / 5.0 * 1e4]],
                    unit='1/m',
                    variance=[
                        [(0.1) / 2.0**2 * 1e8, (0.2) / 3.0**2 * 1e8],
                        [(0.3) / 4.0**2 * 1e8, (0.4) / 5.0**2 * 1e8],
                    ],
                    display_name='name / test',
                ),
                False,
            ),
            (
                [[2.0, 3.0], [4.0, -5.0]],
                DescriptorArray(
                    [[0.5, 2.0 / 3.0], [3.0 / 4.0, -4 / 5]],
                    unit='m',
                    variance=[[0.1 / 2**2, 0.2 / 3.0**2], [0.3 / 4**2, 0.4 / 5.0**2]],
                    display_name='name / name',
                ),
                False,
            ),
            (
                2.0,
                DescriptorArray(
                    [[0.5, 1.0], [3.0 / 2.0, 2.0]],
                    unit='m',
                    variance=[[0.1 / 2.0**2, 0.2 / 2.0**2], [0.3 / 2.0**2, 0.4 / 2.0**2]],
                    display_name='name / test',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'descriptor_number_different_units',
            'array_conversion',
            'array_conversion_integer',
            'list',
            'number',
        ],
    )
    def test_division(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = descriptor / test
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = descriptor / test
        # Expect
        assert type(result) == DescriptorArray
        assert np.allclose(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[1.0 / 2.0, 2.0 / 3.0], [3.0 / 4.0, -4.0 / 5.0], [5.0 / 6.0, -6.0 / 8.0]],
                    unit='dimensionless',
                    variance=[
                        [0.1 / 2.0**2, 0.2 / 3.0**2],
                        [0.3 / 4.0**2, 0.4 / 5.0**2],
                        [0.5 / 6.0**2, 0.6 / 8.0**2],
                    ],
                    display_name='test',
                ),
            ),
            (
                2,
                DescriptorArray(
                    [[1.0 / 2.0, 2.0 / 2.0], [3.0 / 2.0, 4.0 / 2.0], [5.0 / 2.0, 6.0 / 2.0]],
                    unit='dimensionless',
                    variance=[
                        [0.1 / 2.0**2, 0.2 / 2.0**2],
                        [0.3 / 2.0**2, 0.4 / 2.0**2],
                        [0.5 / 2.0**2, 0.6 / 2.0**2],
                    ],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_division_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = descriptor_dimensionless / test
        # Expect
        assert type(result) == DescriptorArray
        assert np.allclose(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, expected, raises_warning',
        [
            (
                DescriptorNumber(2, unit='m', variance=0.01, display_name='test'),
                DescriptorArray(
                    [[2.0, 1.0], [2.0 / 3.0, 0.5]],
                    unit='dimensionless',
                    variance=[
                        [0.41, 0.0525],
                        [
                            (0.01 + 0.3 * 2**2 / 3.0**2) / 3.0**2,
                            (0.01 + 0.4 * 2**2 / 4.0**2) / 4.0**2,
                        ],
                    ],
                    display_name='test / name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='cm', variance=10, display_name='test'),
                DescriptorArray(
                    [[1.0 / 100.0, 1.0 / 200.0], [1.0 / 300.0, 1.0 / 400.0]],
                    unit='dimensionless',
                    variance=[
                        [1.01e-3, (1e-3 + 0.2 * 0.01**2 / 2**2) / 2**2],
                        [
                            (1e-3 + 0.3 * 0.01**2 / 3**2) / 3**2,
                            (1e-3 + 0.4 * 0.01**2 / 4**2) / 4**2,
                        ],
                    ],
                    display_name='test / name',
                ),
                True,
            ),
            (
                DescriptorNumber(1, unit='kg', variance=10, display_name='test'),
                DescriptorArray(
                    [[1.0, 0.5], [1.0 / 3.0, 0.25]],
                    unit='kg/m',
                    variance=[
                        [10.1, (10 + 0.2 * 1 / 2**2) / 2**2],
                        [(10 + 0.3 * 1 / 3**2) / 3**2, (10 + 0.4 * 1 / 4**2) / 4**2],
                    ],
                    display_name='test / name',
                ),
                True,
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0], [4.0, -5.0]],
                    unit='cm^2',
                    variance=[[1.0, 2.0], [3.0, 4.0]],
                    display_name='test',
                ),
                DescriptorArray(
                    [[2e-4, 1.5e-4], [4.0 / 3.0 * 1e-4, -1.25e-4]],
                    unit='m',
                    variance=[
                        [1.4e-8, 6.125e-9],
                        [
                            (3.0e-8 + 0.3 * (0.0004) ** 2 / 3**2) / 3**2,
                            (4.0e-8 + 0.4 * (0.0005) ** 2 / 4**2) / 4**2,
                        ],
                    ],
                    display_name='test / name',
                ),
                False,
            ),
            (
                DescriptorArray([[2, 3], [4, -5]], unit='cm^2', display_name='test'),
                DescriptorArray(
                    [[2e-4, 1.5e-4], [4.0 / 3.0 * 1e-4, -1.25e-4]],
                    unit='m',
                    variance=[
                        [
                            (0.1 * 2.0**2 / 1.0**2) / 1.0**2 * 1e-8,
                            (0.2 * 3.0**2 / 2.0**2) / 2.0**2 * 1e-8,
                        ],
                        [
                            (0.3 * 4.0**2 / 3.0**2) / 3.0**2 * 1e-8,
                            (0.4 * 5.0**2 / 4.0**2) / 4.0**2 * 1e-8,
                        ],
                    ],
                    display_name='test / name',
                ),
                False,
            ),
            (
                [[2.0, 3.0], [4.0, -5.0]],
                DescriptorArray(
                    [[2, 1.5], [4.0 / 3.0, -1.25]],
                    unit='1/m',
                    variance=[
                        [0.1 * 2**2 / 1**4, 0.2 * 3.0**2 / 2.0**4],
                        [0.3 * 4**2 / 3**4, 0.4 * 5.0**2 / 4.0**4],
                    ],
                    display_name='test / name',
                ),
                False,
            ),
            (
                2.0,
                DescriptorArray(
                    [[2, 1.0], [2.0 / 3.0, 0.5]],
                    unit='1/m',
                    variance=[
                        [0.1 * 2**2 / 1**4, 0.2 * 2.0**2 / 2.0**4],
                        [0.3 * 2**2 / 3**4, 0.4 * 2.0**2 / 4.0**4],
                    ],
                    display_name='test / name',
                ),
                False,
            ),
        ],
        ids=[
            'descriptor_number_regular',
            'descriptor_number_unit_conversion',
            'descriptor_number_different_units',
            'array_conversion',
            'array_conversion_integer',
            'list',
            'number',
        ],
    )
    def test_reverse_division(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test / descriptor
            assert len(record) == 1
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test / descriptor
        # Expect
        assert type(result) == DescriptorArray
        assert np.allclose(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                [[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]],
                DescriptorArray(
                    [[2.0 / 1.0, 3.0 / 2.0], [4.0 / 3.0, -5.0 / 4.0], [6.0 / 5.0, -8.0 / 6.0]],
                    unit='dimensionless',
                    variance=[
                        [0.1 * 2.0**2, 0.2 * 3.0**2 / 2**4],
                        [0.3 * 4.0**2 / 3.0**4, 0.4 * 5.0**2 / 4**4],
                        [0.5 * 6.0**2 / 5**4, 0.6 * 8.0**2 / 6**4],
                    ],
                    display_name='test',
                ),
            ),
            (
                2,
                DescriptorArray(
                    [[2.0, 1.0], [2.0 / 3.0, 0.5], [2.0 / 5.0, 1.0 / 3.0]],
                    unit='dimensionless',
                    variance=[
                        [0.1 * 2.0**2, 0.2 / 2**2],
                        [0.3 * 2**2 / 3**4, 0.4 * 2**2 / 4**4],
                        [0.5 * 2**2 / 5**4, 0.6 * 2**2 / 6**4],
                    ],
                    display_name='test',
                ),
            ),
        ],
        ids=['list', 'number'],
    )
    def test_reverse_division_dimensionless(
        self, descriptor_dimensionless: DescriptorArray, test, expected
    ):
        # When Then
        result = test / descriptor_dimensionless
        # Expect
        assert type(result) == DescriptorArray
        assert np.allclose(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test',
        [
            [[2.0, 3.0], [4.0, -5.0], [6.0, 0.0]],
            0.0,
            DescriptorNumber(0, unit='cm', variance=10, display_name='test'),
            DescriptorArray(
                [[1.5, 0.0], [4.5, 6.0], [7.5, 9.0]],
                unit='dimensionless',
                variance=[[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]],
                display_name='test',
            ),
        ],
        ids=['list', 'number', 'DescriptorNumber', 'DescriptorArray'],
    )
    def test_division_exception(self, descriptor_dimensionless: DescriptorArray, test):
        # When Then
        with pytest.raises(ZeroDivisionError):
            descriptor_dimensionless / test

        # Also test reverse division where `self` is a DescriptorArray with a zero
        zero_descriptor = DescriptorArray(
            [[1.5, 0.0], [4.5, 6.0], [7.5, 0.0]],
            unit='dimensionless',
            variance=[[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]],
            display_name='test',
        )
        with pytest.raises(ZeroDivisionError):
            test / zero_descriptor

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                DescriptorNumber(2, unit='dimensionless', display_name='test'),
                DescriptorArray(
                    [[1.0, 4.0], [9.0, 16.0]],
                    unit='m^2',
                    variance=[[4 * 0.1 * 1, 4 * 0.2 * 2**2], [4 * 0.3 * 3**2, 4 * 0.4 * 4**2]],
                    display_name='test ** name',
                ),
            ),
            (
                DescriptorNumber(3, unit='dimensionless', display_name='test'),
                DescriptorArray(
                    [[1.0, 8.0], [27, 64.0]],
                    unit='m^3',
                    variance=[[9 * 0.1, 9 * 0.2 * 2**4], [9 * 0.3 * 3**4, 9 * 0.4 * 4**4]],
                    display_name='test ** name',
                ),
            ),
            (
                DescriptorNumber(0.0, unit='dimensionless', display_name='test'),
                DescriptorArray(
                    [[1.0, 1.0], [1.0, 1.0]],
                    unit='dimensionless',
                    variance=[[0.0, 0.0], [0.0, 0.0]],
                    display_name='test ** name',
                ),
            ),
            (
                0.0,
                DescriptorArray(
                    [[1.0, 1.0], [1.0, 1.0]],
                    unit='dimensionless',
                    variance=[[0.0, 0.0], [0.0, 0.0]],
                    display_name='test ** name',
                ),
            ),
        ],
        ids=[
            'descriptor_number_squared',
            'descriptor_number_cubed',
            'descriptor_number_zero',
            'number_zero',
        ],
    )
    def test_power(self, descriptor: DescriptorArray, test, expected):
        # When Then
        result = descriptor**test
        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                DescriptorNumber(0.1, unit='dimensionless', display_name='test'),
                DescriptorArray(
                    [[1, 2**0.1], [3**0.1, 4**0.1], [5**0.1, 6**0.1]],
                    unit='dimensionless',
                    variance=[
                        [0.1**2 * 0.1 * 1, 0.1**2 * 0.2 * 2 ** (-1.8)],
                        [0.1**2 * 0.3 * 3 ** (-1.8), 0.1**2 * 0.4 * 4 ** (-1.8)],
                        [0.1**2 * 0.5 * 5 ** (-1.8), 0.1**2 * 0.6 * 6 ** (-1.8)],
                    ],
                    display_name='test ** name',
                ),
            ),
            (
                DescriptorNumber(2.0, unit='dimensionless', display_name='test'),
                DescriptorArray(
                    [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]],
                    unit='dimensionless',
                    variance=[[0.4, 3.2], [10.8, 25.6], [50.0, 86.4]],
                    display_name='test ** name',
                ),
            ),
        ],
        ids=['descriptor_number_fractional', 'descriptor_number_integer'],
    )
    def test_power_dimensionless(self, descriptor_dimensionless: DescriptorArray, test, expected):
        # When Then
        result = descriptor_dimensionless**test
        # Expect
        assert type(result) == DescriptorArray
        assert np.allclose(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'test, exception',
        [
            (DescriptorNumber(2, unit='m', display_name='test'), UnitError),
            (
                DescriptorNumber(2, unit='dimensionless', variance=10, display_name='test'),
                ValueError,
            ),
            (DescriptorNumber(np.nan, unit='dimensionless', display_name='test'), UnitError),
            (DescriptorNumber(np.nan, unit='dimensionless', display_name='test'), UnitError),
            (DescriptorNumber(1.5, unit='dimensionless', display_name='test'), UnitError),
            (
                DescriptorNumber(0.5, unit='dimensionless', display_name='test'),
                UnitError,
            ),  # Square roots are not legal
        ],
        ids=[
            'units',
            'variance',
            'scipp_nan',
            'nan_result',
            'non_integer_exponent_on_units',
            'square_root_on_units',
        ],
    )
    def test_power_exception(self, descriptor: DescriptorArray, test, exception):
        # When Then
        with pytest.raises(exception):
            result = descriptor**2**test
        with pytest.raises(ValueError):
            # Exponentiation with an array does not make sense
            test**descriptor

    @pytest.mark.parametrize(
        'test',
        [
            DescriptorNumber(2, unit='s', display_name='test'),
            DescriptorArray([[1, 2], [3, 4]], unit='s', display_name='test'),
        ],
        ids=['add_array_to_unit', 'incompatible_units'],
    )
    def test_addition_exception(self, descriptor: DescriptorArray, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = descriptor + test
        with pytest.raises(UnitError):
            result_reverse = test + descriptor

    @pytest.mark.parametrize(
        'test',
        [
            DescriptorNumber(2, unit='s', display_name='test'),
            DescriptorArray([[1, 2], [3, 4]], unit='s', display_name='test'),
        ],
        ids=['add_array_to_unit', 'incompatible_units'],
    )
    def test_sub_exception(self, descriptor: DescriptorArray, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = descriptor - test
        with pytest.raises(UnitError):
            result_reverse = test - descriptor

    @pytest.mark.parametrize(
        'function',
        [np.sin, np.cos, np.exp, np.add, np.multiply],
        ids=['sin', 'cos', 'exp', 'add', 'multiply'],
    )
    def test_numpy_ufuncs_exception(self, descriptor_dimensionless, function):
        ((np.add, np.array([[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]])),)
        """
        Not implemented ufuncs should return NotImplemented.
        """
        test = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError) as e:
            function(descriptor_dimensionless, test)
        assert 'returned NotImplemented from' in str(e)

    def test_negation(self, descriptor):
        # When
        # Then
        result = -descriptor

        # Expect
        expected = DescriptorArray(
            value=[[-1.0, -2.0], [-3.0, -4.0]],
            unit='m',
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description='description',
            url='url',
            display_name='display_name',
        )
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    def test_abs(self, descriptor):
        # When
        negated = DescriptorArray(
            value=[[-1.0, -2.0], [-3.0, -4.0]],
            unit='m',
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description='description',
            url='url',
            display_name='display_name',
        )

        # Then
        result = abs(negated)

        # Expect
        assert type(result) == DescriptorArray
        assert np.array_equal(result.value, descriptor.value)
        assert result.unit == descriptor.unit
        assert np.allclose(result.variance, descriptor.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                DescriptorArray(
                    [[3.0, 4.0], [5.0, 6.0]],
                    unit='m',
                    variance=[[0.11, 0.21], [0.31, 0.41]],
                    display_name='test + name',
                ),
                DescriptorNumber(9, unit='m', variance=0.52, display_name='test'),
            ),
            (
                DescriptorArray(
                    [[101.0, 201.0], [301.0, 401.0]],
                    unit='dimensionless',
                    variance=[[1010.0, 2010.0], [3010.0, 4010.0]],
                    display_name='test + name',
                ),
                DescriptorNumber(
                    502.0, unit='dimensionless', variance=5020.0, display_name='test'
                ),
            ),
            (
                DescriptorArray(
                    np.ones((9, 9)),
                    unit='dimensionless',
                    variance=np.ones((9, 9)),
                    display_name='test',
                ),
                DescriptorNumber(9.0, unit='dimensionless', variance=9.0, display_name='test'),
            ),
            (
                DescriptorArray(
                    np.ones((3, 3, 3)),
                    unit='dimensionless',
                    variance=np.ones((3, 3, 3)),
                    display_name='test',
                ),
                DescriptorArray(
                    [3.0, 3.0, 3.0],
                    unit='dimensionless',
                    variance=[
                        3.0,
                        3.0,
                        3.0,
                    ],
                    dimensions=['dim2'],
                    display_name='test',
                ),
            ),
            (
                DescriptorArray([[2.0]], unit='dimensionless', display_name='test'),
                DescriptorNumber(2.0, unit='dimensionless', display_name='test'),
            ),
        ],
        ids=['2d_unit', '2d_dimensionless', '2d_large', '3d_dimensionless', '1d_dimensionless'],
    )
    def test_trace(self, test: DescriptorArray, expected: DescriptorNumber):
        result = test.trace()
        assert type(result) == type(expected)
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        if test.variance is not None:
            assert np.allclose(result.variance, expected.variance)
        if isinstance(expected, DescriptorArray):
            assert np.all(result.full_value.dims == expected.full_value.dims)

    @pytest.mark.parametrize(
        'test, expected, dimensions',
        [
            (
                DescriptorArray(
                    np.ones((3, 3, 4, 5)),
                    unit='dimensionless',
                    variance=np.ones((3, 3, 4, 5)),
                    display_name='test',
                ),
                DescriptorArray(
                    3 * np.ones((3, 4)),
                    unit='dimensionless',
                    variance=3 * np.ones((3, 4)),
                    dimensions=['dim0', 'dim2'],
                    display_name='test',
                ),
                ('dim1', 'dim3'),
            )
        ],
        ids=['4d'],
    )
    def test_trace_select_dimensions(
        self, test: DescriptorArray, expected: DescriptorNumber, dimensions
    ):
        result = test.trace(dimension1=dimensions[0], dimension2=dimensions[1])
        assert type(result) == type(expected)
        assert np.array_equal(result.value.shape, expected.value.shape)
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.all(result.full_value.dims == expected.full_value.dims)

    @pytest.mark.parametrize(
        'test,dimensions,message',
        [
            (
                DescriptorArray(
                    np.ones((3, 3, 3)),
                    unit='dimensionless',
                    variance=np.ones((3, 3, 3)),
                    display_name='test',
                ),
                ('dim0', None),
                'Either both or none',
            ),
            (
                DescriptorArray(
                    np.ones((3, 3, 3)),
                    unit='dimensionless',
                    variance=np.ones((3, 3, 3)),
                    display_name='test',
                ),
                ('dim0', 'dim0'),
                'must be different',
            ),
            (
                DescriptorArray(
                    np.ones((3, 3, 3)),
                    unit='dimensionless',
                    variance=np.ones((3, 3, 3)),
                    display_name='test',
                ),
                ('dim0', 'dim1337'),
                'does not exist',
            ),
        ],
        ids=['one_defined_dimension', 'same_dimension', 'invalid_dimension'],
    )
    def test_trace_exception(self, test: DescriptorArray, dimensions, message):
        with pytest.raises(ValueError) as e:
            test.trace(dimension1=dimensions[0], dimension2=dimensions[1])
        assert message in str(e)

    def test_slicing(self, descriptor: DescriptorArray):
        # When
        first_value = descriptor['dim0', 0]
        last_value = descriptor['dim0', -1]
        second_array = descriptor['dim1', :]

        # Then
        assert type(first_value) == DescriptorArray
        assert type(last_value) == DescriptorArray
        assert type(second_array) == DescriptorArray

        # Slices are fresh objects with their own identity, not the sliced parent.
        assert first_value.unique_name != descriptor.unique_name
        assert last_value.unique_name != descriptor.unique_name
        assert second_array.unique_name != descriptor.unique_name

        assert np.array_equal(
            first_value.full_value.values, descriptor.full_value['dim0', 0].values
        )
        assert np.array_equal(
            last_value.full_value.values, descriptor.full_value['dim0', -1].values
        )
        assert np.array_equal(
            second_array.full_value.values, descriptor.full_value['dim1', :].values
        )

        assert np.array_equal(
            first_value.full_value.variances, descriptor.full_value['dim0', 0].variances
        )
        assert np.array_equal(
            last_value.full_value.variances, descriptor.full_value['dim0', -1].variances
        )
        assert np.array_equal(
            second_array.full_value.variances, descriptor.full_value['dim1', :].variances
        )

        assert np.array_equal(first_value.full_value.unit, descriptor.unit)
        assert np.array_equal(last_value.full_value.unit, descriptor.unit)
        assert np.array_equal(second_array.full_value.unit, descriptor.unit)

    def test_slice_deletion(self, descriptor: DescriptorArray):
        with pytest.raises(AttributeError) as e:
            del descriptor['dim0', 0]
        assert 'has no attribute' in str(e)

    @pytest.mark.parametrize('test', [1.0, [3.0, 4.0, 5.0]], ids=['number', 'list'])
    def test_slice_assignment_exception(self, descriptor_dimensionless: DescriptorArray, test):
        # When
        with pytest.raises(AttributeError) as e:
            descriptor_dimensionless['dim0', :] = test
        assert 'cannot be edited via slicing' in str(e)

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                DescriptorArray(
                    [[3.0, 4.0], [5.0, 6.0]],
                    unit='m',
                    variance=[[0.11, 0.21], [0.31, 0.41]],
                    display_name='test + name',
                ),
                DescriptorNumber(18, unit='m', variance=1.04, display_name='test'),
            ),
            (
                DescriptorArray(
                    [[101.0, 201.0], [301.0, 401.0]],
                    unit='cm',
                    variance=[[1010.0, 2010.0], [3010.0, 4010.0]],
                    display_name='test + name',
                ),
                DescriptorNumber(1004.0, unit='cm', variance=10040.0, display_name='test'),
            ),
            (
                DescriptorArray(
                    [[2.0, 3.0]], unit='dimensionless', variance=[[1.0, 2.0]], display_name='test'
                ),
                DescriptorNumber(5.0, unit='dimensionless', variance=3.0, display_name='test'),
            ),
            (
                DescriptorArray([[2.0, 3.0]], unit='dimensionless', display_name='test'),
                DescriptorNumber(5.0, unit='dimensionless', display_name='test'),
            ),
        ],
        ids=[
            'descriptor_array_m',
            'd=descriptor_array_cm',
            'descriptor_array_dimensionless',
            'descriptor_array_dim_varless',
        ],
    )
    def test_sum(self, test, expected):
        result = test.sum()
        assert type(result) == DescriptorNumber
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        if test.variance is not None:
            assert np.allclose(result.variance, expected.variance)

    @pytest.mark.parametrize(
        'expected, dim',
        [
            (
                DescriptorArray([4.0, 6.0], unit='m', variance=[0.4, 0.6], display_name='test'),
                'dim0',
            ),
            (
                DescriptorArray([3.0, 7.0], unit='m', variance=[0.3, 0.7], display_name='test'),
                'dim1',
            ),
        ],
        ids=['descriptor_array_dim0', 'descriptor_array_dim1'],
    )
    def test_sum_over_subset(self, descriptor, expected, dim):
        result = descriptor.sum(dim)
        assert type(result) == type(expected)
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)

    @pytest.mark.parametrize(
        'test, dimensions',
        [
            (
                DescriptorArray([1.0], unit='dimensionless', variance=[1.0], display_name='test'),
                ['dim0'],
            ),
            (
                DescriptorArray(
                    [[1.0, 1.0]], unit='dimensionless', variance=[[1.0, 1.0]], display_name='test'
                ),
                ['dim0', 'dim1'],
            ),
            (
                DescriptorArray(
                    [[1.0], [1.0]],
                    unit='dimensionless',
                    variance=[[1.0], [1.0]],
                    display_name='test',
                ),
                ['dim0', 'dim1'],
            ),
            (
                DescriptorArray(
                    [[[1.0, 1.0, 1.0]]],
                    unit='dimensionless',
                    variance=[[[1.0, 1.0, 1.0]]],
                    display_name='test',
                ),
                ['dim0', 'dim1', 'dim2'],
            ),
            (
                DescriptorArray(
                    [[[1.0]], [[1.0]], [[1.0]]],
                    unit='dimensionless',
                    variance=[[[1.0]], [[1.0]], [[1.0]]],
                    display_name='test',
                ),
                ['dim0', 'dim1', 'dim2'],
            ),
        ],
        ids=['1x1', '1x2', '2x1', '1x3', '3x1'],
    )
    def test_array_generate_dimensions(self, test, dimensions):
        assert test.dimensions == dimensions

    def test_array_set_dimensions_exception(self, descriptor):
        with pytest.raises(ValueError) as e:
            descriptor.dimensions = ['too_few']
        assert 'must have the same shape'
        with pytest.raises(ValueError) as e:
            DescriptorArray(
                [[1.0]], unit='m', variance=[[1.0]], dimensions=['dim'], display_name='test'
            )
        assert 'Length of dimensions' in str(e)

    def test_array_set_integer_value(self, descriptor):
        """
        Scipp does not convert ints to floats, but values need to be floats for optimization.
        """
        # When
        descriptor.value = [[1, 2], [3, 4]]
        # Then Expect
        assert isinstance(descriptor.value[0][0], float)

    def test_array_set_integer_variance(self, descriptor):
        # When
        descriptor.variance = [[1, 2], [3, 4]]
        # Then Expect
        assert isinstance(descriptor.variance[0][0], float)

    def test_array_create_with_mixed_integers_and_floats(self):
        # When
        value = [[1, 2], [3, 4]]
        variance = [[0.1, 0.2], [0.3, 0.4]]
        # Then Expect
        descriptor = DescriptorArray(
            value, unit='dimensionless', variance=variance, display_name='test'
        )  # Should not raise
        assert isinstance(descriptor.value[0][0], float)
        assert isinstance(descriptor.variance[0][0], float)

    def test_array_set_dims(self, descriptor):
        # When
        descriptor.dimensions = ['x', 'y']
        # Then Expect
        assert descriptor.dimensions[0] == 'x'
        assert descriptor.dimensions[1] == 'y'
        assert descriptor.full_value.dims[0] == 'x'
        assert descriptor.full_value.dims[1] == 'y'
