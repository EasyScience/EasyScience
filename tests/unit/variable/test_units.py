# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import scipp as sc
from scipp import UnitError

from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import global_object
from easyscience.variable.units import has_numeric_factor
from easyscience.variable.units import normalisation_target
from easyscience.variable.units import si_base_unit


class TestUnitHelpers:
    @pytest.mark.parametrize(
        'unit_string, expected',
        [
            ('m', False),
            ('mm', False),
            ('meV', False),
            ('m/s', False),
            ('counts', False),
            ('1/m', False),
            ('1/meV', False),
            ('1/J**2', False),
            ('mm*mm', False),
            ('nm*m/s', False),
            ('degC', False),
            ('deg', False),
            ('1000', True),
            ('1e+9', True),
            ('m/mm', True),
            ('dm*m', True),
            ('10dm^2', True),
            ('meV*meV', True),
            ('1/meV**2', True),
        ],
    )
    def test_has_numeric_factor(self, unit_string, expected):
        assert has_numeric_factor(sc.Unit(unit_string)) is expected

    @pytest.mark.parametrize(
        'unit_string, expected',
        [
            ('m', 'm'),
            ('mm', 'm'),
            ('dm*m', 'm^2'),
            ('m/mm', 'dimensionless'),
            ('1000', 'dimensionless'),
            ('nm*m/s', 'm^2/s'),
            ('meV', 'J'),
            ('counts/s', 'counts/s'),
        ],
    )
    def test_si_base_unit(self, unit_string, expected):
        assert str(si_base_unit(sc.Unit(unit_string))) == expected

    @pytest.mark.parametrize('unit_string', ['degC', 'deg', 'm', 'mm', 'counts'])
    def test_normalisation_target_leaves_clean_units_alone(self, unit_string):
        # Offset units such as degC must not be touched either: they cannot be
        # converted to their base unit this way.
        assert normalisation_target(sc.Unit(unit_string)) is None


class TestNumericFactorFolding:
    @pytest.mark.parametrize(
        'unit_string, expected_unit, expected_value',
        [
            ('1e+9', 'dimensionless', 1e9),
            ('1000', 'dimensionless', 1000.0),
            ('m/mm', 'dimensionless', 1000.0),
            ('10dm^2', 'm^2', 0.1),
        ],
        ids=['scientific_notation', 'number', 'cancelling_prefixes', 'unit_prefix'],
    )
    def test_construction_folds_the_factor(self, unit_string, expected_unit, expected_value):
        descriptor = DescriptorNumber(name='name', value=1.0, unit=unit_string)

        assert descriptor.unit == expected_unit
        assert descriptor.value == pytest.approx(expected_value)

    @pytest.mark.parametrize(
        'left, right, expected_unit, expected_value',
        [
            ('m', 'mm', 'dimensionless', 1000.0),
            ('m', 'm', 'dimensionless', 1.0),
            ('m', 's', 'm/s', 1.0),
        ],
        ids=['cancelling_prefixes', 'identical', 'unrelated'],
    )
    def test_division_folds_the_factor(self, left, right, expected_unit, expected_value):
        result = DescriptorNumber('a', 1.0, left) / DescriptorNumber('b', 1.0, right)

        assert result.unit == expected_unit
        assert result.value == pytest.approx(expected_value)

    @pytest.mark.parametrize(
        'left, right, expected_unit, expected_value',
        [
            ('dm', 'm', 'm^2', 0.2),
            ('mm', 'mm', 'mm^2', 2.0),
            ('cm', 'm', 'dm^2', 2.0),
        ],
        ids=['scaled', 'already_named', 'already_named_prefix'],
    )
    def test_multiplication_folds_only_when_needed(
        self, left, right, expected_unit, expected_value
    ):
        # A unit scipp can name is left exactly as it names it; only a unit displayed
        # with a numeric factor is converted.
        result = DescriptorNumber('a', 2.0, left) * DescriptorNumber('b', 1.0, right)

        assert result.unit == expected_unit
        assert result.value == pytest.approx(expected_value)

    def test_bounds_are_folded_with_the_value(self):
        parameter = Parameter(name='name', value=1.0, min=0.0, max=10.0, unit='m/mm')

        assert parameter.unit == 'dimensionless'
        assert parameter.value == pytest.approx(1000.0)
        assert parameter.min == pytest.approx(0.0)
        assert parameter.max == pytest.approx(10000.0)

    def test_invalid_unit_still_raises(self):
        with pytest.raises(UnitError):
            DescriptorNumber(name='name', value=1.0, unit='not_a_unit')


class TestUnitSpelling:
    @pytest.mark.parametrize(
        'unit_string',
        ['angstrom', 'nm*m/s', '1/meV^2', 'dm*m', 'Hz'],
    )
    def test_unit_is_reported_as_written(self, unit_string):
        descriptor = DescriptorNumber(name='name', value=1.0, unit=unit_string)

        assert descriptor.unit == unit_string
        # The spelling is a display concern only; the meaning is unchanged.
        assert descriptor._scalar.unit == sc.Unit(unit_string)

    def test_spellings_are_per_object(self):
        # The property a global alias table could not provide: scipp permits only one
        # alias per unit, so one of these two would have had to report the other's
        # spelling.
        angstrom = DescriptorNumber(name='a', value=1.0, unit='angstrom')
        symbol = DescriptorNumber(name='b', value=1.0, unit='Å')

        assert angstrom.unit == 'angstrom'
        assert symbol.unit == 'Å'

    @pytest.mark.parametrize('unit_string', ['', 'dimensionless', 'one'])
    def test_dimensionless_is_always_reported_as_dimensionless(self, unit_string):
        # Reporting '' or 'one' would break the comparisons against 'dimensionless'
        # made throughout DescriptorNumber.
        descriptor = DescriptorNumber(name='name', value=1.0, unit=unit_string)

        assert descriptor.unit == 'dimensionless'
        assert (descriptor + 1.0).value == pytest.approx(2.0)

    def test_derived_units_use_scipps_spelling(self):
        result = DescriptorNumber('a', 1.0, 'm') * DescriptorNumber('b', 1.0, 'm')

        assert result.unit == 'm^2'

    @pytest.mark.parametrize('unit_string', ['angstrom', '1/meV^2', 'dm*m'])
    def test_unit_preserving_operations_keep_the_spelling(self, unit_string):
        # An operation which leaves the unit alone should leave its spelling alone too,
        # and must not disturb the value on the way.
        a = DescriptorNumber('a', 2.0, unit_string)
        b = DescriptorNumber('b', 3.0, unit_string)

        assert (a + b).unit == unit_string
        assert (a + b).value == pytest.approx(5.0, rel=0, abs=0)
        assert (a * 2).unit == unit_string
        assert (a * 2).value == pytest.approx(4.0, rel=0, abs=0)
        assert (-a).unit == unit_string
        assert abs(a).unit == unit_string

    def test_a_new_unit_is_never_relabelled(self):
        # Only an exactly equal unit is inherited from an operand.
        a = DescriptorNumber('a', 2.0, 'dm')
        b = DescriptorNumber('b', 1.0, 'm')

        assert (a * b).unit == 'm^2'
        assert (a * b).value == pytest.approx(0.2)

    def test_bounds_follow_an_inherited_spelling(self):
        parameter = Parameter(name='p', value=1.0, min=0.0, max=10.0, unit='angstrom')

        result = parameter * 2

        assert result.unit == 'angstrom'
        assert result.max == pytest.approx(20.0)

    def test_spelling_falls_back_when_the_scalar_changes_underneath(self):
        descriptor = DescriptorNumber(name='name', value=1.0, unit='angstrom')
        assert descriptor.unit == 'angstrom'

        # Bypass convert_unit entirely, as Parameter._update does.
        descriptor._scalar = descriptor._scalar.to(unit='m')

        assert descriptor.unit == 'm'

    def test_convert_unit_adopts_the_new_spelling(self):
        descriptor = DescriptorNumber(name='name', value=1.0, unit='m')

        descriptor.convert_unit('angstrom')

        assert descriptor.unit == 'angstrom'
        assert descriptor.value == pytest.approx(1e10)

    def test_serialisation_round_trip_preserves_the_spelling(self):
        descriptor = DescriptorNumber(name='name', value=1.0, unit='nm*m/s')

        restored = DescriptorNumber.from_dict(descriptor.as_dict(skip=['unique_name']))

        assert restored.unit == 'nm*m/s'
        assert restored._scalar.unit == descriptor._scalar.unit

    @pytest.mark.parametrize('unit_string', ['m', 'meV', 'angstrom', 'nm*m/s', '1/m'])
    def test_sc_unit_is_treated_like_scipps_spelling_of_it(self, unit_string):
        # A sc.Unit does not carry the string it was made from, so it behaves exactly
        # like the string scipp prints for it.
        from_unit = DescriptorNumber(name='name', value=1.0, unit=sc.Unit(unit_string))
        from_string = DescriptorNumber(name='name', value=1.0, unit=str(sc.Unit(unit_string)))

        assert from_unit.unit == from_string.unit == str(sc.Unit(unit_string))
        assert from_unit.value == from_string.value == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'unit_string, expected_unit, expected_value',
        [('dm*m', 'm^2', 0.1), ('m/mm', 'dimensionless', 1000.0)],
    )
    def test_sc_unit_printed_with_a_factor_is_folded(
        self, unit_string, expected_unit, expected_value
    ):
        # scipp has already reduced e.g. 'dm*m' to '0.1m^2' before we see it, so it is
        # handled like the string '0.1m^2' would be.
        descriptor = DescriptorNumber(name='name', value=1.0, unit=sc.Unit(unit_string))

        assert descriptor.unit == expected_unit
        assert descriptor.value == pytest.approx(expected_value)


class TestUnitUndoRedo:
    def test_undo_restores_value_bounds_and_spelling_together(self):
        global_object.stack.enabled = True
        try:
            parameter = Parameter(name='name', value=1.0, min=0.0, max=10.0, unit='m')
            parameter.convert_unit('mm')
            assert parameter.unit == 'mm'
            assert parameter.value == pytest.approx(1000.0)
            assert parameter.max == pytest.approx(10000.0)

            global_object.stack.undo()

            # One undo, and everything the conversion touched comes back with it.
            assert parameter.unit == 'm'
            assert parameter.value == pytest.approx(1.0)
            assert parameter.min == pytest.approx(0.0)
            assert parameter.max == pytest.approx(10.0)
        finally:
            global_object.stack.enabled = False

    def test_construction_does_not_record_an_undo_entry(self):
        global_object.stack.enabled = True
        try:
            global_object.stack.clear()
            # A unit needing normalisation must not leave anything to undo.
            DescriptorNumber(name='name', value=1.0, unit='m/mm')

            assert global_object.stack.canUndo() is False
        finally:
            global_object.stack.enabled = False
