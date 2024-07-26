import pytest
from unittest.mock import MagicMock
import scipp as sc

from scipp import UnitError

from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber
from easyscience import global_object

class TestDescriptorNumber:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorNumber(
            name="name",
            value=1,
            unit="m",
            variance=0.1,
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )
        return descriptor

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor._scalar.value == 1
        assert descriptor._scalar.unit == "m"
        assert descriptor._scalar.variance == 0.1

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"

    def test_init_sc_unit(self):
        # When Then
        descriptor = DescriptorNumber(
            name="name",
            value=1,
            unit=sc.units.Unit("m"),
            variance=0.1,
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )

        # Expect
        assert descriptor._scalar.value == 1
        assert descriptor._scalar.unit == "m"
        assert descriptor._scalar.variance == 0.1

    def test_init_sc_unit_unknown(self):
        # When Then Expect
        with pytest.raises(ValueError):
            DescriptorNumber(
                name="name",
                value=1,
                unit="unknown",
                variance=0.1,
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    @pytest.mark.parametrize("value", [True, "string"])
    def test_init_value_type_exception(self, value):
        # When 

        # Then Expect
        with pytest.raises(TypeError):
            DescriptorNumber(
                name="name",
                value=value,
                unit="m",
                variance=0.1,
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    def test_init_variance_exception(self):
        # When 
        variance = -1

        # Then Expect
        with pytest.raises(ValueError):
            DescriptorNumber(
                name="name",
                value=1,
                unit="m",
                variance=variance,
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    # test from_scipp
    def test_from_scipp(self):
        # When
        full_value = sc.scalar(1, unit='m')

        # Then
        descriptor = DescriptorNumber.from_scipp(name="name", full_value=full_value)

        # Expect
        assert descriptor._scalar.value == 1
        assert descriptor._scalar.unit == "m"
        assert descriptor._scalar.variance == None

    @pytest.mark.parametrize("full_value", [sc.array(values=[1,2], dims=["x"]), sc.array(values=[[1], [2]], dims=["x","y"]), object(), 1, "string"], ids=["1D", "2D", "object", "int", "string"])
    def test_from_scipp_type_exception(self, full_value):
        # When Then Expect
        with pytest.raises(TypeError):
            DescriptorNumber.from_scipp(name="name", full_value=full_value)

    def test_full_value(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.full_value == sc.scalar(1, unit='m')
        
    def test_set_full_value(self, descriptor: DescriptorNumber):
        with pytest.raises(AttributeError):
            descriptor.full_value = sc.scalar(2, unit='s')

    def test_unit(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.unit == 'm'
        
    def test_set_unit(self, descriptor: DescriptorNumber):
        with pytest.raises(AttributeError):
            descriptor.unit = 's'

    def test_convert_unit(self, descriptor: DescriptorNumber):
        # When  Then
        descriptor.convert_unit('mm')

        # Expect
        assert descriptor._scalar.unit == 'mm'
        assert descriptor._scalar.value == 1000
        assert descriptor._scalar.variance == 100000

    def test_variance(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.variance == 0.1
        
    def test_set_variance(self, descriptor: DescriptorNumber):
        # When Then
        descriptor.variance = 0.2

        # Expect
        assert descriptor._scalar.variance == 0.2

    def test_value(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.value == 1

    def test_set_value(self, descriptor: DescriptorNumber):
        # When Then
        descriptor.value = 2

        # Expect
        assert descriptor._scalar.value == 2

    def test_repr(self, descriptor: DescriptorNumber):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert repr_str ==  "<DescriptorNumber 'name': 1.0000 ± 0.3162 m>"

    def test_copy(self, descriptor: DescriptorNumber):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorNumber
        assert descriptor_copy._scalar.value == descriptor._scalar.value
        assert descriptor_copy._scalar.unit == descriptor._scalar.unit

    def test_as_data_dict(self, clear, descriptor: DescriptorNumber):
        # When Then
        descriptor_dict = descriptor.as_data_dict()

        # Expect
        assert descriptor_dict == {
            "name": "name",
            "value": 1.0,
            "unit": "m",
            "variance": 0.1,
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "unique_name": "DescriptorNumber_0",
        }

    @pytest.mark.parametrize("test, expected", [
        (DescriptorNumber("test", 2, "m", 0.01,),   DescriptorNumber("test + name", 3, "m", 0.11)),
        (DescriptorNumber("test", 2, "cm", 0.01),   DescriptorNumber("test + name", 102, "cm", 1000.01))],
        ids=["regular", "unit_conversion"])
    def test_addition(self, descriptor: DescriptorNumber, test, expected):
        # When Then
        result = test + descriptor

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == expected.name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        
        assert descriptor.unit == 'm'

    def test_addition_with_scalar(self):
        # When 
        descriptor = DescriptorNumber(name="name", value=1, variance=0.1)

        # Then
        result = descriptor + 1.0
        result_reverse = 1.0 + descriptor

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == "name + 1.0"
        assert result.value == 2.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.1

        assert type(result_reverse) == DescriptorNumber
        assert result_reverse.name == "1.0 + name"
        assert result_reverse.value == 2.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.1

    @pytest.mark.parametrize("test", [1.0, DescriptorNumber("test", 2, "s",)], ids=["add_scalar_to_unit", "incompatible_units"])
    def test_addition_exception(self, descriptor: DescriptorNumber, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = descriptor + test
        with pytest.raises(UnitError):
            result_reverse = test + descriptor
        
    @pytest.mark.parametrize("test, expected", [
        (DescriptorNumber("test", 2, "m", 0.01,),   DescriptorNumber("test - name", 1, "m", 0.11)),
        (DescriptorNumber("test", 2, "cm", 0.01),   DescriptorNumber("test - name", -98, "cm", 1000.01))],
        ids=["regular", "unit_conversion"])
    def test_subtraction(self, descriptor: DescriptorNumber, test, expected):
        # When Then
        result = test - descriptor

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == expected.name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance

        assert descriptor.unit == 'm'

    def test_subtraction_with_scalar(self):
        # When 
        descriptor = DescriptorNumber(name="name", value=2, variance=0.1)

        # Then
        result = descriptor - 1.0
        result_reverse = 1.0 - descriptor

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == "name - 1.0"
        assert result.value == 1.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.1

        assert type(result_reverse) == DescriptorNumber
        assert result_reverse.name == "1.0 - name"
        assert result_reverse.value == -1.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.1

    @pytest.mark.parametrize("test", [1.0, DescriptorNumber("test", 2, "s",)], ids=["sub_scalar_to_unit", "incompatible_units"])
    def test_subtraction_exception(self, descriptor: DescriptorNumber, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = test - descriptor
        with pytest.raises(UnitError):
            result_reverse = descriptor - test

    @pytest.mark.parametrize("test, expected", [
        (DescriptorNumber("test", 2, "m", 0.01,),   DescriptorNumber("test * name", 2, "m^2", 0.41)),
        (DescriptorNumber("test", 2, "dm", 0.01),   DescriptorNumber("test * name", 0.2, "m^2", 0.0041)),
        (DescriptorNumber("test", 2, "1/dm", 0.01), DescriptorNumber("test * name", 20.0, "dimensionless", 41))],
        ids=["regular", "base_unit_conversion", "base_unit_conversion_dimensionless"])
    def test_multiplication(self, descriptor: DescriptorNumber, test, expected):
        # When Then
        result = test * descriptor

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == expected.name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == pytest.approx(expected.variance)

    def test_multiplication_with_scalar(self, descriptor: DescriptorNumber):
        # When Then
        result = descriptor * 2.0
        result_reverse = 2.0 * descriptor

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == "name * 2.0"
        assert result.value == 2.0
        assert result.unit == "m"
        assert result.variance == 0.4

        assert type(result_reverse) == DescriptorNumber
        assert result_reverse.name == "2.0 * name"
        assert result_reverse.value == 2.0
        assert result_reverse.unit == "m"
        assert result_reverse.variance == 0.4