import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber


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
                unit=sc.units.Unit("unknown"),
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
        assert descriptor._scalar.variance == 0

    @pytest.mark.parametrize("full_value", [sc.array(values=[1,2], dims=["x"]), sc.array(values=[[1], [2]], dims=["x","y"]), object(), 1, "string"], ids=["1D", "2D", "object", "int", "string"])
    def test_from_scipp_type_exception(self, full_value):
        # When Then Expect
        with pytest.raises(TypeError):
            DescriptorNumber.from_scipp(name="name", full_value=full_value)

    def test_full_value(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.full_value == sc.scalar(1, unit='m')
        
    def test_set_full_value(self, descriptor: DescriptorNumber):
        # When Then
        descriptor.full_value = sc.scalar(2, unit='m')

        # Expect
        assert descriptor._scalar == sc.scalar(2, unit='m')

    @pytest.mark.parametrize("full_value", [sc.array(values=[1,2], dims=["x"]), sc.array(values=[[1], [2]], dims=["x","y"]), object(), 1, "string"], ids=["1D", "2D", "object", "int", "string"])
    def test_set_full_value_type_exception(self, descriptor: DescriptorNumber, full_value):
        # When Then Expect
        with pytest.raises(TypeError):
            descriptor.full_value = full_value

    def test_unit(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.unit == 'm'
        
    def test_set_unit(self, descriptor: DescriptorNumber):
        # When  Then
        descriptor.unit = 's'

        # Expect
        assert descriptor._scalar.unit == 's'

    def test_set_unit_none(self, descriptor: DescriptorNumber):
        # When  Then
        descriptor.unit = ''

        # Expect
        assert descriptor._scalar.unit == 'dimensionless'

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
        assert repr_str == "<DescriptorNumber 'name': 1.0000 m>"

    def test_copy(self, descriptor: DescriptorNumber):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorNumber
        assert descriptor_copy._scalar.value == descriptor._scalar.value
        assert descriptor_copy._scalar.unit == descriptor._scalar.unit

    def test_as_data_dict(self, descriptor: DescriptorNumber):
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
        }