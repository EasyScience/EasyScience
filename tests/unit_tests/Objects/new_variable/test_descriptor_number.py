import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber


class TestDescriptorNumber:
    @pytest.fixture
    def descriptor_number(self):
        self.mock_callback = MagicMock()
        descriptor_number = DescriptorNumber(
            name="name",
            value=1,
            unit="m",
            variance=0.1,
            description="description",
            url="url",
            display_name="display_name",
            callback=self.mock_callback,
            enabled="enabled",
            parent=None,
        )
        return descriptor_number
    
    def test_init(self, descriptor_number: DescriptorNumber):
        assert descriptor_number._value.value == 1
        assert descriptor_number._value.unit == "m"
        assert descriptor_number._value.variance == 0.1
        assert descriptor_number._name == "name"
        assert descriptor_number._description == "description"
        assert descriptor_number._url == "url"
        assert descriptor_number._display_name == "display_name"
        assert descriptor_number._callback == self.mock_callback
        assert descriptor_number._enabled == "enabled"

    def test_init_sc_unit(self):
        mock_callback = MagicMock()
        descriptor_number = DescriptorNumber(
            name="name",
            value=1,
            unit=sc.units.Unit("m"),
            variance=0.1,
            description="description",
            url="url",
            display_name="display_name",
            callback=mock_callback,
            enabled="enabled",
            parent=None,
        )
        assert descriptor_number._value.value == 1
        assert descriptor_number._value.unit == "m"
        assert descriptor_number._value.variance == 0.1

    def test_value_match_callback(self, descriptor_number: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then Expect
        assert descriptor_number.value == sc.scalar(1, unit='m')
        assert descriptor_number._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, descriptor_number: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

        # Then Expect
        assert descriptor_number.value == sc.scalar(2, unit='m')
        assert descriptor_number._callback.fget.call_count == 1

    def test_set_value(self, descriptor_number: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_number.value = sc.scalar(2, unit='m')

        # Expect
        descriptor_number._callback.fset.assert_called_once_with(sc.scalar(2, unit='m')) 
        assert descriptor_number._value == sc.scalar(2, unit='m')

    def test_unit(self, descriptor_number: DescriptorNumber):
        # When Then Expect
        assert descriptor_number.unit == 'm'
        
    def test_set_unit(self, descriptor_number: DescriptorNumber):
        # When  Then
        descriptor_number.unit = 's'

        # Expect
        assert descriptor_number._value.unit == 's'

    def test_set_unit_none(self, descriptor_number: DescriptorNumber):
        # When  Then
        descriptor_number.unit = ''

        # Expect
        assert descriptor_number._value.unit == 'dimensionless'

    def test_convert_unit(self, descriptor_number: DescriptorNumber):
        # When  Then
        descriptor_number.convert_unit('mm')

        # Expect
        assert descriptor_number._value.unit == 'mm'
        assert descriptor_number._value.value == 1000
        assert descriptor_number._value.variance == 100000

    def test_variance(self, descriptor_number: DescriptorNumber):
        # When Then Expect
        assert descriptor_number.variance == 0.1
        
    def test_set_variance(self, descriptor_number: DescriptorNumber):
        # When  Then
        descriptor_number.variance = 0.2

        # Expect
        assert descriptor_number._value.variance == 0.2

    def test_raw_value(self, descriptor_number: DescriptorNumber):
        # When Then Expect
        assert descriptor_number.raw_value == 1

    def test_repr(self, descriptor_number: DescriptorNumber):
        # When Then
        repr_str = str(descriptor_number)

        # Expect
        assert repr_str == "<DescriptorNumber 'name': 1.0000m>"

    def test_copy(self, descriptor_number: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_number_copy = descriptor_number.__copy__()

        # Expect
        assert type(descriptor_number_copy) == DescriptorNumber
        assert descriptor_number_copy._value.value == descriptor_number._value.value
        assert descriptor_number_copy._value.unit == descriptor_number._value.unit
