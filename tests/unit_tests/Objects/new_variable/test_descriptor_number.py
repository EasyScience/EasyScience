import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber


class TestDescriptorNumber:
    @pytest.fixture
    def descriptor(self):
        self.mock_callback = MagicMock()
        descriptor = DescriptorNumber(
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
        return descriptor
    
    def test_init(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor._value.value == 1
        assert descriptor._value.unit == "m"
        assert descriptor._value.variance == 0.1

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"
        assert descriptor._callback == self.mock_callback
        assert descriptor._enabled == "enabled"

    def test_init_sc_unit(self):
        # When Then
        mock_callback = MagicMock()
        descriptor = DescriptorNumber(
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

        # Expect
        assert descriptor._value.value == 1
        assert descriptor._value.unit == "m"
        assert descriptor._value.variance == 0.1

    def test_value_match_callback(self, descriptor: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then Expect
        assert descriptor.value == sc.scalar(1, unit='m')
        assert descriptor._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, descriptor: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

        # Then Expect
        assert descriptor.value == sc.scalar(2, unit='m')
        assert descriptor._callback.fget.call_count == 1

    def test_set_value(self, descriptor: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor.value = sc.scalar(2, unit='m')

        # Expect
        descriptor._callback.fset.assert_called_once_with(sc.scalar(2, unit='m')) 
        assert descriptor._value == sc.scalar(2, unit='m')

    def test_unit(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.unit == 'm'
        
    def test_set_unit(self, descriptor: DescriptorNumber):
        # When  Then
        descriptor.unit = 's'

        # Expect
        assert descriptor._value.unit == 's'

    def test_set_unit_none(self, descriptor: DescriptorNumber):
        # When  Then
        descriptor.unit = ''

        # Expect
        assert descriptor._value.unit == 'dimensionless'

    def test_convert_unit(self, descriptor: DescriptorNumber):
        # When  Then
        descriptor.convert_unit('mm')

        # Expect
        assert descriptor._value.unit == 'mm'
        assert descriptor._value.value == 1000
        assert descriptor._value.variance == 100000

    def test_variance(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.variance == 0.1
        
    def test_set_variance(self, descriptor: DescriptorNumber):
        # When Then
        descriptor.variance = 0.2

        # Expect
        assert descriptor._value.variance == 0.2

    def test_raw_value(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.raw_value == 1

    def test_set_raw_value(self, descriptor: DescriptorNumber):
        # When Then
        descriptor.raw_value = 2

        # Expect
        assert descriptor.raw_value == 2

    def test_repr(self, descriptor: DescriptorNumber):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert repr_str == "<DescriptorNumber 'name': 1.0000m>"

    def test_copy(self, descriptor: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorNumber
        assert descriptor_copy._value.value == descriptor._value.value
        assert descriptor_copy._value.unit == descriptor._value.unit
