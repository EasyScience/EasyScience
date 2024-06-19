import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber


class TestDescriptorNumber:
    @pytest.fixture
    def descriptor_scalar(self):
        self.mock_callback = MagicMock()
        descriptor_scalar = DescriptorNumber(
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
        return descriptor_scalar
    
    def test_init(self, descriptor_scalar: DescriptorNumber):
        assert descriptor_scalar._value.value == 1
        assert descriptor_scalar._value.unit == "m"
        assert descriptor_scalar._value.variance == 0.1
        assert descriptor_scalar._name == "name"
        assert descriptor_scalar._description == "description"
        assert descriptor_scalar._url == "url"
        assert descriptor_scalar._display_name == "display_name"
        assert descriptor_scalar._callback == self.mock_callback
        assert descriptor_scalar._enabled == "enabled"

    def test_init_sc_unit(self):
        mock_callback = MagicMock()
        descriptor_scalar = DescriptorNumber(
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
        assert descriptor_scalar._value.value == 1
        assert descriptor_scalar._value.unit == "m"
        assert descriptor_scalar._value.variance == 0.1

    def test_value_match_callback(self, descriptor_scalar: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then Expect
        assert descriptor_scalar.value == sc.scalar(1, unit='m')
        assert descriptor_scalar._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, descriptor_scalar: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

        # Then Expect
        assert descriptor_scalar.value == sc.scalar(2, unit='m')
        assert descriptor_scalar._callback.fget.call_count == 1

    def test_set_value(self, descriptor_scalar: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_scalar.value = sc.scalar(2, unit='m')

        # Expect
        descriptor_scalar._callback.fset.assert_called_once_with(sc.scalar(2, unit='m')) 
        assert descriptor_scalar._value == sc.scalar(2, unit='m')

    def test_unit(self, descriptor_scalar: DescriptorNumber):
        # When Then Expect
        assert descriptor_scalar.unit == 'm'
        
    def test_set_unit(self, descriptor_scalar: DescriptorNumber):
        # When  Then
        descriptor_scalar.unit = 's'

        # Expect
        assert descriptor_scalar._value.unit == 's'

    def test_set_unit_none(self, descriptor_scalar: DescriptorNumber):
        # When  Then
        descriptor_scalar.unit = ''

        # Expect
        assert descriptor_scalar._value.unit == 'dimensionless'

    def test_convert_unit(self, descriptor_scalar: DescriptorNumber):
        # When  Then
        descriptor_scalar.convert_unit('mm')

        # Expect
        assert descriptor_scalar._value.unit == 'mm'
        assert descriptor_scalar._value.value == 1000
        assert descriptor_scalar._value.variance == 100000

    def test_variance(self, descriptor_scalar: DescriptorNumber):
        # When Then Expect
        assert descriptor_scalar.variance == 0.1
        
    def test_set_variance(self, descriptor_scalar: DescriptorNumber):
        # When  Then
        descriptor_scalar.variance = 0.2

        # Expect
        assert descriptor_scalar._value.variance == 0.2

    def test_raw_value(self, descriptor_scalar: DescriptorNumber):
        # When Then Expect
        assert descriptor_scalar.raw_value == 1

    def test_repr(self, descriptor_scalar: DescriptorNumber):
        # When Then
        repr_str = str(descriptor_scalar)

        # Expect
        assert repr_str == "<DescriptorNumber 'name': 1.0000m>"

    def test_copy(self, descriptor_scalar: DescriptorNumber):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_scalar_copy = descriptor_scalar.__copy__()

        # Expect
        assert type(descriptor_scalar_copy) == DescriptorNumber
        assert descriptor_scalar_copy._value.value == descriptor_scalar._value.value
        assert descriptor_scalar_copy._value.unit == descriptor_scalar._value.unit
