import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.descriptor_scalar import DescriptorScalar


class TestDesciptorStr:
    @pytest.fixture
    def descriptor_scalar(self):
        self.mock_callback = MagicMock()
        descriptor_scalar = DescriptorScalar(
            name="name",
            value=1,
            unit="m",
            description="description",
            url="url",
            display_name="display_name",
            callback=self.mock_callback,
            enabled="enabled",
            parent=None,
        )
        return descriptor_scalar
    
    def test_init(self, descriptor_scalar: DescriptorScalar):
        assert descriptor_scalar._value.value == 1
        assert descriptor_scalar._value.unit == "m"
        assert descriptor_scalar._name == "name"
        assert descriptor_scalar._description == "description"
        assert descriptor_scalar._url == "url"
        assert descriptor_scalar._display_name == "display_name"
        assert descriptor_scalar._callback == self.mock_callback
        assert descriptor_scalar._enabled == "enabled"

    def test_init_sc_unit(self):
        mock_callback = MagicMock()
        descriptor_scalar = DescriptorScalar(
            name="name",
            value=1,
            unit=sc.units.Unit("m"),
            description="description",
            url="url",
            display_name="display_name",
            callback=mock_callback,
            enabled="enabled",
            parent=None,
        )
        assert descriptor_scalar._value.value == 1
        assert descriptor_scalar._value.unit == "m"

    def test_value_match_callback(self, descriptor_scalar: DescriptorScalar):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then Expect
        assert descriptor_scalar.value == sc.scalar(1, unit='m')
        assert descriptor_scalar._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, descriptor_scalar: DescriptorScalar):
        # When
        self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

        # Then Expect
        assert descriptor_scalar.value == sc.scalar(2, unit='m')
        assert descriptor_scalar._callback.fget.call_count == 1

    def test_set_value(self, descriptor_scalar: DescriptorScalar):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_scalar.value = sc.scalar(2, unit='m')

        # Expect
        descriptor_scalar._callback.fset.assert_called_once_with(sc.scalar(2, unit='m')) 
        assert descriptor_scalar._value == sc.scalar(2, unit='m')

    def test_units(self, descriptor_scalar: DescriptorScalar):
        # When Then Expect
        assert descriptor_scalar.unit == 'm'
        
    def test_set_unit(self, descriptor_scalar: DescriptorScalar):
        # When  Then
        descriptor_scalar.unit = 's'

        # Expect
        assert descriptor_scalar._value.unit == 's'

    def test_set_unit_none(self, descriptor_scalar: DescriptorScalar):
        # When  Then
        descriptor_scalar.unit = ''

        # Expect
        assert descriptor_scalar._value.unit == 'dimensionless'

    def test_convert_unit(self, descriptor_scalar: DescriptorScalar):
        # When  Then
        descriptor_scalar.convert_unit('mm')

        # Expect
        assert descriptor_scalar._value.unit == 'mm'
        assert descriptor_scalar._value.value == 1000

    def test_raw_value(self, descriptor_scalar: DescriptorScalar):
        # When Then Expect
        assert descriptor_scalar.raw_value == 1

    def test_repr(self, descriptor_scalar: DescriptorScalar):
        # When Then
        repr_str = str(descriptor_scalar)

        # Expect
        assert repr_str == "<DescriptorScalar 'name': 1.0000m>"

    def test_copy(self, descriptor_scalar: DescriptorScalar):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_scalar_copy = descriptor_scalar.__copy__()

        # Expect
        assert type(descriptor_scalar_copy) == DescriptorScalar
        assert descriptor_scalar_copy._value.value == descriptor_scalar._value.value
        assert descriptor_scalar_copy._value.unit == descriptor_scalar._value.unit
