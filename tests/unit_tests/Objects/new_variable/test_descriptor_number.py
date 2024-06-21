import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber


class TestDescriptorNumber:
    @pytest.fixture
    def descriptor(self):
#        self.mock_callback = MagicMock()
        descriptor = DescriptorNumber(
            name="name",
            value=1,
            unit="m",
            variance=0.1,
            description="description",
            url="url",
            display_name="display_name",
#            callback=self.mock_callback,
#            enabled="enabled",
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
#        assert descriptor._callback == self.mock_callback
#        assert descriptor._enabled == "enabled"

    def test_init_sc_unit(self):
        # When Then
#        mock_callback = MagicMock()
        descriptor = DescriptorNumber(
            name="name",
            value=1,
            unit=sc.units.Unit("m"),
            variance=0.1,
            description="description",
            url="url",
            display_name="display_name",
#            callback=mock_callback,
#            enabled="enabled",
            parent=None,
        )

        # Expect
        assert descriptor._scalar.value == 1
        assert descriptor._scalar.unit == "m"
        assert descriptor._scalar.variance == 0.1

    @pytest.mark.parametrize("value", [True, "string"])
    def test_init_value_type_exception(self, value):
        # When 
 #       mock_callback = MagicMock()

        # Then Expect
        with pytest.raises(ValueError):
            DescriptorNumber(
                name="name",
                value=value,
                unit="m",
                variance=0.1,
                description="description",
                url="url",
                display_name="display_name",
  #              callback=mock_callback,
#                enabled="enabled",
                parent=None,
            )

    def test_init_variance_exception(self):
        # When 
   #     mock_callback = MagicMock()
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
    #            callback=mock_callback,
#                enabled="enabled",
                parent=None,
            )

    def test_full_value(self, descriptor: DescriptorNumber):
        # When Then Expect
        assert descriptor.full_value == sc.scalar(1, unit='m')
        
    def test_set_full_value(self, descriptor: DescriptorNumber):
        # When Then
        descriptor.full_value = sc.scalar(2, unit='m')

        # Expect
        assert descriptor._scalar == sc.scalar(2, unit='m')

    # def test_value_match_callback(self, descriptor: DescriptorNumber):
    #     # When
    #     self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

    #     # Then Expect
    #     assert descriptor.value == sc.scalar(1, unit='m')
    #     assert descriptor._callback.fget.call_count == 1
        
    # def test_value_no_match_callback(self, descriptor: DescriptorNumber):
    #     # When
    #     self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

    #     # Then Expect
    #     assert descriptor.value == sc.scalar(2, unit='m')
    #     assert descriptor._callback.fget.call_count == 1

    # def test_set_value(self, descriptor: DescriptorNumber):
    #     # When
    #     self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

    #     # Then
    #     descriptor.value = sc.scalar(2, unit='m')

    #     # Expect
    #     descriptor._callback.fset.assert_called_once_with(sc.scalar(2, unit='m')) 
    #     assert descriptor._value == sc.scalar(2, unit='m')

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
        # When
#        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorNumber
        assert descriptor_copy._scalar.value == descriptor._scalar.value
        assert descriptor_copy._scalar.unit == descriptor._scalar.unit
