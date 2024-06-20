import pytest
from unittest.mock import MagicMock

from easyscience.Objects.new_variable.descriptor_str import DescriptorStr


class TestDescriptorStr:
    @pytest.fixture
    def descriptor(self):
#        self.mock_callback = MagicMock()
        descriptor = DescriptorStr(
            name="name",
            string="string",
            description="description",
            url="url",
            display_name="display_name",
#            callback=self.mock_callback,
#            enabled="enabled",
            parent=None,
        )
        return descriptor
    
    def test_init(self, descriptor: DescriptorStr):
        # When Then Expect
        assert descriptor._string == "string"

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"
#        assert descriptor._callback == self.mock_callback
#        assert descriptor._enabled == "enabled"

    @pytest.mark.parametrize("string", [True, 0, 1.0])
    def test_init_string_type_exception(self, string):
        # When 
        mock_callback = MagicMock()

        # Then Expect
        with pytest.raises(ValueError):
            DescriptorStr(
                name="name",
                string=string,
                description="description",
                url="url",
                display_name="display_name",
#                callback=mock_callback,
#                enabled="enabled",
                parent=None,
            )

    def test_value(self, descriptor: DescriptorStr):
        # When Then Expect
        assert descriptor.value == "string"

    def test_set_value(self, descriptor: DescriptorStr):
        # When Then
        descriptor.value = "new_string"

        # Expect
        assert descriptor._string == "new_string"
    # def test_value_match_callback(self, descriptor: DescriptorStr):
    #     # When Then
    #     self.mock_callback.fget.return_value = "string"

    #     # Expect
    #     assert descriptor.value == "string"
    #     assert descriptor._callback.fget.call_count == 1
        
    # def test_value_no_match_callback(self, descriptor: DescriptorStr):
    #     # When Then
    #     self.mock_callback.fget.return_value = "call_back"

    #     # Expect
    #     assert descriptor.value == "call_back"
    #     assert descriptor._callback.fget.call_count == 1

    # def test_set_value(self, descriptor: DescriptorStr):
    #     # When
    #     self.mock_callback.fget.return_value = "string"

    #     # Then
    #     descriptor.value = "new_string"

    #     # Expect
    #     descriptor._callback.fset.assert_called_once_with("new_string") 
    #     assert descriptor._string == "new_string"

    def test_repr(self, descriptor: DescriptorStr):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert repr_str == "<DescriptorStr 'name': string>"

    def test_copy(self, descriptor: DescriptorStr):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorStr
        assert descriptor_copy._string == descriptor._string
