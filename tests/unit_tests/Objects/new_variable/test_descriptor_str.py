import pytest
from unittest.mock import MagicMock

from easyscience.Objects.new_variable.descriptor_str import DescriptorStr


class TestDescriptorStr:
    @pytest.fixture
    def descriptor_str(self):
        self.mock_callback = MagicMock()
        descriptor_str = DescriptorStr(
            name="name",
            string="string",
            description="description",
            url="url",
            display_name="display_name",
            callback=self.mock_callback,
            enabled="enabled",
            parent=None,
        )
        return descriptor_str
    
    def test_init(self, descriptor_str: DescriptorStr):
        assert descriptor_str._string == "string"
        assert descriptor_str._name == "name"
        assert descriptor_str._description == "description"
        assert descriptor_str._url == "url"
        assert descriptor_str._display_name == "display_name"
        assert descriptor_str._callback == self.mock_callback
        assert descriptor_str._enabled == "enabled"

    def test_value_match_callback(self, descriptor_str: DescriptorStr):
        # When Then
        self.mock_callback.fget.return_value = "string"

        # Expect
        assert descriptor_str.value == "string"
        assert descriptor_str._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, descriptor_str: DescriptorStr):
        # When Then
        self.mock_callback.fget.return_value = "call_back"

        # Expect
        assert descriptor_str.value == "call_back"
        assert descriptor_str._callback.fget.call_count == 1

    def test_set_value(self, descriptor_str: DescriptorStr):
        # When
        self.mock_callback.fget.return_value = "string"

        # Then
        descriptor_str.value = "new_string"

        # Expect
        descriptor_str._callback.fset.assert_called_once_with("new_string") 
        assert descriptor_str._string == "new_string"

    def test_repr(self, descriptor_str: DescriptorStr):
        # When Then
        repr_str = str(descriptor_str)

        # Expect
        assert repr_str == "<DescriptorStr 'name': string>"

    def test_copy(self, descriptor_str: DescriptorStr):
        # When Then
        descriptor_str_copy = descriptor_str.__copy__()

        # Expect
        assert type(descriptor_str_copy) == DescriptorStr
        assert descriptor_str_copy._string == descriptor_str._string
