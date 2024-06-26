import pytest

from easyscience.Objects.new_variable.descriptor_str import DescriptorStr


class TestDescriptorStr:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorStr(
            name="name",
            value="string",
            description="description",
            url="url",
            display_name="display_name",
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

    @pytest.mark.parametrize("string", [True, 0, 1.0])
    def test_init_string_type_exception(self, string):
        # When Then Expect
        with pytest.raises(ValueError):
            DescriptorStr(
                name="name",
                value=string,
                description="description",
                url="url",
                display_name="display_name",
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