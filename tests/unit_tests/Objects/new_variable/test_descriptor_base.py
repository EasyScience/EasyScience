import pytest

from easyscience import borg
from easyscience.Objects.new_variable.descriptor_base import DescriptorBase


class TestDesciptorBase:
    @pytest.fixture
    def descriptor(self):
        # This avoids the error: TypeError: Can't instantiate abstract class DescriptorBase with abstract methods __init__
        DescriptorBase.__abstractmethods__ = set()
        DescriptorBase.__repr__ = lambda x: "DescriptorBase"
        self.objs_before_new_descriptor = len(borg.map.created_objs)
        descriptor = DescriptorBase(
            name="name",
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )
        return descriptor

    def test_init(self, descriptor: DescriptorBase):
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"
        assert len(descriptor._borg.map.created_objs) - self.objs_before_new_descriptor == 1


    def test_display_name(self, descriptor: DescriptorBase):
        # When Then Expect
        assert descriptor.display_name == "display_name"

    def test_display_name_none(self, descriptor: DescriptorBase):
        # When 
        descriptor._display_name = None
        # Then Expect
        assert descriptor.display_name == "name"
    

    @pytest.mark.parametrize("stack_enabled,stack_elements", [(False, 0), (True, 1)])
    def test_set_display_name_without_borg_stack(self, descriptor: DescriptorBase, stack_enabled, stack_elements):
        # When
        descriptor.__repr__ = lambda x: "DescriptorBase"
        descriptor._borg.stack.clear()
        descriptor._borg.stack._enabled = stack_enabled

        # Then 
        descriptor.display_name = "new_display_name" 

        # Expect
        assert descriptor.display_name == "new_display_name"
        assert len(descriptor._borg.stack.history)  == stack_elements

    def test_copy(self, descriptor: DescriptorBase):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorBase
        assert descriptor_copy._name == descriptor._name
        assert descriptor_copy._description == descriptor._description
        assert descriptor_copy._url == descriptor._url
        assert descriptor_copy._display_name == descriptor._display_name
