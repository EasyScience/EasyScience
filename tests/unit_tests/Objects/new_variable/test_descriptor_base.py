import pytest
from unittest.mock import MagicMock
from unittest.mock import patch

from easyscience.Objects.new_variable.descriptor_base import DescriptorBase


class TestDesciptorBase:
    @pytest.fixture
    def descriptor_base(self):
        self.mock_callback = MagicMock()
        # This avoids the error: TypeError: Can't instantiate abstract class DescriptorBase with abstract methods __init__
        DescriptorBase.__abstractmethods__ = set()
        DescriptorBase.__repr__ = lambda x: "DescriptorBase"
        descriptor_base = DescriptorBase(
            name="name",
            description="description",
            url="url",
            display_name="display_name",
            callback=self.mock_callback,
            enabled=True,
            parent=None,
        )
        return descriptor_base

    def test_init(self, descriptor_base: DescriptorBase):
        assert descriptor_base._name == "name"
        assert descriptor_base._description == "description"
        assert descriptor_base._url == "url"
        assert descriptor_base._display_name == "display_name"
        assert descriptor_base._callback == self.mock_callback
        assert descriptor_base._enabled is True
        assert len(descriptor_base._borg.map.created_objs) == 1


    def test_display_name(self, descriptor_base: DescriptorBase):
        # When Then Expect
        assert descriptor_base.display_name == "display_name"

    def test_display_name_none(self, descriptor_base: DescriptorBase):
        # When 
        descriptor_base._display_name = None
        # Then Expect
        assert descriptor_base.display_name == "name"
    

    @pytest.mark.parametrize("stack_enabled,stack_elements", [(False, 0), (True, 1)])
    def test_set_display_name_without_borg_stack(self, descriptor_base: DescriptorBase, stack_enabled, stack_elements):
        # When
        descriptor_base.__repr__ = lambda x: "DescriptorBase"
        descriptor_base._borg.stack.clear()
        descriptor_base._borg.stack._enabled = stack_enabled

        # Then 
        descriptor_base.display_name = "new_display_name" 

        # Expect
        assert descriptor_base.display_name == "new_display_name"
        assert len(descriptor_base._borg.stack.history)  == stack_elements

    def test_enabled(self, descriptor_base: DescriptorBase):
        # When Then Expect
        assert descriptor_base.enabled is True

    def test_set_enabled(self, descriptor_base: DescriptorBase):
        # When
        descriptor_base.enabled = False

        # Then Expect
        assert descriptor_base._enabled is False

    def test_copy(self, descriptor_base: DescriptorBase):
        # When Then
        descriptor_base_copy = descriptor_base.__copy__()

        # Expect
        assert type(descriptor_base_copy) == DescriptorBase
        assert descriptor_base_copy._name == descriptor_base._name
        assert descriptor_base_copy._description == descriptor_base._description
        assert descriptor_base_copy._url == descriptor_base._url
        assert descriptor_base_copy._display_name == descriptor_base._display_name
        assert descriptor_base_copy._callback == descriptor_base._callback
        assert descriptor_base_copy._enabled == descriptor_base._enabled