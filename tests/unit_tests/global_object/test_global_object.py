import easyscience
from easyscience.global_object.global_object import GlobalObject
from easyscience.Objects.new_variable.descriptor_bool import DescriptorBool

class TestGlobalObject:
    def test_init(self):
        # When Then
        global_object = GlobalObject()

        # Expect
        assert global_object.log == GlobalObject().log 
        assert global_object.map == GlobalObject().map
        assert global_object.stack == GlobalObject().stack
        assert global_object.debug == GlobalObject().debug

    def test_generate_unique_name(self):
        # When Then
        global_object = GlobalObject()
        name = global_object.generate_unique_name("name_prefix")

        # Expect
        assert name == "name_prefix_0"

    def test_generate_unique_name_already_taken(self):
        # When
        global_object = GlobalObject()
        # Block the other_name_prefix_2 name
        keep_due_toweakref = DescriptorBool(name="test", value=True, unique_name="other_name_prefix_2")

        # Then 
        zero_name = global_object.generate_unique_name("other_name_prefix")
        first_name = global_object.generate_unique_name("other_name_prefix")
        second_name = global_object.generate_unique_name("other_name_prefix")

        # Expect
        assert zero_name == "other_name_prefix_0"
        assert first_name == "other_name_prefix_1"
        # Name was already taken, so the next available name is 3
        assert second_name == "other_name_prefix_3"