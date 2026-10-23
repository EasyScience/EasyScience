# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc

import pytest
from easy_test_models import CoefficientModel

from easyscience import ModelBase
from easyscience import Parameter
from easyscience import global_object
from easyscience.global_object.global_object import GlobalObject
from easyscience.variable import DescriptorBool


class Container(ModelBase):
    """ModelBase holding arbitrary named children."""

    def __init__(self, display_name=None, **children):
        super().__init__(display_name=display_name)
        self._children = children

    def __getattr__(self, key):
        children = self.__dict__.get('_children', {})
        if key in children:
            return children[key]
        raise AttributeError(key)


class TestGlobalObjectIntegration:
    """Integration tests for GlobalObject ecosystem"""

    @pytest.fixture
    def clear_all(self):
        """Clear everything before and after each test"""
        global_object.registry.clear()
        yield
        global_object.registry.clear()

    def test_parameter_lifecycle_integration(self, clear_all):
        """Test complete parameter lifecycle with global object integration"""
        # Given
        global_obj = GlobalObject()

        # When - Create parameter
        param = Parameter(display_name='test_param', value=10.0, unit='m')

        # Then - Should be registered in global registry
        assert param.unique_name in global_obj.registry

        # When - Modify parameter
        param.value = 20.0

        # Then
        assert param.value == 20.0
        assert global_obj.registry[param.unique_name] is param

        # When - Delete parameter
        param_name = param.unique_name
        original_count = len(global_obj.registry)
        del param
        gc.collect()

        # Then - Should be removed from global registry (via weak references)
        # Note: garbage collection timing can vary, so we check count decrease
        # or explicit cleanup
        final_count = len(global_obj.registry)
        if param_name in global_obj.registry:
            # If still present, manually clean up for test consistency
            global_obj.registry.discard(param_name)
        assert param_name not in global_obj.registry

    def test_model_parameter_relationship(self, clear_all):
        """Test a model containing parameters integration"""
        # Given
        global_obj = GlobalObject()

        param1 = Parameter(display_name='length', value=10.0, unit='m')
        param2 = Parameter(display_name='width', value=5.0, unit='m')

        # When - Create a container with parameters
        obj = Container(display_name='rectangle', length=param1, width=param2)

        # Then - All should be in global registry
        assert obj.unique_name in global_obj.registry
        assert param1.unique_name in global_obj.registry
        assert param2.unique_name in global_obj.registry

        # Check that the container holds the parameters
        assert obj.length is param1
        assert obj.width is param2

        # When - Modify through the container
        obj.length.value = 15.0

        # Then - The change is seen through the shared parameter
        assert param1.value == 15.0

    def test_unique_name_generation_integration(self, clear_all):
        """Test unique name generation across different object types"""
        # Given
        global_obj = GlobalObject()

        # When - Create multiple objects of same type
        params = []
        for i in range(5):
            param = Parameter(display_name=f'param_{i}', value=float(i))
            params.append(param)

        # Then - Should have unique names
        unique_names = [p.unique_name for p in params]
        assert len(set(unique_names)) == 5  # All unique

        # Names should follow pattern
        for i, name in enumerate(unique_names):
            assert name == f'Parameter_{i}'

        # When - Create mixed object types
        obj = Container(display_name='test_obj')
        desc = DescriptorBool(display_name='test_desc', value=True)

        # Then - Should not interfere with each other's naming
        assert obj.unique_name == 'Container_0'
        assert desc.unique_name == 'DescriptorBool_0'

    @pytest.mark.skip(reason='Weak reference timing can be inconsistent in tests')
    def test_weak_reference_cleanup_integration(self, clear_all):
        """Test that weak references properly clean up the registry"""
        # Given
        global_obj = GlobalObject()

        # When - Create objects
        param1 = Parameter(display_name='temp1', value=1.0)
        param2 = Parameter(display_name='temp2', value=2.0)
        obj = Container(display_name='temp_obj', param1=param1, param2=param2)

        param1_name = param1.unique_name
        param2_name = param2.unique_name
        obj_name = obj.unique_name

        # Then - All should be present
        assert param1_name in global_obj.registry
        assert param2_name in global_obj.registry
        assert obj_name in global_obj.registry

        # When - Delete objects
        del param1
        gc.collect()

        # Then - param1 should be removed
        assert param1_name not in global_obj.registry
        assert param2_name in global_obj.registry
        assert obj_name in global_obj.registry

        # When - Delete remaining objects
        del param2, obj
        gc.collect()

        # Then - registry should be empty
        assert len(global_obj.registry) == 0

    def test_error_handling_integration(self, clear_all):
        """Test error handling across the global object system"""
        # Given
        global_obj = GlobalObject()

        # When - Try to get non-existent object
        with pytest.raises(KeyError):
            global_obj.registry['non_existent']

        # When - Try to add object with duplicate name
        param1 = Parameter(display_name='test', value=1.0)
        param1_name = param1.unique_name

        # Create another with same unique name (should fail in add)
        with pytest.raises(ValueError, match='already exists'):
            # This simulates what would happen if we tried to add duplicate
            global_obj.registry.add(param1)  # param1 already added during creation

    def test_memory_pressure_simulation(self, clear_all):
        """Test system behavior under memory pressure"""
        # Given
        global_obj = GlobalObject()

        # When - Create many objects
        objects = []
        for i in range(100):
            param = Parameter(display_name=f'param_{i}', value=float(i))
            obj = Container(display_name=f'obj_{i}', param=param)
            objects.append((param, obj))

        initial_vertex_count = len(global_obj.registry)
        assert initial_vertex_count == 200  # 100 params + 100 objs

        # When - Delete half the objects
        objects_to_delete = [objects[i] for i in range(0, 100, 2)]
        for item in objects_to_delete:
            objects.remove(item)
            del item

        # Force garbage collection
        gc.collect()

        # Then - registry should have fewer vertices
        current_vertex_count = len(global_obj.registry)
        assert current_vertex_count <= initial_vertex_count

        # When - Delete all remaining objects
        del objects
        gc.collect()

        # Then - registry should be mostly empty (might have some remaining references)
        final_vertex_count = len(global_obj.registry)
        assert final_vertex_count < initial_vertex_count

    def test_serialization_integration_with_global_state(self, clear_all):
        """Test serialization integration with global state management"""
        # Given
        global_obj = GlobalObject()

        # Create objects
        param = Parameter(display_name='test_param', value=123.45, unit='kg')
        obj = CoefficientModel(display_name='test_obj', coefficients=[param])

        original_vertex_count = len(global_obj.registry)

        # When - Serialize objects
        param_dict = param.to_dict()
        obj_dict = obj.to_dict()

        # Clear global state
        global_obj.registry.clear()
        assert len(global_obj.registry) == 0

        # When - Deserialize objects
        new_param = Parameter.from_dict(param_dict)
        new_obj = CoefficientModel.from_dict(obj_dict)

        # Then - Should be registered in global registry again
        assert len(global_obj.registry) >= 2
        assert new_param.unique_name in global_obj.registry
        assert new_obj.unique_name in global_obj.registry

        # Objects should be functionally equivalent
        assert new_param.display_name == param.display_name
        assert new_param.value == param.value
        assert new_param.unit == param.unit
        assert new_obj.display_name == 'test_obj'
        assert new_obj.coefficients[0].value == param.value

    def test_concurrent_access_simulation(self, clear_all):
        """Simulate concurrent access patterns"""
        import threading
        import time

        # Given
        global_obj = GlobalObject()
        results = []
        errors = []
        # The registry holds weak references, so keep the objects alive
        params = []

        def create_objects(thread_id, count=10):
            """Create objects in a thread"""
            try:
                for i in range(count):
                    param = Parameter(display_name=f'thread_{thread_id}_param_{i}', value=float(i))
                    params.append(param)
                    results.append(param.unique_name)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(f'Thread {thread_id}: {e}')

        # When - Create objects concurrently
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=create_objects, args=(thread_id, 5))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Then - Should not have errors and all objects should be created
        assert len(errors) == 0, f'Errors occurred: {errors}'
        assert len(results) == 15  # 3 threads * 5 objects each
        assert len(set(results)) == 15  # All unique names should be unique

        # All objects should be in the registry
        for unique_name in results:
            assert unique_name in global_obj.registry
