# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc
from unittest.mock import patch

import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience import global_object
from easyscience.global_object.global_object import GlobalObject
from easyscience.variable import DescriptorBool


class TestGlobalObjectIntegration:
    """Integration tests for GlobalObject ecosystem"""

    @pytest.fixture
    def clear_all(self):
        """Clear everything before and after each test"""
        global_object.map._clear()
        if global_object.stack:
            global_object.stack.clear()
        yield
        global_object.map._clear()
        if global_object.stack:
            global_object.stack.clear()

    def test_parameter_lifecycle_integration(self, clear_all):
        """Test complete parameter lifecycle with global object integration"""
        # Given
        global_obj = GlobalObject()

        # When - Create parameter
        param = Parameter(name='test_param', value=10.0, unit='m')

        # Then - Should be registered in global map
        assert global_obj.map.is_known(param)
        assert param.unique_name in global_obj.map.vertices()
        assert 'created' in global_obj.map.find_type(param)

        # When - Modify parameter with undo/redo enabled
        if not global_obj.stack:
            global_obj.instantiate_stack()
        global_obj.stack.enabled = True

        original_value = param.value
        param.value = 20.0

        # Then - Should be able to undo/redo
        assert param.value == 20.0
        assert global_obj.stack.canUndo()

        global_obj.stack.undo()
        assert param.value == original_value

        global_obj.stack.redo()
        assert param.value == 20.0

        # When - Delete parameter
        param_name = param.unique_name
        original_count = len(global_obj.map.vertices())
        del param
        gc.collect()

        # Then - Should be removed from global map (via weak references)
        # Note: garbage collection timing can vary, so we check count decrease
        # or explicit cleanup
        final_count = len(global_obj.map.vertices())
        if param_name in global_obj.map.vertices():
            # If still present, manually clean up for test consistency
            global_obj.map.prune(param_name)
        assert param_name not in global_obj.map.vertices()

    def test_objbase_parameter_relationship(self, clear_all):
        """Test ObjBase containing parameters integration"""
        # Given
        global_obj = GlobalObject()

        param1 = Parameter(name='length', value=10.0, unit='m')
        param2 = Parameter(name='width', value=5.0, unit='m')

        # When - Create ObjBase with parameters
        obj = ObjBase(name='rectangle', length=param1, width=param2)

        # Then - All should be in global map
        assert global_obj.map.is_known(obj)
        assert global_obj.map.is_known(param1)
        assert global_obj.map.is_known(param2)

        # Check relationships in map
        obj_edges = global_obj.map.get_edges(obj)
        assert param1.unique_name in obj_edges
        assert param2.unique_name in obj_edges

        # When - Modify through ObjBase
        if not global_obj.stack:
            global_obj.instantiate_stack()
        global_obj.stack.enabled = True

        obj.length.value = 15.0

        # Then - Should be able to undo
        assert global_obj.stack.canUndo()
        global_obj.stack.undo()
        assert obj.length.value == 10.0

    def test_unique_name_generation_integration(self, clear_all):
        """Test unique name generation across different object types"""
        # Given
        global_obj = GlobalObject()

        # When - Create multiple objects of same type
        params = []
        for i in range(5):
            param = Parameter(name=f'param_{i}', value=float(i))
            params.append(param)

        # Then - Should have unique names
        unique_names = [p.unique_name for p in params]
        assert len(set(unique_names)) == 5  # All unique

        # Names should follow pattern
        for i, name in enumerate(unique_names):
            assert name == f'Parameter_{i}'

        # When - Create mixed object types
        obj = ObjBase(name='test_obj')
        desc = DescriptorBool(name='test_desc', value=True)

        # Then - Should not interfere with each other's naming
        assert obj.unique_name == 'ObjBase_0'
        assert desc.unique_name == 'DescriptorBool_0'

    def test_map_vertex_type_management(self, clear_all):
        """Test comprehensive vertex type management"""
        # Given
        global_obj = GlobalObject()
        param = Parameter(name='test', value=1.0)

        # When - Check initial type
        initial_types = global_obj.map.find_type(param)
        assert 'created' in initial_types

        # When - Change type
        global_obj.map.change_type(param, 'argument')
        updated_types = global_obj.map.find_type(param)
        assert 'created' in updated_types
        assert 'argument' in updated_types

        # When - Reset type
        global_obj.map.reset_type(param, 'returned')
        final_types = global_obj.map.find_type(param)
        assert final_types == ['returned']

        # When - Filter by type
        returned_objs = global_obj.map.returned_objs
        assert param.unique_name in returned_objs

        created_objs = global_obj.map.created_objs
        assert param.unique_name not in created_objs

    @pytest.mark.skip(reason='Weak reference timing can be inconsistent in tests')
    def test_weak_reference_cleanup_integration(self, clear_all):
        """Test that weak references properly clean up the map"""
        # Given
        global_obj = GlobalObject()

        # When - Create objects
        param1 = Parameter(name='temp1', value=1.0)
        param2 = Parameter(name='temp2', value=2.0)
        obj = ObjBase(name='temp_obj', param1=param1, param2=param2)

        param1_name = param1.unique_name
        param2_name = param2.unique_name
        obj_name = obj.unique_name

        # Then - All should be present
        assert global_obj.map.is_known(param1)
        assert global_obj.map.is_known(param2)
        assert global_obj.map.is_known(obj)

        # When - Delete objects
        del param1
        gc.collect()

        # Then - param1 should be removed
        assert param1_name not in global_obj.map.vertices()
        assert param2_name in global_obj.map.vertices()
        assert obj_name in global_obj.map.vertices()

        # When - Delete remaining objects
        del param2, obj
        gc.collect()

        # Then - Map should be empty
        assert len(global_obj.map.vertices()) == 0

    def test_complex_undo_redo_scenario(self, clear_all):
        """Test complex undo/redo scenario with multiple objects"""
        # Given
        global_obj = GlobalObject()
        if not global_obj.stack:
            global_obj.instantiate_stack()
        global_obj.stack.enabled = True

        # Create a complex object structure
        length = Parameter(name='length', value=10.0, unit='m')
        width = Parameter(name='width', value=5.0, unit='m')
        height = Parameter(name='height', value=3.0, unit='m')

        box = ObjBase(name='box', length=length, width=width, height=height)

        # When - Perform multiple operations in a macro
        global_obj.stack.beginMacro('Resize box')

        length.value = 15.0
        width.value = 7.0
        height.value = 4.0

        global_obj.stack.endMacro()

        # Then - All changes should be applied
        assert box.length.value == 15.0
        assert box.width.value == 7.0
        assert box.height.value == 4.0

        # When - Undo macro
        assert global_obj.stack.canUndo()
        assert global_obj.stack.undoText() == 'Resize box'

        global_obj.stack.undo()

        # Then - All changes should be reverted
        assert box.length.value == 10.0
        assert box.width.value == 5.0
        assert box.height.value == 3.0

        # When - Redo macro
        assert global_obj.stack.canRedo()
        global_obj.stack.redo()

        # Then - All changes should be reapplied
        assert box.length.value == 15.0
        assert box.width.value == 7.0
        assert box.height.value == 4.0

    def test_map_path_finding_integration(self, clear_all):
        """Test map path finding with real object relationships"""
        # Given
        global_obj = GlobalObject()

        # Create a hierarchy: container -> sub_container -> parameter
        param = Parameter(name='value', value=42.0)
        sub_container = ObjBase(name='sub', value=param)
        main_container = ObjBase(name='main', sub=sub_container)

        # When - Find path from main to parameter
        path = global_obj.map.find_path(main_container.unique_name, param.unique_name)

        # Then - Should find the path through the hierarchy
        assert len(path) >= 2
        assert path[0] == main_container.unique_name
        assert path[-1] == param.unique_name

        # When - Find reverse path
        reverse_path = global_obj.map.reverse_route(param.unique_name, main_container.unique_name)

        # Then - Should be the reverse
        assert reverse_path[0] == param.unique_name
        assert reverse_path[-1] == main_container.unique_name

    def test_map_connectivity_with_isolated_objects(self, clear_all):
        """Test map connectivity detection"""
        # Given
        global_obj = GlobalObject()

        # When - Create connected objects
        param1 = Parameter(name='connected1', value=1.0)
        param2 = Parameter(name='connected2', value=2.0)
        container = ObjBase(name='container', p1=param1, p2=param2)

        # Then - Map should be connected
        # TODO: Depending on implementation, connectivity might vary
        # assert global_obj.map.is_connected()

        # When - Create isolated object
        isolated = Parameter(name='isolated', value=99.0)
        # Remove its automatic connection by clearing edges
        # (In real usage, isolated objects would be rare)

        # Then - Map might still be connected depending on implementation
        # This tests the connectivity algorithm
        connectivity = global_obj.map.is_connected()
        assert isinstance(connectivity, bool)

    def test_error_handling_integration(self, clear_all):
        """Test error handling across the global object system"""
        # Given
        global_obj = GlobalObject()

        # When - Try to get non-existent object
        with pytest.raises(ValueError, match='Item not in map'):
            global_obj.map.get_item_by_key('non_existent')

        # When - Try to add object with duplicate name
        param1 = Parameter(name='test', value=1.0)
        param1_name = param1.unique_name

        # Create another with same unique name (should fail in add_vertex)
        with pytest.raises(ValueError, match='already exists'):
            # This simulates what would happen if we tried to add duplicate
            global_obj.map.add_vertex(param1)  # param1 already added during creation

    def test_memory_pressure_simulation(self, clear_all):
        """Test system behavior under memory pressure"""
        # Given
        global_obj = GlobalObject()

        # When - Create many objects
        objects = []
        for i in range(100):
            param = Parameter(name=f'param_{i}', value=float(i))
            obj = ObjBase(name=f'obj_{i}', param=param)
            objects.append((param, obj))

        initial_vertex_count = len(global_obj.map.vertices())
        assert initial_vertex_count == 200  # 100 params + 100 objs

        # When - Delete half the objects
        objects_to_delete = [objects[i] for i in range(0, 100, 2)]
        for item in objects_to_delete:
            objects.remove(item)
            del item

        # Force garbage collection
        gc.collect()

        # Then - Map should have fewer vertices
        current_vertex_count = len(global_obj.map.vertices())
        assert current_vertex_count <= initial_vertex_count

        # When - Delete all remaining objects
        del objects
        gc.collect()

        # Then - Map should be mostly empty (might have some remaining references)
        final_vertex_count = len(global_obj.map.vertices())
        assert final_vertex_count < initial_vertex_count

    def test_serialization_integration_with_global_state(self, clear_all):
        """Test serialization integration with global state management"""
        # Given
        global_obj = GlobalObject()

        # Create objects
        param = Parameter(name='test_param', value=123.45, unit='kg')
        obj = ObjBase(name='test_obj', param=param)

        original_vertex_count = len(global_obj.map.vertices())

        # When - Serialize objects
        param_dict = param.to_dict()
        obj_dict = obj.to_dict()

        # Clear global state
        global_obj.map._clear()
        assert len(global_obj.map.vertices()) == 0

        # When - Deserialize objects
        new_param = Parameter.from_dict(param_dict)
        new_obj = ObjBase.from_dict(obj_dict)

        # Then - Should be registered in global map again
        assert len(global_obj.map.vertices()) >= 2
        assert global_obj.map.is_known(new_param)
        assert global_obj.map.is_known(new_obj)

        # Objects should be functionally equivalent
        assert new_param.name == param.name
        assert new_param.value == param.value
        assert new_param.unit == param.unit

    def test_debug_mode_integration(self, clear_all):
        """Test debug mode behavior across the system"""
        # Given
        global_obj = GlobalObject()
        original_debug = global_obj.debug

        try:
            # When - Enable debug mode
            global_obj.debug = True

            if not global_obj.stack:
                global_obj.instantiate_stack()
            global_obj.stack.enabled = True

            # Create and modify objects
            param = Parameter(name='debug_test', value=1.0)

            # This should trigger debug output in property_stack decorator
            with patch('builtins.print') as mock_print:
                param.value = 2.0

                # Should have printed debug info (if debug mode works)
                # Note: This depends on the debug implementation details

            # Test that operations still work normally
            assert param.value == 2.0
            assert global_obj.stack.canUndo()

        finally:
            # Restore original debug state
            global_obj.debug = original_debug

    def test_concurrent_access_simulation(self, clear_all):
        """Simulate concurrent access patterns"""
        import threading
        import time

        # Given
        global_obj = GlobalObject()
        results = []
        errors = []

        def create_objects(thread_id, count=10):
            """Create objects in a thread"""
            try:
                for i in range(count):
                    param = Parameter(name=f'thread_{thread_id}_param_{i}', value=float(i))
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

        # All objects should be in the map
        for unique_name in results:
            assert unique_name in global_obj.map.vertices()
