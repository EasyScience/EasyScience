# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import pytest

from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.base_classes import SerializerBase
from easyscience.variable import DescriptorStr


class MockModelComponent(ModelBase):
    """
    A simple mock model component with some parameters and descriptors.
    """

    def __init__(self, display_name=None, unique_name=None, temperature=0, room_temperature=22):
        super().__init__(display_name=display_name, unique_name=unique_name)
        self._temperature = Parameter(display_name='temperature', value=temperature)
        self._room_temperature = DescriptorNumber(
            display_name='room_temperature', value=room_temperature
        )
        self._status = DescriptorStr(display_name='status', value='OK')

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature.value = value

    @property
    def room_temperature(self):
        return self._room_temperature

    @room_temperature.setter
    def room_temperature(self, value):
        self._room_temperature.value = value

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status.value = value


class MockModelFull(ModelBase):
    """
    A mock model that contains another model as a component. To test building nested models using the ModelBase class.
    """

    def __init__(self, component=None, display_name=None, unique_name=None, pressure=0, area=1):
        super().__init__(display_name=display_name, unique_name=unique_name)
        self._pressure = Parameter(display_name='pressure', value=pressure)
        self._area = Parameter(display_name='area', value=area)
        if component is not None:
            self._component = component
        else:
            self._component = MockModelComponent(temperature=25, room_temperature=22)

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        self._pressure.value = value

    @property
    def component(self):
        return self._component

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        self._area.value = value


class TestModelBase:
    @pytest.fixture
    def nested_model(self, monkeypatch):
        model = MockModelFull(pressure=1)
        model.pressure.make_dependent_on(
            dependency_expression='2 * temperature + 5',
            dependency_map={'temperature': model.component.temperature},
        )
        model.area.fixed = True
        monkeypatch.setattr(
            model.component,
            'get_all_variables',
            MagicMock(wraps=model.component.get_all_variables),
        )
        return model

    @pytest.fixture
    def clear(self):
        # Clear the global object map before each test
        global_object.map._clear()

    def test_init(self):
        # When Then
        model = ModelBase(unique_name='test_model', display_name='Test Model')
        # Expect
        assert model.unique_name == 'test_model'
        assert model.display_name == 'Test Model'

    def test_get_all_variables_flat(self):
        # When
        model = MockModelComponent(temperature=25, room_temperature=22)
        # Then
        vars = model.get_all_variables()
        # Expect
        assert len(vars) == 3
        assert any(isinstance(p, Parameter) for p in vars)
        assert any(type(p) is DescriptorNumber for p in vars)
        assert any(type(p) is DescriptorStr for p in vars)

    def test_get_all_variables_nested(self, nested_model):
        # When Then
        vars = nested_model.get_all_variables()
        # Expect
        assert len(vars) == 5
        assert any(isinstance(p, Parameter) and not p.independent for p in vars)
        assert any(isinstance(p, Parameter) and p.fixed for p in vars)
        assert any(isinstance(p, Parameter) and p.independent and not p.fixed for p in vars)
        assert any(type(p) is DescriptorNumber for p in vars)
        assert any(type(p) is DescriptorStr for p in vars)
        assert nested_model.component.get_all_variables.call_count == 1

    def test_get_all_parameters_nested(self, nested_model):
        # When Then
        params = nested_model.get_all_parameters()
        # Expect
        assert len(params) == 3
        assert any(isinstance(p, Parameter) and not p.independent for p in params)
        assert any(isinstance(p, Parameter) and p.fixed for p in params)
        assert any(isinstance(p, Parameter) and p.independent and not p.fixed for p in params)
        assert nested_model.component.get_all_variables.call_count == 1

    def test_get_fittable_parameters_nested(self, nested_model):
        # When Then
        fittable_params = nested_model.get_fittable_parameters()
        # Expect
        assert len(fittable_params) == 2
        assert any(isinstance(p, Parameter) and p.fixed for p in fittable_params)
        assert any(
            isinstance(p, Parameter) and p.independent and not p.fixed for p in fittable_params
        )
        assert nested_model.component.get_all_variables.call_count == 1

    def test_get_fit_parameters_alias(self, monkeypatch):
        # When
        model = MockModelFull(pressure=1)
        monkeypatch.setattr(model, 'get_free_parameters', MagicMock())
        # Then
        fit_params = model.get_fit_parameters()
        # Expect
        assert model.get_free_parameters.call_count == 1

    def test_get_free_parameters_nested(self, nested_model):
        # When Then
        free_params = nested_model.get_free_parameters()
        # Expect
        assert len(free_params) == 1
        assert any(isinstance(p, Parameter) and p.independent and not p.fixed for p in free_params)
        assert nested_model.component.get_all_variables.call_count == 1

    def test_from_dict(self, monkeypatch, clear):
        # When
        model = MockModelComponent()
        obj_dict = (
            model.to_dict()
        )  # We only care about deserializing dictorionaries currently created by EasyScience
        obj_dict['@module'] = 'easyscience.base_classes.model_base'
        monkeypatch.setattr(
            SerializerBase,
            '_import_class',
            MagicMock(
                side_effect=lambda module_name, class_name: (
                    MockModelComponent
                    if class_name == 'MockModelComponent'
                    else Parameter
                    if class_name == 'Parameter'
                    else DescriptorNumber
                )
            ),
        )
        global_object.map._clear()
        # Then
        new_model = MockModelComponent.from_dict(obj_dict)
        # Expect
        assert isinstance(new_model, MockModelComponent)
        assert new_model._temperature.value == 0.0
        assert new_model._room_temperature.value == 22.0
        assert isinstance(new_model._temperature, Parameter)
        assert isinstance(new_model._room_temperature, DescriptorNumber)
        assert len(global_object.map.vertices()) == 4
        assert global_object.map.get_item_by_key('Parameter_0') is new_model._temperature
        assert (
            len([param for param in global_object.map.vertices() if param.startswith('Parameter')])
            == 1
        )

    def test_from_dict_nested(self, monkeypatch, clear):
        # When
        model = MockModelFull()
        model.__class__.__module__ = 'easyscience'  # Ensure mock class is seen as easyscience
        model.component.__class__.__module__ = (
            'easyscience'  # Ensure nested class is also seen as easyscience
        )
        obj_dict = (
            model.to_dict()
        )  # We only care about deserializing dictorionaries currently created by EasyScience
        monkeypatch.setattr(
            SerializerBase,
            '_import_class',
            MagicMock(
                side_effect=lambda module_name, class_name: (
                    MockModelFull
                    if class_name == 'MockModelFull'
                    else MockModelComponent
                    if class_name == 'MockModelComponent'
                    else Parameter
                    if class_name == 'Parameter'
                    else DescriptorNumber
                )
            ),
        )
        global_object.map._clear()
        # Then
        new_model = MockModelFull.from_dict(obj_dict)
        # Expect
        assert isinstance(new_model, MockModelFull)
        assert isinstance(new_model.component, MockModelComponent)
        assert new_model._pressure.value == 0.0
        assert new_model._area.value == 1.0
        assert new_model.component._temperature.value == 25.0
        assert new_model.component._room_temperature.value == 22.0
        assert isinstance(new_model._pressure, Parameter)
        assert isinstance(new_model._area, Parameter)
        assert isinstance(new_model.component._temperature, Parameter)
        assert isinstance(new_model.component._room_temperature, DescriptorNumber)
        assert len(global_object.map.vertices()) == 7
        assert global_object.map.get_item_by_key('Parameter_0') in [
            new_model._pressure,
            new_model._area,
            new_model.component._temperature,
        ]
        assert (
            len([param for param in global_object.map.vertices() if param.startswith('Parameter')])
            == 3
        )

    def test_from_dict_not_easyscience(self):
        # When
        obj_dict = {
            '@module': 'some.other.module',
            '@class': 'NotAnEasyScienceObject',
            'some_property': 42,
        }
        # Then / Expect
        with pytest.raises(
            ValueError, match='Input must be a dictionary representing an EasyScience object.'
        ):
            MockModelComponent.from_dict(obj_dict)

    def test_from_dict_wrong_class(self):
        # When
        obj_dict = {
            '@module': 'easyscience.base_classes.model_base',
            '@class': 'SomeOtherClass',
            'some_property': 42,
        }
        # Then / Expect
        with pytest.raises(
            ValueError,
            match='Class name in dictionary does not match the expected class: MockModelComponent.',
        ):
            MockModelComponent.from_dict(obj_dict)

    def test_from_dict_parameter_setting_failure(self, monkeypatch, clear):
        # When
        model = MockModelComponent()
        model.temperature.min = -100
        obj_dict = (
            model.to_dict()
        )  # We only care about deserializing dictorionaries currently created by EasyScience
        obj_dict['@module'] = 'easyscience.base_classes.model_base'
        monkeypatch.setattr(
            SerializerBase,
            '_import_class',
            MagicMock(
                side_effect=lambda module_name, class_name: (
                    MockModelComponent
                    if class_name == 'MockModelComponent'
                    else Parameter
                    if class_name == 'Parameter'
                    else DescriptorNumber
                )
            ),
        )
        monkeypatch.setattr(
            MockModelComponent,
            'temperature',
            property(lambda self: (_ for _ in ()).throw(Exception('Simulated failure'))),
        )
        global_object.map._clear()
        # Then Expect
        with pytest.raises(
            SyntaxError,
            match='Could not set parameter temperature during `from_dict` with full deserialized variable.',
        ):
            MockModelComponent.from_dict(obj_dict)
