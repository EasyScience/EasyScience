import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.parameter import Parameter
from easyscience import global_object

class TestParameter:
    @pytest.fixture
    def parameter(self) -> Parameter:
        self.mock_callback = MagicMock()
        parameter = Parameter(
            name="name",
            value=1,
            unit="m",
            variance=0.01,
            min=0,
            max=10,
            description="description",
            url="url",
            display_name="display_name",
            callback=self.mock_callback,
            enabled="enabled",
            parent=None,
        )
        return parameter

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, parameter: Parameter):
        # When Then Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == "m"
        assert parameter._max.value == 10
        assert parameter._max.unit == "m"
        assert parameter._callback == self.mock_callback
        assert parameter._enabled == "enabled"

        # From super
        assert parameter._scalar.value == 1
        assert parameter._scalar.unit == "m"
        assert parameter._scalar.variance == 0.01
        assert parameter._name == "name"
        assert parameter._description == "description"
        assert parameter._url == "url"
        assert parameter._display_name == "display_name"

    def test_init_value_min_exception(self):
        # When 
        mock_callback = MagicMock()
        value = -1

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name="name",
                value=value,
                unit="m",
                variance=0.01,
                min=0,
                max=10,
                description="description",
                url="url",
                display_name="display_name",
                callback=mock_callback,
                enabled="enabled",
                parent=None,
            )

    def test_init_value_max_exception(self):
        # When 
        mock_callback = MagicMock()
        value = 100

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name="name",
                value=value,
                unit="m",
                variance=0.01,
                min=0,
                max=10,
                description="description",
                url="url",
                display_name="display_name",
                callback=mock_callback,
                enabled="enabled",
                parent=None,
            )

    def test_min(self, parameter: Parameter):
        # When Then Expect
        assert parameter.min == 0

    def test_set_min(self, parameter: Parameter):
        # When Then 
        parameter.min = 0.1

        # Expect
        assert parameter.min == 0.1

    def test_set_min_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.min = 10

    def test_set_max(self, parameter: Parameter):
        # When Then 
        parameter.max = 10

        # Expect
        assert parameter.max == 10

    def test_set_max_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.max = 0.1

    def test_convert_unit(self, parameter: Parameter):
        # When Then
        parameter.convert_unit("mm")

        # Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == "mm"
        assert parameter._max.value == 10000
        assert parameter._max.unit == "mm"

    def test_fixed(self, parameter: Parameter):
        # When Then Expect
        assert parameter.fixed == False

    def test_set_fixed(self, parameter: Parameter):
        # When Then 
        parameter.fixed = True

        # Expect
        assert parameter.fixed == True

    @pytest.mark.parametrize("fixed", ["True", 1])
    def test_set_fixed_exception(self, parameter: Parameter, fixed):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.fixed = fixed

    def test_error(self, parameter: Parameter):
        # When Then Expect
        assert parameter.error == 0.1
    
    def test_set_error(self, parameter: Parameter):
        # When 
        parameter.error = 10

        # Then Expect
        assert parameter.error == 10
        assert parameter._scalar.variance == 100

    def test_set_error_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.error = -0.1

    def test_float(self, parameter: Parameter):
        # When Then Expect
        assert float(parameter) == 1.0

    def test_repr(self, parameter: Parameter):
        # When Then Expect
        assert repr(parameter) == "<Parameter 'name': 1.0000 ± 0.1000 m, bounds=[0.0:10.0]>"

    def test_repr_fixed(self, parameter: Parameter):
        # When 
        parameter.fixed = True

        # Then Expect
        assert repr(parameter) == "<Parameter 'name': 1.0000 ± 0.1000 m (fixed), bounds=[0.0:10.0]>"

    def test_bounds(self, parameter: Parameter):
        # When Then Expect
        assert parameter.bounds == (0, 10)
    
    def test_set_bounds(self, parameter: Parameter):
        # When 
        parameter._enabled = False
        parameter._fixed = True

        # Then 
        parameter.bounds = (-10, 5)

        # Expect
        assert parameter.min == -10
        assert parameter.max == 5
        assert parameter._enabled == True
        assert parameter._fixed == False

    def test_set_bounds_exception_min(self, parameter: Parameter):
        # When 
        parameter._enabled = False
        parameter._fixed = True

        # Then
        with pytest.raises(ValueError):
            parameter.bounds = (2, 10)

        # Expect
        assert parameter.min == 0
        assert parameter.max == 10
        assert parameter._enabled == False
        assert parameter._fixed == True

    def test_set_bounds_exception_max(self, parameter: Parameter):
        # When 
        parameter._enabled = False
        parameter._fixed = True

        # Then
        with pytest.raises(ValueError):
            parameter.bounds = (0, 0.1)

        # Expect
        assert parameter.min == 0
        assert parameter.max == 10
        assert parameter._enabled == False
        assert parameter._fixed == True

    def test_enabled(self, parameter: Parameter):
        # When
        parameter._enabled = True
        
        # Then Expect
        assert parameter.enabled is True

    def test_set_enabled(self, parameter: Parameter):
        # When
        parameter.enabled = False

        # Then Expect
        assert parameter._enabled is False

    def test_value_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = 1.0

        # Then Expect
        assert parameter.value == 1.0
        assert parameter._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = 2.0

        # Then Expect
        assert parameter.value == 2.0
        assert parameter._callback.fget.call_count == 1

    def test_set_value(self, parameter: Parameter):
        # When
        # First call returns 1.0 that is used to enforce the undo/redo functionality to register the value has changed
        # Second and third call returns 2.0 is used in the constraint check
        self.mock_callback.fget.side_effect = [1.0, 2.0, 2.0]

        # Then
        parameter.value = 2

        # Expect
        parameter._callback.fset.assert_called_with(2)
        assert parameter._callback.fset.call_count == 1
        assert parameter._scalar == sc.scalar(2, unit='m')

    def test_full_value_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then Expect
        assert parameter.full_value == sc.scalar(1, unit='m')
        assert parameter._callback.fget.call_count == 1
        
    def test_full_value_no_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

        # Then Expect
        assert parameter.full_value == sc.scalar(2, unit='m')
        assert parameter._callback.fget.call_count == 1

    def test_set_full_value(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then
        parameter.full_value = sc.scalar(2, unit='m')

        # Expect
        parameter._callback.fset.assert_called_once_with(sc.scalar(2, unit='m')) 
        assert parameter._scalar == sc.scalar(2, unit='m')
    
    def test_copy(self, parameter: Parameter):
        # When Then
        parameter_copy = parameter.__copy__()

        # Expect
        assert type(parameter_copy) == Parameter
        assert id(parameter_copy._scalar) != id(parameter._scalar)
        assert isinstance(parameter_copy._callback, property)

        assert parameter_copy._name == parameter._name
        assert parameter_copy._scalar == parameter._scalar
        assert parameter_copy._min == parameter._min
        assert parameter_copy._max == parameter._max
        assert parameter_copy._fixed == parameter._fixed
        assert parameter_copy._description == parameter._description
        assert parameter_copy._url == parameter._url
        assert parameter_copy._display_name == parameter._display_name
        assert parameter_copy._enabled == parameter._enabled

    def test_as_data_dict(self, clear, parameter: Parameter):
        # When Then
        parameter_dict = parameter.as_data_dict()

        # Expect
        assert parameter_dict == {
            "name": "name",
            "value": 1.0,
            "unit": "m",
            "variance": 0.01,
            "min": 0,
            "max": 10,
            "fixed": False,
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "enabled": "enabled",
            "unique_name": "Parameter_0",
        }