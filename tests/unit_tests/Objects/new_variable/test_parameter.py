import pytest
from unittest.mock import MagicMock
import scipp as sc
import numpy as np

from scipp import UnitError

from easyscience.Objects.new_variable.parameter import Parameter
from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber
from easyscience.Objects.new_variable.descriptor_str import DescriptorStr
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

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "m", 0.01, -10, 20),  Parameter("name + test", 3, "m", 0.02, -10, 30),                Parameter("test + name", 3, "m", 0.02, -10, 30)),
            (Parameter("test", 2, "m", 0.01),           Parameter("name + test", 3, "m", 0.02, min=-np.Inf, max=np.Inf),Parameter("test + name", 3, "m", 0.02, min=-np.Inf, max=np.Inf)),
            (Parameter("test", 2, "cm", 0.01, -10, 10), Parameter("name + test", 1.02, "m", 0.010001, -0.1, 10.1),      Parameter("test + name", 102, "cm", 100.01, -10, 1010))],
            ids=["regular", "no_bounds", "unit_conversion"])
    def test_addition_with_parameter(self, parameter : Parameter, test : Parameter, expected : Parameter, expected_reverse : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter + test
        result_reverse = test + parameter

        # Expect
        assert result.name == expected.name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == expected_reverse.name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

        assert parameter.unit == "m"

    def test_addition_with_scalar(self):
        # When
        parameter = Parameter(name="name", value=1, variance=0.01, min=0, max=10)

        # Then
        result = parameter + 1.0
        result_reverse = 1.0 + parameter

        # Expect
        assert result.name == "name + 1.0"
        assert result.value == 2.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.01
        assert result.min == 1.0
        assert result.max == 11.0

        assert result_reverse.name == "1.0 + name"
        assert result_reverse.value == 2.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.01
        assert result_reverse.min == 1.0
        assert result_reverse.max == 11.0

    def test_addition_with_descriptor_number(self, parameter : Parameter):
        # When 
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name="test", value=1, variance=0.1, unit="cm")

        # Then
        result = parameter + descriptor_number
        result_reverse = descriptor_number + parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == "name + test"
        assert result.value == 1.01
        assert result.unit == "m"
        assert result.variance == 0.01001
        assert result.min == 0.01
        assert result.max == 10.01

        assert type(result_reverse) == Parameter
        assert result_reverse.name == "test + name"
        assert result_reverse.value == 101.0
        assert result_reverse.unit == "cm"
        assert result_reverse.variance == 100.1
        assert result_reverse.min == 1
        assert result_reverse.max == 1001

        assert parameter.unit == "m"
        assert descriptor_number.unit == "cm"

    @pytest.mark.parametrize("test", [1.0, Parameter("test", 2, "s",)], ids=["add_scalar_to_unit", "incompatible_units"])
    def test_addition_exception(self, parameter : Parameter, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = parameter + test
        with pytest.raises(UnitError):
            result_reverse = test + parameter
        
    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "m", 0.01, -20, 20),  Parameter("name - test", -1, "m", 0.02, -20, 30),                Parameter("test - name", 1, "m", 0.02, -30, 20)),
            (Parameter("test", 2, "m", 0.01),           Parameter("name - test", -1, "m", 0.02, min=-np.Inf, max=np.Inf),Parameter("test - name", 1, "m", 0.02, min=-np.Inf, max=np.Inf)),
            (Parameter("test", 2, "cm", 0.01, -10, 10), Parameter("name - test", 0.98, "m", 0.010001, -0.1, 10.1),       Parameter("test - name", -98, "cm", 100.01, -1010, 10))],
            ids=["regular", "no_bounds", "unit_conversion"])
    def test_subtraction_with_parameter(self, parameter : Parameter, test : Parameter, expected : Parameter, expected_reverse : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter - test
        result_reverse = test - parameter

        # Expect
        assert result.name == expected.name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == expected_reverse.name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

        assert parameter.unit == "m"

    def test_subtraction_with_parameter_nan_cases(self):
        # When
        parameter = Parameter(name="name", value=1, variance=0.01, min=-np.Inf, max=np.Inf)
        test = Parameter(name="test", value=2, variance=0.01, min=-np.Inf, max=np.Inf)

        # Then
        result = parameter - test
        result_reverse = test - parameter

        # Expect
        assert result.name == "name - test"
        assert result.value == -1.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.02
        assert result.min == -np.Inf
        assert result.max == np.Inf

        assert result_reverse.name == "test - name"
        assert result_reverse.value == 1.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.02
        assert result_reverse.min == -np.Inf
        assert result_reverse.max == np.Inf

    def test_subtraction_with_scalar(self):
        # When
        parameter = Parameter(name="name", value=2, variance=0.01, min=0, max=10)

        # Then
        result = parameter - 1.0
        result_reverse = 1.0 - parameter

        # Expect
        assert result.name == "name - 1.0"
        assert result.value == 1.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.01
        assert result.min == -1.0
        assert result.max == 9.0

        assert result_reverse.name == "1.0 - name"
        assert result_reverse.value == -1.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.01
        assert result_reverse.min == -9.0
        assert result_reverse.max == 1.0

    def test_subtraction_with_descriptor_number(self, parameter : Parameter):
        # When 
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name="test", value=1, variance=0.1, unit="cm")

        # Then
        result = parameter - descriptor_number
        result_reverse = descriptor_number - parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == "name - test"
        assert result.value == 0.99
        assert result.unit == "m"
        assert result.variance == 0.01001
        assert result.min == -0.01
        assert result.max == 9.99

        assert type(result_reverse) == Parameter
        assert result_reverse.name == "test - name"
        assert result_reverse.value == -99.0
        assert result_reverse.unit == "cm"
        assert result_reverse.variance == 100.1
        assert result_reverse.min == -999
        assert result_reverse.max == 1

        assert parameter.unit == "m"
        assert descriptor_number.unit == "cm"

    @pytest.mark.parametrize("test", [1.0, Parameter("test", 2, "s",)], ids=["sub_scalar_to_unit", "incompatible_units"])
    def test_subtraction_exception(self, parameter : Parameter, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = parameter - test
        with pytest.raises(UnitError):
            result_reverse = test - parameter

        # parameter = Parameter(
        #     name="name",
        #     value=1,
        #     unit="m",
        #     variance=0.01,
        #     min=0,
        #     max=10,
        #     description="description",
        #     url="url",
        #     display_name="display_name",
        #     callback=self.mock_callback,
        #     enabled="enabled",
        #     parent=None,

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "m", 0.01, -10, 20),    Parameter("name * test", 2, "m^2", 0.05, -100, 200),               Parameter("test * name", 2, "m^2", 0.05, -100, 200)),
            (Parameter("test", 2, "m", 0.01),             Parameter("name * test", 2, "m^2", 0.05, min=-np.Inf, max=np.Inf), Parameter("test * name", 2, "m^2", 0.05, min=-np.Inf, max=np.Inf)),
            (Parameter("test", 2, "dm", 0.01, -10, 20),   Parameter("name * test", 0.2, "m^2", 0.0005, -10, 20),             Parameter("test * name", 0.2, "m^2", 0.0005, -10, 20)),
            (Parameter("test", 2, "1/dm", 0.01, -10, 20), Parameter("name * test", 20.0, "dimensionless", 5, -1000, 2000),   Parameter("test * name", 20.0, "dimensionless", 5, -1000, 2000))],
            ids=["regular", "no_bounds", "base_unit_conversion", "base_unit_conversion_dimensionless"])
    def test_multiplication_with_parameter(self, parameter : Parameter, test : Parameter, expected : Parameter, expected_reverse : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter * test
        result_reverse = test * parameter

        # Expect
        assert result.name == expected.name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == pytest.approx(expected.variance)
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == expected_reverse.name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == pytest.approx(expected_reverse.variance)
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

    def test_multiplication_with_parameter_nan_cases(self):
        # When
        parameter = Parameter(name="name", value=1, variance=0.01, min=-np.Inf, max=np.Inf)
        test = Parameter(name="test", value=0, variance=0.01, min=0, max=0)

        # Then
        result = parameter * test
        result_reverse = test * parameter

        # Expect
        assert result.name == "name * test"
        assert result.value == 0
        assert result.unit == "dimensionless"
        assert result.variance == 0.01
        assert result.min == -np.Inf
        assert result.max == np.Inf

        assert result_reverse.name == "test * name"
        assert result_reverse.value == 0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.01
        assert result_reverse.min == -np.Inf
        assert result_reverse.max == np.Inf

    def test_multiplication_with_descriptor_number(self, parameter : Parameter):
        # When 
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name="test", value=2, variance=0.1, unit="cm")

        # Then
        result = parameter * descriptor_number
        result_reverse = descriptor_number * parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == "name * test"
        assert result.value == 2
        assert result.unit == "dm^2"
        assert result.variance == 0.14
        assert result.min == 0
        assert result.max == 20

        assert type(result_reverse) == Parameter
        assert result_reverse.name == "test * name"
        assert result_reverse.value == 2
        assert result_reverse.unit == "dm^2"
        assert result_reverse.variance == 0.14
        assert result_reverse.min == 0
        assert result_reverse.max == 20

    def test_multiplication_with_scalar(self, parameter : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter * 2
        result_reverse = 2 * parameter

        # Expect
        assert result.name == "name * 2"
        assert result.value == 2.0
        assert result.unit == "m"
        assert result.variance == 0.04
        assert result.min == 0.0
        assert result.max == 20.0

        assert result_reverse.name == "2 * name"
        assert result_reverse.value == 2.0
        assert result_reverse.unit == "m"
        assert result_reverse.variance == 0.04
        assert result_reverse.min == 0.0
        assert result_reverse.max == 20.0