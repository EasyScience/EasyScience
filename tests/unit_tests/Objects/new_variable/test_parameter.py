import pytest
from unittest.mock import MagicMock
import scipp as sc

from easyscience.Objects.new_variable.parameter import Parameter


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

    def test_init(self, parameter: Parameter):
        # When Then Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == "m"
        assert parameter._max.value == 10
        assert parameter._max.unit == "m"

        # From super
        assert parameter._value.value == 1
        assert parameter._value.unit == "m"
        assert parameter._value.variance == 0.01
        assert parameter._name == "name"
        assert parameter._description == "description"
        assert parameter._url == "url"
        assert parameter._display_name == "display_name"
        assert parameter._callback == self.mock_callback
        assert parameter._enabled == "enabled"

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

    def test_init_variance_exception(self):
        # When 
        mock_callback = MagicMock()
        variance = -1

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name="name",
                value=1,
                unit="m",
                variance=variance,
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

    def test_error(self, parameter: Parameter):
        # When Then Expect
        assert parameter.error == 0.1