# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
import pytest

from easyscience import Parameter
from easyscience import global_object

NORMAL = {'distribution': 'norm', 'loc': 1.0, 'scale': 0.2}


class TestPrior:
    def test_the_default_is_no_prior(self):
        # When Then
        parameter = Parameter(1.0, min=0.0, max=2.0)

        # Expect
        assert parameter.prior is None
        assert parameter.prior_distribution.is_uniform
        assert parameter.prior_distribution.lower == 0.0

    def test_a_prior_is_given_as_plain_data(self):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0, prior=NORMAL)

        # Then
        distribution = parameter.prior_distribution

        # Expect
        assert parameter.prior == NORMAL
        assert not distribution.is_uniform
        assert distribution.upper == 2.0

    def test_the_prior_is_a_copy(self):
        # When
        parameter = Parameter(1.0, prior=NORMAL)

        # Then
        parameter.prior['loc'] = 99.0

        # Expect
        assert parameter.prior == NORMAL

    def test_setting_and_removing_the_prior(self):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0)

        # Then
        parameter.prior = NORMAL
        with_prior = parameter.prior
        parameter.prior = None

        # Expect
        assert with_prior == NORMAL
        assert parameter.prior is None

    @pytest.mark.parametrize(
        ('spec', 'error'),
        [
            ('norm', TypeError),
            ({'loc': 1.0}, TypeError),
            ({'distribution': 'norm', 'loc': 'a'}, TypeError),
            ({'distribution': 'unknown'}, ValueError),
            ({'distribution': 'norm', 'scale': -1.0}, ValueError),
        ],
    )
    def test_an_invalid_prior_is_rejected(self, spec, error):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0)

        # Then Expect
        with pytest.raises(error):
            parameter.prior = spec
        with pytest.raises(error):
            Parameter(1.0, prior=spec)
        assert parameter.prior is None

    def test_a_prior_without_probability_inside_the_bounds_is_rejected(self):
        # When
        parameter = Parameter(55.0, min=50.0, max=60.0)

        # Then Expect
        with pytest.raises(ValueError, match='no probability'):
            parameter.prior = {'distribution': 'norm', 'loc': 0.0, 'scale': 1e-3}

    def test_a_prior_without_probability_inside_the_bounds_is_rejected_at_construction(self):
        # When Then Expect
        with pytest.raises(ValueError, match='no probability'):
            Parameter(
                55.0,
                min=50.0,
                max=60.0,
                prior={'distribution': 'norm', 'loc': 0.0, 'scale': 1e-3},
            )

    def test_the_prior_follows_changed_bounds(self):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0, prior=NORMAL)

        # Then
        parameter.max = 3.0

        # Expect
        assert parameter.prior_distribution.upper == 3.0

    def test_a_dependent_parameter_has_no_prior(self):
        # When
        a = Parameter(2.0, display_name='a')
        dependent = Parameter.from_dependency(
            dependency_expression='2 * a', dependency_map={'a': a}
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent.prior = NORMAL

    def test_undo_and_redo(self):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0)
        global_object.stack.enabled = True
        try:
            parameter.prior = NORMAL

            # Then
            global_object.stack.undo()
            undone = parameter.prior
            global_object.stack.redo()
            redone = parameter.prior
        finally:
            global_object.stack.enabled = False
            global_object.stack.clear()

        # Expect
        assert undone is None
        assert redone == NORMAL

    def test_serialization_and_copies_keep_the_prior(self):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0, prior=NORMAL)

        # Then
        restored = Parameter.from_dict(parameter.to_dict())

        # Expect
        assert parameter.to_dict()['prior'] == NORMAL
        assert restored.prior == NORMAL
        assert copy.copy(parameter).prior == NORMAL
        assert copy.deepcopy(parameter).prior == NORMAL

    def test_a_file_without_a_prior_still_loads(self):
        # When
        data = Parameter(1.0, min=0.0, max=2.0).to_dict()
        data.pop('prior')

        # Then
        restored = Parameter.from_dict(data)

        # Expect
        assert restored.prior is None

    def test_results_of_a_calculation_have_no_prior(self):
        # When
        parameter = Parameter(1.0, min=0.0, max=2.0, prior=NORMAL)

        # Then
        result = parameter * 2

        # Expect
        assert result.prior is None

    def test_the_prior_transform_uses_the_truncated_prior(self):
        # When
        parameter = Parameter(1.0, min=0.5, max=2.0, prior=NORMAL)

        # Then
        values = parameter.prior_distribution.ppf(np.linspace(0, 1, 11))

        # Expect
        assert values.min() >= 0.5
        assert values.max() <= 2.0
