# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import copy
import gc
import pickle

import numpy as np
import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.variable.covariance import ParameterCovariance

COVARIANCE = np.array([[0.04, 0.03], [0.03, 0.09]])


@pytest.fixture
def pair():
    first = Parameter(1.0, display_name='first')
    second = Parameter(2.0, display_name='second')
    return first, second


class TestConstruction:
    def test_stores_the_matrix_and_labels(self, pair):
        # When Then
        record = ParameterCovariance(pair, COVARIANCE)

        # Expect
        np.testing.assert_allclose(record.covariance, COVARIANCE)
        assert record.names == ('first', 'second')
        assert record.source == 'hessian'
        assert len(record) == 2
        assert pair[0] in record

    def test_the_matrix_is_read_only_and_a_copy(self, pair):
        # When
        matrix = COVARIANCE.copy()
        record = ParameterCovariance(pair, matrix)
        matrix[0, 0] = 99.0

        # Then Expect
        assert record.covariance[0, 0] == 0.04
        with pytest.raises(ValueError):
            record.covariance[0, 0] = 1.0

    @pytest.mark.parametrize(
        'matrix',
        [np.eye(3), np.array([[1.0, 0.0], [0.5, 1.0]]), np.array([[np.nan, 0], [0, 1.0]])],
    )
    def test_rejects_a_bad_matrix(self, pair, matrix):
        # When Then Expect
        with pytest.raises(ValueError):
            ParameterCovariance(pair, matrix)

    def test_rejects_a_negative_variance(self, pair):
        # When Then Expect
        with pytest.raises(ValueError, match='negative variance'):
            ParameterCovariance(pair, np.diag([1.0, -1.0]))

    def test_rejects_an_unknown_source(self, pair):
        # When Then Expect
        with pytest.raises(ValueError, match='source'):
            ParameterCovariance(pair, COVARIANCE, source='guess')

    def test_rejects_duplicates_and_non_parameters(self, pair):
        # When Then Expect
        with pytest.raises(ValueError, match='distinct'):
            ParameterCovariance([pair[0], pair[0]], COVARIANCE)
        with pytest.raises(TypeError):
            ParameterCovariance([pair[0], 1.0], COVARIANCE)

    def test_rejects_a_dependent_parameter(self, pair):
        # When
        first, second = pair
        dependent = Parameter.from_dependency(
            display_name='dep',
            dependency_expression='a',
            dependency_map={'a': first},
        )

        # Then Expect
        with pytest.raises(ValueError, match='dependent'):
            ParameterCovariance([first, dependent], COVARIANCE)

    def test_from_samples_uses_the_column_order(self, pair):
        # When
        rng = np.random.default_rng(0)
        samples = rng.multivariate_normal([1.0, 2.0], COVARIANCE, size=20000)

        # Then
        record = ParameterCovariance.from_samples(pair, samples)

        # Expect
        np.testing.assert_allclose(record.covariance, COVARIANCE, atol=3e-3)
        assert record.source == 'posterior'
        assert record.samples.shape == (20000, 2)


class TestDerivedValues:
    def test_std_and_correlation(self, pair):
        # When Then
        record = ParameterCovariance(pair, COVARIANCE)

        # Expect
        np.testing.assert_allclose(record.std, [0.2, 0.3])
        np.testing.assert_allclose(record.correlation, [[1.0, 0.5], [0.5, 1.0]])
        assert record.correlation_of(pair[0], pair[1]) == pytest.approx(0.5)
        assert record.correlation_of(pair[0], pair[0]) == 1.0
        assert record.covariance_of(pair[0], pair[1]) == pytest.approx(0.03)

    def test_a_parameter_without_variance_is_uncorrelated(self, pair):
        # When Then
        record = ParameterCovariance(pair, np.diag([0.0, 1.0]))

        # Expect
        np.testing.assert_allclose(record.correlation, np.eye(2))

    def test_unknown_parameters(self, pair):
        # When
        record = ParameterCovariance(pair, COVARIANCE)
        stranger = Parameter(0.0, display_name='stranger')

        # Then Expect
        assert record.index_of(stranger) is None
        assert record.correlation_of(pair[0], stranger) == 0.0
        assert record.covariance_of(pair[0], stranger) is None

    def test_submatrix_follows_the_requested_order(self, pair):
        # When
        record = ParameterCovariance(pair, COVARIANCE)

        # Then
        reordered = record.submatrix([pair[1], pair[0]])

        # Expect
        np.testing.assert_allclose(reordered, [[0.09, 0.03], [0.03, 0.04]])
        with pytest.raises(KeyError):
            record.submatrix([Parameter(0.0, display_name='stranger')])

    def test_scaled_is_a_new_unattached_record(self, pair):
        # When
        record = ParameterCovariance(pair, COVARIANCE).attach()

        # Then
        scaled = record.scaled(4.0)

        # Expect
        np.testing.assert_allclose(scaled.covariance, 4 * COVARIANCE)
        assert pair[0]._covariance_record is record
        with pytest.raises(ValueError):
            record.scaled(-1.0)


class TestAttachment:
    def test_attach_makes_it_the_covariance_of_the_parameters(self, pair):
        # When
        first, second = pair
        record = ParameterCovariance(pair, COVARIANCE)
        assert first.covariance is None

        # Then
        record.attach()

        # Expect
        assert first.covariance is record
        assert second.covariance is record
        assert record.is_current

    def test_a_new_record_replaces_the_old_one(self, pair):
        # When
        first, second = pair
        old = ParameterCovariance(pair, COVARIANCE).attach()

        # Then
        new = ParameterCovariance(pair, 2 * COVARIANCE).attach()

        # Expect
        assert first.covariance is new
        assert old.covariance[0, 0] == 0.04

    def test_detach(self, pair):
        # When
        record = ParameterCovariance(pair, COVARIANCE).attach()

        # Then
        record.detach()

        # Expect
        assert pair[0].covariance is None
        assert not record.is_current or pair[0]._covariance_record is None

    @pytest.mark.parametrize('change', ['value', 'variance', 'error', 'fixed'])
    def test_any_change_makes_the_record_stale(self, pair, change):
        # When
        first, _ = pair
        first.error = 0.2
        ParameterCovariance(pair, COVARIANCE).attach()
        assert first.covariance is not None

        # Then
        if change == 'fixed':
            first.fixed = True
        elif change == 'value':
            first.value = 1.5
        elif change == 'variance':
            first.variance = 0.5
        else:
            first.error = 0.7

        # Expect
        assert first.covariance is None
        assert pair[1].covariance is None

    def test_restoring_the_value_makes_it_current_again(self, pair):
        # When
        first, _ = pair
        ParameterCovariance(pair, COVARIANCE).attach()
        first.value = 1.5
        assert first.covariance is None

        # Then
        first.value = 1.0

        # Expect
        assert first.covariance is not None

    def test_undo_and_redo_of_a_change(self, pair):
        # When
        first, _ = pair
        ParameterCovariance(pair, COVARIANCE).attach()
        global_object.stack.enabled = True
        try:
            first.value = 1.5
            assert first.covariance is None

            # Then
            global_object.stack.undo()
            restored = first.covariance
            global_object.stack.redo()
            changed = first.covariance
        finally:
            global_object.stack.enabled = False
            global_object.stack.clear()

        # Expect
        assert restored is not None
        assert changed is None


class TestStaleness:
    def test_restoring_the_error_keeps_the_record_current(self, pair):
        # When
        first, _ = pair
        first.variance = 0.0123456789
        ParameterCovariance(pair, COVARIANCE).attach()

        # Then
        first.error = first.error

        # Expect
        assert first.covariance is not None

    def test_a_scaled_record_has_no_draws(self, pair):
        # When
        samples = np.random.default_rng(0).multivariate_normal([1, 2], COVARIANCE, size=100)
        record = ParameterCovariance.from_samples(pair, samples)

        # Then
        scaled = record.scaled(2.0)

        # Expect
        assert record.samples is not None
        assert scaled.samples is None
        np.testing.assert_allclose(scaled.covariance, 2 * record.covariance)


class TestLifetime:
    def test_the_record_does_not_keep_parameters_alive(self):
        # When
        first = Parameter(1.0, display_name='first')
        second = Parameter(2.0, display_name='second')
        record = ParameterCovariance([first, second], COVARIANCE).attach()

        # Then
        del first
        gc.collect()

        # Expect
        assert record.parameters[0] is None
        assert record.parameters[1] is second
        assert not record.is_current
        assert second.covariance is None

    def test_copies_pickles_and_serialization_do_not_carry_it(self, pair):
        # When
        first, _ = pair
        record = ParameterCovariance(pair, COVARIANCE).attach()

        # Then Expect
        assert copy.copy(first).covariance is None
        assert copy.deepcopy(first).covariance is None
        assert 'covariance' not in first.to_dict()
        assert Parameter.from_dict(first.to_dict()).covariance is None
        assert copy.copy(record) is record
        assert copy.deepcopy(record) is record
        assert pickle.loads(pickle.dumps(record)) is None


class TestCorrelationWith:
    def test_itself_is_one(self, pair):
        # When Then Expect
        assert pair[0].correlation_with(pair[0]) == 1.0

    def test_unrelated_is_zero(self, pair):
        # When Then Expect
        assert pair[0].correlation_with(pair[1]) == 0.0

    def test_shared_record_gives_the_coefficient(self, pair):
        # When
        ParameterCovariance(pair, COVARIANCE).attach()

        # Then Expect
        assert pair[0].correlation_with(pair[1]) == pytest.approx(0.5)
        assert pair[1].correlation_with(pair[0]) == pytest.approx(0.5)

    def test_parameters_of_different_records_are_unrelated(self, pair):
        # When
        other = Parameter(3.0, display_name='other')
        ParameterCovariance(pair, COVARIANCE).attach()
        ParameterCovariance([other], np.array([[1.0]])).attach()

        # Then Expect
        assert pair[0].correlation_with(other) == 0.0

    def test_stale_record_gives_zero(self, pair):
        # When
        ParameterCovariance(pair, COVARIANCE).attach()
        pair[0].value = 5.0

        # Then Expect
        assert pair[0].correlation_with(pair[1]) == 0.0

    def test_rejects_a_non_parameter(self, pair):
        # When Then Expect
        with pytest.raises(TypeError):
            pair[0].correlation_with(1.0)
