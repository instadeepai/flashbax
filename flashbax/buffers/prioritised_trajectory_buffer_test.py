# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import List

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flashbax.buffers import prioritised_trajectory_buffer, sum_tree, trajectory_buffer
from flashbax.buffers.conftest import get_fake_batch
from flashbax.conftest import _DEVICE_COUNT_MOCK


@pytest.fixture()
def sample_sequence_length() -> int:
    return 5


@pytest.fixture()
def add_sequence_length() -> int:
    return 7


@pytest.fixture()
def sample_period() -> int:
    return 3


@pytest.fixture()
def priority_exponent() -> float:
    return 0.6


@pytest.fixture()
def device() -> str:
    return "tpu"


@pytest.fixture()
def prioritised_state(
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    sample_period: int,
) -> prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState:
    """Initialise the trajectory buffer state."""
    return prioritised_trajectory_buffer.prioritised_init(
        fake_transition,
        add_batch_size,
        max_length,
        sample_period,
    )


def get_fake_batch_sequence(
    fake_transition: chex.ArrayTree, batch_size: int, sequence_length: int
) -> chex.ArrayTree:
    return get_fake_batch(get_fake_batch(fake_transition, sequence_length), batch_size)


def test_add_and_can_sample_prioritised(
    prioritised_state: prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState,
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    device: str,
) -> None:
    """Check the `add` function by filling the buffer all
    the way to the max_length and checking that it produces the expected behaviour .
    """
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )
    init_state = deepcopy(prioritised_state)  # Save for later checks.

    n_batches_to_fill = int(np.ceil(max_length / add_sequence_length))
    n_batches_to_sample = int(np.ceil(min_length / add_sequence_length))

    for i in range(n_batches_to_fill):
        assert not prioritised_state.is_full
        prioritised_state = prioritised_trajectory_buffer.prioritised_add(
            prioritised_state,
            fake_batch_sequence,
            sample_sequence_length,
            sample_period,
            device,
        )
        num_added_timesteps = (i + 1) * add_sequence_length
        assert prioritised_state.current_index == (num_added_timesteps % max_length)

        # Check that the `can_sample` function behavior is correct.
        is_ready_to_sample = trajectory_buffer.can_sample(prioritised_state, min_length)
        if i < (n_batches_to_sample - 1):
            assert not is_ready_to_sample
        else:
            assert is_ready_to_sample

    assert prioritised_state.is_full

    # Check that the trajectorys have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(prioritised_state.experience, init_state.experience)


def test_prioritised_sample(
    prioritised_state: prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState,
    fake_transition: chex.ArrayTree,
    min_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
    device: str,
) -> None:
    """Test the random sampling from the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Fill buffer to the point that we can sample
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, min_length + 10
    )
    prioritised_state = prioritised_trajectory_buffer.prioritised_add(
        prioritised_state,
        fake_batch_sequence,
        sample_sequence_length,
        sample_period,
        device,
    )

    assert trajectory_buffer.can_sample(prioritised_state, min_length)

    # Sample from the buffer with different keys and check it gives us different batches.
    batch1 = prioritised_trajectory_buffer.prioritised_sample(
        prioritised_state,
        rng_key1,
        sample_batch_size,
        sample_sequence_length,
        sample_period,
    )
    batch2 = prioritised_trajectory_buffer.prioritised_sample(
        prioritised_state,
        rng_key2,
        sample_batch_size,
        sample_sequence_length,
        sample_period,
    )

    # Check that the trajectorys have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(batch1, batch2)

    assert (batch1.priorities > 0).all()
    assert (batch2.priorities > 0).all()

    # Check correct the shape prefix is correct.
    chex.assert_trees_all_equal_dtypes(
        fake_transition, batch1.experience, batch2.experience
    )


@pytest.mark.parametrize("sample_period", [1, 2, 3, 4, 5])
def test_prioritised_sample_with_period(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
    device: str,
) -> None:
    """Test the random sampling with different periods."""

    # Choose period based on the degree of overlap tested
    assert sample_sequence_length >= sample_period

    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Initialise the buffer
    state = prioritised_trajectory_buffer.prioritised_init(
        fake_transition, add_batch_size, max_length, sample_period
    )

    # Create a batch but specifically ensure that sequences in different add_batch rows
    # are distinct - this is simply for testing purposes in order to verify periodicity
    fake_batch_sequence = jax.tree_map(
        lambda x: jnp.stack([x + i * (max_length - 1) for i in range(add_batch_size)]),
        get_fake_batch(fake_transition, max_length - 1),
    )

    assert np.prod(fake_batch_sequence["reward"].shape) == np.prod(
        jnp.unique(fake_batch_sequence["reward"]).shape
    )

    # Add the fake sequence to the buffer
    state = prioritised_trajectory_buffer.prioritised_add(
        state, fake_batch_sequence, sample_sequence_length, sample_period, device
    )

    assert trajectory_buffer.can_sample(state, min_length)

    # Sample from the buffer
    batch1 = prioritised_trajectory_buffer.prioritised_sample(
        state, rng_key1, sample_batch_size, sample_sequence_length, sample_period
    )

    # Check correct the shape prefix is correct.
    chex.assert_tree_shape_prefix(
        batch1.experience, (sample_batch_size, sample_sequence_length)
    )

    # Check that the initial value in each sequence is always in a position that is a
    # multiple of the sample period or zero.
    # We check each sequence compared to every other sequence.
    for i in range(sample_batch_size):
        equal = batch1.experience["reward"][i][0] == batch1.experience["reward"]  # type: ignore
        pos = jnp.argmax(equal, axis=1)
        test = (pos % sample_period == 0).astype(jnp.int32) + (pos == 0).astype(
            jnp.int32
        )
        assert jnp.all(test.astype(jnp.bool_))


def test_adjust_priorities(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_sequence_length: int,
    sample_batch_size: int,
    sample_period: int,
    priority_exponent: float,
    device: str,
) -> None:
    """Test the adjustment of priorities in the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    state = prioritised_trajectory_buffer.prioritised_init(
        fake_transition,
        add_batch_size,
        max_length,
        sample_period,
    )

    # Fill buffer to the point that we can sample.

    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, min_length + 10
    )

    state = prioritised_trajectory_buffer.prioritised_add(
        state, fake_batch_sequence, sample_sequence_length, sample_period, device
    )

    # Sample from the buffer.
    batch = prioritised_trajectory_buffer.prioritised_sample(
        state, rng_key1, sample_batch_size, sample_sequence_length, sample_period
    )

    # Create fake new priorities, and apply the adjustment.
    new_priorities = jnp.ones_like(batch.priorities) + 10007
    state = prioritised_trajectory_buffer.set_priorities(
        state, batch.indices, new_priorities, priority_exponent, device
    )

    # Check that this results in the correct changes to the state.
    assert (
        state.priority_state.max_recorded_priority
        == jnp.max(new_priorities) ** priority_exponent
    )
    assert (
        sum_tree.get(state.priority_state, batch.indices)
        == new_priorities**priority_exponent
    ).all()


def test_prioritised_trajectory_buffer_does_not_smoke(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    sample_period: int,
    rng_key: chex.PRNGKey,
):
    """Create the prioritiesed trajectory buffer dataclass, and check that it is
    pmap-able and does not smoke."""
    buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer(
        max_length_time_axis=max_length,
        min_length_time_axis=min_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=sample_period,
    )

    # Initialise the buffer's state.
    fake_trajectory_per_device = jax.tree_map(
        lambda x: jnp.stack([x + i for i in range(_DEVICE_COUNT_MOCK)]), fake_transition
    )
    state = jax.pmap(buffer.init)(fake_trajectory_per_device)
    assert not buffer.can_sample(state).all()

    # Now fill the buffer above its minimum length.
    add_n_steps = int(min_length + 5)
    fake_batch = jax.pmap(get_fake_batch_sequence, static_broadcasted_argnums=(1, 2))(
        fake_trajectory_per_device, add_batch_size, add_n_steps
    )
    state = jax.pmap(buffer.add)(state, fake_batch)
    assert buffer.can_sample(state).all()

    # Sample from the buffer.
    rng_key_per_device = jax.random.split(rng_key, _DEVICE_COUNT_MOCK)
    batch = jax.pmap(buffer.sample)(state, rng_key_per_device)
    chex.assert_tree_shape_prefix(
        batch.experience,
        (_DEVICE_COUNT_MOCK, sample_batch_size, sample_sequence_length),
    )
    chex.assert_tree_shape_prefix(
        batch.priorities, (_DEVICE_COUNT_MOCK, sample_batch_size)
    )

    state = jax.pmap(buffer.set_priorities)(
        state, batch.indices, jnp.ones_like(batch.priorities)
    )


def check_index_calc(
    fake_transition: chex.ArrayTree,
    add_batch_size: int,
    sample_batch_size: int,
    max_length: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    add_iter: int,
    expected_priority_indices: List[int],
    expected_priority_values: List[float],
):
    """Helper function to check the indices and values returned by the
    calculate_item_indices_and_values function."""
    buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer(
        max_length_time_axis=max_length,
        min_length_time_axis=sample_sequence_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=sample_period,
    )

    state = buffer.init(fake_transition)

    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )

    # We add to the buffer before calling function for testing purposes because
    # the calculate item indices and values function would be called before the
    # data is officially inserted into the buffer. Thats why on add_iter = 0,
    # we are technically referring to the items created after adding 1 sequence
    # to the buffer but for testing the indices we must not actually add the data.
    for _ in range(add_iter):
        state = buffer.add(state, fake_batch_sequence)

    (
        priority_indices,
        priority_values,
    ) = prioritised_trajectory_buffer.calculate_item_indices_and_priorities(
        state,
        sample_sequence_length,
        sample_period,
        add_sequence_length,
        add_batch_size,
        max_length,
    )
    assert jnp.all(priority_indices == jnp.array(expected_priority_indices))
    assert jnp.all(priority_values == jnp.array(expected_priority_values))


@pytest.mark.parametrize(
    "add_batch_size, max_length, add_sequence_length, sample_sequence_length, sample_period",
    [(1, 13, 7, 5, 3)],
)
@pytest.mark.parametrize(
    "add_iter, expected_priority_indices, expected_priority_values",
    [
        (0, [0, 1, 2], [1.0, 0.0, 0.0]),
        (1, [1, 2, 3], [1.0, 1.0, 1.0]),
        (2, [0, 1, 2], [1.0, 1.0, 0.0]),
        (3, [2, 3, 0], [1.0, 1.0, 0.0]),
        (4, [0, 1, 2], [1.0, 1.0, 0.0]),
        (5, [2, 3, 0], [1.0, 1.0, 0.0]),
        (6, [0, 1, 2], [1.0, 1.0, 0.0]),
        (7, [2, 3, 0], [1.0, 1.0, 0.0]),
        (8, [0, 1, 2], [1.0, 1.0, 1.0]),
        (9, [3, 0, 1], [1.0, 1.0, 0.0]),
        (10, [1, 2, 3], [1.0, 1.0, 0.0]),
        (11, [3, 0, 1], [1.0, 1.0, 0.0]),
        (12, [1, 2, 3], [1.0, 1.0, 0.0]),
    ],
)
def test_item_and_priority_calculation_case1(
    fake_transition: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_batch_size: int,
    max_length: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    add_iter: int,
    expected_priority_indices: List[int],
    expected_priority_values: List[float],
):
    """
    Case 1: max_length = 13, add_sequence_length = 7, sample_sequence_length = 5, sample_period = 3

    Check max_length not divisible by sample_period & add_sequence_length not divisible by
    sample_period & max_length not divisible by add_sequence_length.
    """

    check_index_calc(
        fake_transition,
        add_batch_size,
        sample_batch_size,
        max_length,
        add_sequence_length,
        sample_sequence_length,
        sample_period,
        add_iter,
        expected_priority_indices,
        expected_priority_values,
    )


@pytest.mark.parametrize(
    "add_batch_size, max_length, add_sequence_length, sample_sequence_length, sample_period",
    [(1, 8, 3, 5, 2)],
)
@pytest.mark.parametrize(
    "add_iter, expected_priority_indices, expected_priority_values",
    [
        (0, [0, 1], [0.0, 0.0]),
        (1, [0, 1], [1.0, 0.0]),
        (
            2,
            [1, 2],
            [
                1.0,
                1.0,
            ],
        ),
        (
            3,
            [3, 0],
            [
                1.0,
                0.0,
            ],
        ),
        (
            4,
            [0, 1],
            [
                1.0,
                1.0,
            ],
        ),
        (
            5,
            [2, 3],
            [
                1.0,
                0.0,
            ],
        ),
        (
            6,
            [3, 0],
            [
                1.0,
                1.0,
            ],
        ),
        (
            7,
            [1, 2],
            [
                1.0,
                0.0,
            ],
        ),
    ],
)
def test_item_and_priority_calculation_case2(
    fake_transition: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_batch_size: int,
    max_length: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    add_iter: int,
    expected_priority_indices: List[int],
    expected_priority_values: List[float],
):
    """
    Case 2: max_length = 8, add_sequence_length = 3, sample_sequence_length = 5, sample_period = 2

    Check max length divisible by sample period & add sequence length not divisible by sample
    period & sample sequence length greater than add sequence length
    """
    check_index_calc(
        fake_transition,
        add_batch_size,
        sample_batch_size,
        max_length,
        add_sequence_length,
        sample_sequence_length,
        sample_period,
        add_iter,
        expected_priority_indices,
        expected_priority_values,
    )


@pytest.mark.parametrize(
    "add_batch_size, max_length, add_sequence_length, sample_sequence_length, sample_period",
    [(1, 8, 3, 3, 4)],
)
@pytest.mark.parametrize(
    "add_iter, expected_priority_indices, expected_priority_values",
    [
        (0, [0], [1.0]),
        (1, [1], [0.0]),
        (2, [1], [1.0]),
        (3, [0], [1.0]),
        (4, [1], [1.0]),
        (5, [0], [0.0]),
        (6, [0], [1.0]),
        (7, [1], [1.0]),
    ],
)
def test_item_and_priority_calculation_case3(
    fake_transition: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_batch_size: int,
    max_length: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    add_iter: int,
    expected_priority_indices: List[int],
    expected_priority_values: List[float],
):
    """Case 3: max_length = 8, add_sequence_length = 3, sample_sequence_length = 3,
    sample_period = 4

    Check period greater than add sequence length and sample sequence length but Max
    length divisible by sample period.
    """
    check_index_calc(
        fake_transition,
        add_batch_size,
        sample_batch_size,
        max_length,
        add_sequence_length,
        sample_sequence_length,
        sample_period,
        add_iter,
        expected_priority_indices,
        expected_priority_values,
    )


@pytest.mark.parametrize(
    "add_batch_size, max_length, add_sequence_length, sample_sequence_length, sample_period",
    [(2, 8, 3, 3, 4)],
)
@pytest.mark.parametrize(
    "add_iter, expected_priority_indices, expected_priority_values",
    [
        (0, [0, 2], [1.0, 1.0]),
        (1, [1, 3], [0.0, 0.0]),
        (2, [1, 3], [1.0, 1.0]),
        (3, [0, 2], [1.0, 1.0]),
        (4, [1, 3], [1.0, 1.0]),
        (5, [0, 2], [0.0, 0.0]),
        (6, [0, 2], [1.0, 1.0]),
        (7, [1, 3], [1.0, 1.0]),
    ],
)
def test_item_and_priority_calculation_case4(
    fake_transition: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_batch_size: int,
    max_length: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    add_iter: int,
    expected_priority_indices: List[int],
    expected_priority_values: List[float],
):
    """Case 4: add_batch_size = 2 max_length = 8, add_sequence_length = 3,
    sample_sequence_length = 3, sample_period = 4

    Check simple case when using multiple rows in the buffer.
    """
    check_index_calc(
        fake_transition,
        add_batch_size,
        sample_batch_size,
        max_length,
        add_sequence_length,
        sample_sequence_length,
        sample_period,
        add_iter,
        expected_priority_indices,
        expected_priority_values,
    )


@pytest.mark.parametrize(
    "add_batch_size, max_length, add_sequence_length, sample_sequence_length, sample_period",
    [(1, 15, 5, 5, 1)],
)
@pytest.mark.parametrize(
    "add_iter, expected_priority_indices, expected_priority_values",
    [
        (0, [0, 1, 2, 3, 4, 5], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (1, [1, 2, 3, 4, 5, 6], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        (2, [6, 7, 8, 9, 10, 11], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        (3, [11, 12, 13, 14, 0, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    ],
)
def test_item_and_priority_calculation_case5(
    fake_transition: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_batch_size: int,
    max_length: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    sample_period: int,
    add_iter: int,
    expected_priority_indices: List[int],
    expected_priority_values: List[float],
):
    """Case 5: add_batch_size = 1 max_length = 15, add_sequence_length = 5,
    sample_sequence_length = 5, sample_period = 1
    """
    check_index_calc(
        fake_transition,
        add_batch_size,
        sample_batch_size,
        max_length,
        add_sequence_length,
        sample_sequence_length,
        sample_period,
        add_iter,
        expected_priority_indices,
        expected_priority_values,
    )
