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

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flashbax.buffers import trajectory_buffer
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
def state(
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    sample_period: int,
) -> trajectory_buffer.TrajectoryBufferState:
    """Initialise the trajectory buffer state."""
    return trajectory_buffer.init(
        fake_transition,
        add_batch_size,
        max_length,
    )


def get_fake_batch_sequence(
    fake_transition: chex.ArrayTree, batch_size: int, sequence_length: int
) -> chex.ArrayTree:
    return get_fake_batch(get_fake_batch(fake_transition, sequence_length), batch_size)


def test_add_and_can_sample(
    state: trajectory_buffer.TrajectoryBufferState,
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    add_sequence_length: int,
) -> None:
    """Check the `add` function by filling the buffer all
    the way to the max_length and checking that it produces the expected behaviour .
    """
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )
    init_state = deepcopy(state)  # Save for later checks.

    n_batches_to_fill = int(np.ceil(max_length / add_sequence_length))
    n_batches_to_sample = int(np.ceil(min_length / add_sequence_length))

    for i in range(n_batches_to_fill):
        assert not state.is_full
        state = trajectory_buffer.add(
            state,
            fake_batch_sequence,
        )
        num_added_timesteps = (i + 1) * add_sequence_length
        assert state.current_index == (num_added_timesteps % max_length)

        # Check that the `can_sample` function behavior is correct.
        is_ready_to_sample = trajectory_buffer.can_sample(state, min_length)
        if i < (n_batches_to_sample - 1):
            assert not is_ready_to_sample
        else:
            assert is_ready_to_sample

    assert state.is_full

    # Check that the trajectorys have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)


def test_uniform_sample(
    state: trajectory_buffer.TrajectoryBufferState,
    fake_transition: chex.ArrayTree,
    min_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
) -> None:
    """Test the random sampling from the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Fill buffer to the point that we can sample
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, min_length + 10
    )
    state = trajectory_buffer.add(
        state,
        fake_batch_sequence,
    )

    assert trajectory_buffer.can_sample(state, min_length)

    # Sample from the buffer with different keys and check it gives us different batches.
    batch1 = trajectory_buffer.sample(
        state, rng_key1, sample_batch_size, sample_sequence_length, sample_period
    )
    batch2 = trajectory_buffer.sample(
        state, rng_key2, sample_batch_size, sample_sequence_length, sample_period
    )

    # Check that the trajectorys have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(batch1, batch2)

    # Check correct the shape prefix is correct.
    chex.assert_trees_all_equal_dtypes(
        fake_transition, batch1.experience, batch2.experience
    )


@pytest.mark.parametrize("add_sequence_length", [1, 2, 3, 4])
def test_add_sample_max_capacity(
    fake_transition: chex.ArrayTree,
    add_batch_size: int,
    rng_key: chex.PRNGKey,
    add_sequence_length: int,
) -> None:
    """Test that we can add an entire batch of sequences to the buffer and sample from it."""
    rng_key1, rng_key2 = jax.random.split(rng_key)
    sample_batch_size = 128
    sample_period = 1

    # Fill buffer to the point that we can sample
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )
    sample_sequence_length = add_sequence_length
    buffer = trajectory_buffer.make_trajectory_buffer(
        max_length_time_axis=add_sequence_length,
        min_length_time_axis=0,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=sample_period,
    )

    state = buffer.init(fake_transition)

    state = buffer.add(state, fake_batch_sequence)

    assert state.current_index == 0
    assert state.is_full

    # Sample from the buffer with different keys and check it gives us different batches.
    batch1 = trajectory_buffer.sample(
        state, rng_key1, sample_batch_size, sample_sequence_length, sample_period
    )
    batch2 = trajectory_buffer.sample(
        state, rng_key2, sample_batch_size, sample_sequence_length, sample_period
    )

    # Check that the trajectorys have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(batch1, batch2)

    # Check correct the shape prefix is correct.
    chex.assert_trees_all_equal_dtypes(
        fake_transition, batch1.experience, batch2.experience
    )

    chex.assert_trees_all_close(state.experience, fake_batch_sequence)

    new_fake_batch_sequence = jax.tree_util.tree_map(
        lambda x: x + add_sequence_length + 1, fake_batch_sequence
    )
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_fake_batch_sequence, fake_batch_sequence)

    new_state = buffer.add(state, new_fake_batch_sequence)

    chex.assert_trees_all_close(new_state.experience, new_fake_batch_sequence)

    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, new_state.experience)

    assert new_state.current_index == state.current_index


@pytest.mark.parametrize("sample_period", [1, 2, 3, 4])
def test_sample_with_period(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
) -> None:
    """Test the random sampling with different periods."""

    # Choose period based on the degree of overlap tested
    assert sample_sequence_length >= sample_period

    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Initialise the buffer
    state = trajectory_buffer.init(
        fake_transition,
        add_batch_size,
        max_length,
    )

    # Create a fake sequence of length min_length + 10
    fake_sequence = get_fake_batch(
        fake_transition,
        min_length + 10,
    )

    # Create a batch but specifically ensure that sequences in different add_batch rows
    # are distinct - this is simply for testing purposes in order to verify periodicity
    fake_batch_sequence = jax.tree_map(
        lambda x: jnp.stack([x + i * (min_length + 10) for i in range(add_batch_size)]),
        fake_sequence,
    )

    # Add the fake sequence to the buffer
    state = trajectory_buffer.add(
        state,
        fake_batch_sequence,
    )

    assert trajectory_buffer.can_sample(state, min_length)

    # Sample from the buffer
    batch1 = trajectory_buffer.sample(
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


def test_trajectory_buffer_does_not_smoke(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    sample_period: int,
    rng_key: chex.PRNGKey,
):
    """Create the trajectoryBuffer NamedTuple, and check that it is pmap-able and does not smoke."""
    buffer = trajectory_buffer.make_trajectory_buffer(
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


@pytest.mark.parametrize("sample_sequence_length", [2, 3, 4])
@pytest.mark.parametrize("add_batch_size", [1, 2, 3])
def test_uniform_index_cal(
    fake_transition: chex.ArrayTree,
    sample_batch_size: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    add_batch_size: int,
    sample_sequence_length: int,
):
    # We enforce a period of 1 so we can deterministically calculate the indices that
    # should be allowed
    sample_period = 1

    buffer = trajectory_buffer.make_trajectory_buffer(
        max_length_time_axis=max_length,
        min_length_time_axis=0,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=sample_period,
    )

    state = buffer.init(fake_transition)

    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, sample_sequence_length
    )

    state = buffer.add(state, fake_batch_sequence)

    assert buffer.can_sample(state)
    rng_key, subkey = jax.random.split(rng_key)
    item_indices = trajectory_buffer.calculate_uniform_item_indices(
        state,
        subkey,
        sample_batch_size,
        sample_sequence_length,
        sample_period,
        add_batch_size,
        max_length,
    )
    # We check the initial add
    check_array = jnp.zeros_like(item_indices)
    for i in range(add_batch_size):
        check_array += item_indices == max_length * i

    assert jnp.all(check_array)

    state = buffer.add(state, fake_batch_sequence)
    rng_key, subkey = jax.random.split(rng_key)
    item_indices = trajectory_buffer.calculate_uniform_item_indices(
        state,
        subkey,
        sample_batch_size,
        sample_sequence_length,
        sample_period,
        add_batch_size,
        max_length,
    )
    # We check the second add
    check_array = jnp.zeros_like(item_indices)
    for i in range(sample_sequence_length + 1):
        for j in range(add_batch_size):
            check_array += item_indices == max_length * j + i

    assert jnp.all(check_array)

    while not state.is_full:
        state = buffer.add(state, fake_batch_sequence)

    invalid_indices = trajectory_buffer.get_invalid_indices(
        state, sample_sequence_length, sample_period, add_batch_size, max_length
    )
    rng_key, subkey = jax.random.split(rng_key)
    item_indices = trajectory_buffer.calculate_uniform_item_indices(
        state,
        subkey,
        sample_batch_size,
        sample_sequence_length,
        sample_period,
        add_batch_size,
        max_length,
    )
    # We check that the invalid indices are not sampled
    assert jnp.all(item_indices != invalid_indices.flatten()[:, jnp.newaxis])
