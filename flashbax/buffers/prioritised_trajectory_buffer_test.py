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


def test_add_max_length(
    fake_transition: chex.ArrayTree,
    device: str,
    sample_period: int,
    max_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
) -> None:
    """Check the `add` function works when adding the max length."""
    prioritised_state = prioritised_trajectory_buffer.prioritised_init(
        fake_transition,
        add_batch_size,
        max_length,
        sample_period,
    )
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, max_length
    )
    assert not prioritised_state.is_full
    prioritised_state = prioritised_trajectory_buffer.prioritised_add(
        prioritised_state,
        fake_batch_sequence,
        sample_sequence_length,
        sample_period,
        device,
    )
    assert prioritised_state.is_full
    sampled = prioritised_trajectory_buffer.prioritised_sample(
        prioritised_state,
        jax.random.PRNGKey(0),
        1,
        sample_sequence_length,
        sample_period,
    )
    assert sampled.experience["reward"].shape == (1, sample_sequence_length)


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

    assert (batch1.probabilities > 0).all()
    assert (batch2.probabilities > 0).all()

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
    fake_batch_sequence = jax.tree.map(
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
    new_priorities = jnp.ones_like(batch.probabilities) + 10007
    state = prioritised_trajectory_buffer.set_priorities(
        state, batch.indices, new_priorities, priority_exponent, device
    )

    # Check that this results in the correct changes to the state.
    assert (
        state.sum_tree_state.max_recorded_priority
        == jnp.max(new_priorities) ** priority_exponent
    )
    assert (
        sum_tree.get(state.sum_tree_state, batch.indices)
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
    fake_trajectory_per_device = jax.tree.map(
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
        batch.probabilities, (_DEVICE_COUNT_MOCK, sample_batch_size)
    )

    state = jax.pmap(buffer.set_priorities)(
        state, batch.indices, jnp.ones_like(batch.probabilities)
    )


def is_strictly_increasing(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Returns a (True/False) indicating whether `arr`
    is strictly increasing (arr[i] > arr[i-1] for all i).
    """
    # If arr has 0 or 1 elements, it's trivially increasing.
    if arr.shape[0] <= 1:
        return jnp.bool_(True)
    # Otherwise, check that every adjacent difference is positive.
    return jnp.all(arr[1:] > arr[:-1])


@pytest.mark.parametrize("add_length", [4, 9, 13])
@pytest.mark.parametrize("add_batch_size", [2, 3])
@pytest.mark.parametrize("sample_sequence_length", [3, 4, 9, 13])
@pytest.mark.parametrize("period", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("max_length_time_axis", [13, 16, 26, 39])
def test_prioritised_sample_doesnt_sample_prev_broken_trajectories(
    add_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    period: int,
    max_length_time_axis: int,
) -> None:
    """Test to ensure that `sample` avoids including rewards from broken
    trajectories.
    """
    # Because we are sweeping over a range of values
    # it is easier to check here if we shouldnt test
    remainder = max_length_time_axis % period
    real_length = max_length_time_axis - remainder
    if real_length <= add_length or real_length <= sample_sequence_length:
        # Due to the new constraints placed on the PER buffer length
        # if this situation arises we simply skip
        return

    fake_transition = {"reward": jnp.array([1])}

    offset = jnp.arange(add_batch_size).reshape(add_batch_size, 1, 1) * 1000

    buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer(
        add_batch_size=add_batch_size,
        sample_batch_size=2048,
        sample_sequence_length=sample_sequence_length,
        period=period,
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=sample_sequence_length,
    )
    buffer_init = jax.jit(buffer.init)
    buffer_add = jax.jit(buffer.add)
    buffer_sample = jax.jit(buffer.sample)

    rng_key = jax.random.PRNGKey(0)
    state = buffer_init(fake_transition)

    for i in range(12):
        fake_batch_sequence = {
            "reward": jnp.arange(add_length)
            .reshape(1, add_length, 1)
            .repeat(add_batch_size, axis=0)
            + offset
            + add_length * i
        }
        state = buffer_add(state, fake_batch_sequence)

        # If the root node of the sum tree is zero
        # this means there are no valid items available
        # then test will definitely fail so we skip this
        # This is under the assumption that the sum tree
        # is correctly implemented which we have tests for.
        if state.sum_tree_state.nodes[0] == 0:
            continue

        rng_key, rng_key1 = jax.random.split(rng_key)

        if buffer.can_sample(state):
            sample = buffer_sample(state, rng_key1)

            sampled_r = sample.experience["reward"]

            for b in range(sampled_r.shape[0]):
                assert is_strictly_increasing(sampled_r[b])


# HUMAN TEST CASES
#####################
def _human_test_cases_for_valid_and_invalid_indices(
    add_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    period: int,
    max_length_time_axis: int,
    correct_valid_item_indices: dict,
    correct_invalid_item_indices: dict,
) -> None:
    """Test to ensure that `sample` avoids including rewards from broken
    trajectories.
    """

    fake_transition = {"reward": jnp.array([1])}

    offset = jnp.arange(add_batch_size).reshape(add_batch_size, 1, 1) * 1000

    buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer(
        add_batch_size=add_batch_size,
        sample_batch_size=2048,
        sample_sequence_length=sample_sequence_length,
        period=period,
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=sample_sequence_length,
    )
    buffer_init = jax.jit(buffer.init)
    buffer_add = jax.jit(buffer.add)

    state = buffer_init(fake_transition)

    for i in range(5):
        fake_batch_sequence = {
            "reward": jnp.arange(add_length)
            .reshape(1, add_length, 1)
            .repeat(add_batch_size, axis=0)
            + offset
            + add_length * i
        }

        (
            valid_items,
            invalid_items,
        ) = prioritised_trajectory_buffer._calculate_new_item_indices(
            state.running_index,
            add_length,
            period,
            max_length_time_axis,
            sample_sequence_length,
            add_batch_size,
        )
        for j, idx in enumerate(correct_valid_item_indices[i]):
            assert idx == valid_items[j], "Incorrectly calculated valid item"

        for j, idx in enumerate(correct_invalid_item_indices[i]):
            assert (
                idx == invalid_items[j]
            ), f"Incorrectly calculated invalid item, {idx} vs {invalid_items[j]}"

        assert jnp.all(state.sum_tree_state.nodes >= 0)

        state = buffer_add(state, fake_batch_sequence)


def test_human_case_1():
    # TEST CASE 1
    test_case_1_valid = {
        0: [0],
        1: [5, 6],
        2: [4, 5],
        3: [2, 3],
        4: [1, 2],
    }
    test_case_1_invalid = {
        0: [],
        1: [0],
        2: [5, 6],
        3: [4, 5],
        4: [2, 3],
    }

    _human_test_cases_for_valid_and_invalid_indices(
        13, 1, 13, 2, 16, test_case_1_valid, test_case_1_invalid
    )


def test_human_case_2():
    # TEST CASE 2
    test_case_2_valid = {
        0: [],
        1: [0, 1],
        2: [2],
        3: [3, 4],
        4: [5],
    }
    test_case_2_invalid = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [0, 1],
    }

    _human_test_cases_for_valid_and_invalid_indices(
        3, 1, 4, 2, 12, test_case_2_valid, test_case_2_invalid
    )


def test_human_case_3():
    # TEST CASE 3
    test_case_3_valid = {0: [], 1: [0], 2: [1, 2], 3: [3], 4: [0, 1]}
    test_case_3_invalid = {0: [], 1: [], 2: [0], 3: [1], 4: [2, 3]}

    _human_test_cases_for_valid_and_invalid_indices(
        3, 1, 5, 2, 8, test_case_3_valid, test_case_3_invalid
    )


#####################
