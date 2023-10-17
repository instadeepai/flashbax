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

from flashbax.buffers import trajectory_queue
from flashbax.buffers.conftest import get_fake_batch
from flashbax.conftest import _DEVICE_COUNT_MOCK


@pytest.fixture()
def sample_sequence_length() -> int:
    return 4


@pytest.fixture()
def add_sequence_length() -> int:
    return 8


@pytest.fixture()
def state(
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
) -> trajectory_queue.TrajectoryQueueState:
    """Initialise the trajectory queue state."""
    return trajectory_queue.init(fake_transition, add_batch_size, max_length)


def get_fake_batch_sequence(
    fake_transition: chex.ArrayTree, batch_size: int, sequence_length: int
) -> chex.ArrayTree:
    return get_fake_batch(get_fake_batch(fake_transition, sequence_length), batch_size)


def test_add(
    state: trajectory_queue.TrajectoryQueueState,
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    add_sequence_length: int,
) -> None:
    """Check the `add` function by filling the queue all
    the way to the max_length and checking that it produces the expected behaviour .
    """
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )
    init_state = deepcopy(state)  # Save for later checks.

    n_batches_to_fill = int(np.ceil(max_length / add_sequence_length))

    assert (
        max_length % add_sequence_length == 0
    )  # For testing ensure adding perfectly fills the buffer

    for i in range(n_batches_to_fill):
        assert not state.is_full

        state = trajectory_queue.add(
            state,
            fake_batch_sequence,
        )

        num_added_timesteps = (i + 1) * add_sequence_length
        assert state.write_index == (num_added_timesteps % max_length)

    assert state.is_full
    assert state.write_index == state.read_index
    assert not trajectory_queue.can_add(state, add_sequence_length, max_length)

    # Check that the trajectorys have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)


def test_can_sample(
    state: trajectory_queue.TrajectoryQueueState,
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    add_sequence_length: int,
    sample_sequence_length: int,
) -> None:
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )

    assert (
        max_length % add_sequence_length == 0
    )  # For testing ensure adding perfectly fills the buffer
    assert (
        max_length % sample_sequence_length == 0
    )  # For testing ensure sampling perfectly deletes the buffer

    # CASE 1: READ INDEX == WRITE INDEX with empty buffer
    assert state.read_index == state.write_index
    assert not state.is_full
    assert not trajectory_queue.can_sample(state, sample_sequence_length, max_length)

    # Fill the buffer
    n_batches_to_fill = int(np.ceil(max_length / add_sequence_length))
    for _ in range(n_batches_to_fill):
        assert not state.is_full
        assert trajectory_queue.can_add(state, add_sequence_length, max_length)
        state = trajectory_queue.add(
            state,
            fake_batch_sequence,
        )
    # Check that the buffer is full
    assert state.is_full
    assert state.write_index == state.read_index
    assert not trajectory_queue.can_add(state, add_sequence_length, max_length)

    # Check that we can now sample
    assert trajectory_queue.can_sample(state, sample_sequence_length, max_length)

    # CASE 2: READ INDEX > WRITE INDEX
    n_batches_to_sample = int(np.ceil(max_length / sample_sequence_length))
    # sample one to get the read index above the write index
    state, _ = trajectory_queue.sample(
        state,
        sample_sequence_length,
    )

    assert state.read_index > state.write_index
    # check that we can sample the rest of the buffer
    for _ in range(n_batches_to_sample - 1):
        assert trajectory_queue.can_sample(state, sample_sequence_length, max_length)
        state, sample = trajectory_queue.sample(
            state,
            sample_sequence_length,
        )
    # check that we cannot sample anymore
    assert not trajectory_queue.can_sample(state, sample_sequence_length, max_length)
    assert state.read_index == state.write_index
    # check that we can add again
    assert trajectory_queue.can_add(state, add_sequence_length, max_length)

    # CASE 3: READ INDEX < WRITE INDEX
    # add one to get the read index below the write index
    state = trajectory_queue.add(
        state,
        fake_batch_sequence,
    )

    assert state.read_index < state.write_index
    # check that we can sample up to the write index
    num_samples = add_sequence_length // sample_sequence_length
    for _ in range(num_samples):
        assert trajectory_queue.can_sample(state, sample_sequence_length, max_length)
        state, sample = trajectory_queue.sample(
            state,
            sample_sequence_length,
        )
    # check that we cannot sample anymore
    assert not trajectory_queue.can_sample(state, sample_sequence_length, max_length)

    assert state.read_index <= state.write_index


def test_can_add(
    state: trajectory_queue.TrajectoryQueueState,
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    add_sequence_length: int,
    sample_sequence_length: int,
) -> None:
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )

    assert (
        max_length % add_sequence_length == 0
    )  # For testing ensure adding perfectly fills the buffer
    assert (
        max_length % sample_sequence_length == 0
    )  # For testing ensure sampling perfectly deletes the buffer
    assert state.read_index == state.write_index
    assert add_sequence_length % sample_sequence_length == 0  # For testing

    # CASE 1: Test that we can write until max length is reached.
    n_batches_to_write = int(np.ceil(max_length / add_sequence_length))
    # We add sequences to fill the buffer and check that we can write the whole time
    for _ in range(n_batches_to_write):
        assert not state.is_full
        assert trajectory_queue.can_add(state, add_sequence_length, max_length)
        state = trajectory_queue.add(
            state,
            fake_batch_sequence,
        )

    # Test that we cannot write now.
    assert not trajectory_queue.can_add(state, add_sequence_length, max_length)
    # And check that the queue is full
    assert state.is_full

    # CASE 2: Test that we can write up to read index when WRITE INDEX < READ INDEX.
    # We know by construction that write_index is zero here.
    # We calculate the number of samples we need to perform to allow us
    # the space to write.
    num_samples = add_sequence_length // sample_sequence_length
    for _ in range(num_samples):
        assert trajectory_queue.can_sample(state, sample_sequence_length, max_length)
        state, sample = trajectory_queue.sample(
            state,
            sample_sequence_length,
        )

    # We check that the read index has been updated and is above the write index.
    assert state.read_index > state.write_index
    assert trajectory_queue.can_add(state, add_sequence_length, max_length)
    # we add sequences until we cannot add anymore
    while trajectory_queue.can_add(state, add_sequence_length, max_length):
        state = trajectory_queue.add(
            state,
            fake_batch_sequence,
        )

        # Check that the write index never exceeds the read index.
        assert state.write_index <= state.read_index

    # CASE 3: Test that we can write up to read index when WRITE INDEX > READ INDEX.
    # First, we sample as much as we can to empty the queue and get the read index
    # below the write index.
    while trajectory_queue.can_sample(state, sample_sequence_length, max_length):
        state, sample = trajectory_queue.sample(
            state,
            sample_sequence_length,
        )
    # We check that the read index has been updated and is below the write index.
    assert state.read_index <= state.write_index
    # We add sequences until we cannot add anymore
    assert trajectory_queue.can_add(state, add_sequence_length, max_length)
    while trajectory_queue.can_add(state, add_sequence_length, max_length):
        state = trajectory_queue.add(
            state,
            fake_batch_sequence,
        )

    # Check that the write index never exceeds the read index after it has looped around.
    assert state.write_index <= state.read_index


def test_can_add_and_sample_non_divisible(
    fake_transition: chex.ArrayTree, max_length: int, add_batch_size: int
) -> None:
    max_length = 13
    add_sequence_length = 3
    sample_sequence_length = 5

    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, add_sequence_length
    )

    queue = trajectory_queue.make_trajectory_queue(
        max_length_time_axis=max_length,
        add_batch_size=add_batch_size,
        add_sequence_length=add_sequence_length,
        sample_sequence_length=sample_sequence_length,
    )

    state = queue.init(fake_transition)  # read_index = 0, write_index = 0

    # when empty we can add
    assert queue.can_add(state)
    # when empty we cannot sample
    assert not queue.can_sample(state)

    state = queue.add(state, fake_batch_sequence)  # write_index = 3

    assert queue.can_add(state)
    # since add is less than sample, we cannot sample yet
    assert not queue.can_sample(state)
    state = queue.add(state, fake_batch_sequence)  # write_index = 6

    # now we can add and sample
    assert queue.can_add(state)
    assert queue.can_sample(state)

    # fill the buffer to a point where adding one more would exceed the read index
    state = queue.add(state, fake_batch_sequence)  # write_index = 9
    state = queue.add(state, fake_batch_sequence)  # write_index = 12
    # check that we can not add
    assert not queue.can_add(state)
    # Also check that although we cannot add, our is full flag should still be false
    assert not state.is_full

    # check that we can sample
    assert queue.can_sample(state)
    state, sample = queue.sample(state)  # read_index = 5

    assert queue.can_sample(state)
    state, sample = queue.sample(state)  # read_index = 10

    assert not queue.can_sample(state)
    # check that we can add again
    assert queue.can_add(state)
    state = queue.add(state, fake_batch_sequence)  # write_index = 2 (Looped around)
    # check that we can sample again
    assert queue.can_sample(state)
    state, sample = queue.sample(state)  # read_index = 2
    assert state.read_index == state.write_index

    # Add as much as we can
    while queue.can_add(state):
        state = queue.add(state, fake_batch_sequence)

    # write index should be at 1 which is less than read index at 2
    assert state.write_index < state.read_index


def test_sample(
    state: trajectory_queue.TrajectoryQueueState,
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
) -> None:
    """Test the random sampling from the queue."""

    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, 2 * sample_sequence_length
    )

    state = trajectory_queue.add(
        state,
        fake_batch_sequence,
    )
    assert trajectory_queue.can_sample(state, sample_sequence_length, max_length)
    state, sample1 = trajectory_queue.sample(
        state,
        sample_sequence_length,
    )

    state, sample2 = trajectory_queue.sample(
        state,
        sample_sequence_length,
    )

    # Check that the trajectories have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(sample1, sample2)

    # Check correct the dtypes are correct.
    chex.assert_trees_all_equal_dtypes(
        fake_transition, sample1.experience, sample2.experience
    )


def test_max_sample(
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
) -> None:
    sample_sequence_length = max_length
    max_length_time_axis = max_length
    queue = trajectory_queue.make_trajectory_queue(
        max_length_time_axis=max_length_time_axis,
        add_batch_size=add_batch_size,
        add_sequence_length=sample_sequence_length,
        sample_sequence_length=sample_sequence_length,
    )

    state = queue.init(fake_transition)

    # Initialise the queue's state.
    fake_batch_sequence = get_fake_batch_sequence(
        fake_transition, add_batch_size, sample_sequence_length
    )
    # perform add and sample 10 times
    for _ in range(10):
        assert state.read_index == 0
        assert state.write_index == 0

        assert queue.can_add(state)  # We should be able to write
        state = queue.add(state, fake_batch_sequence)
        assert not queue.can_add(state)  # We should be able to write

        assert queue.can_sample(state)  # We should be able to sample
        state, sample = queue.sample(state)
        assert not queue.can_sample(state)  # We should not be able to sample
        chex.assert_trees_all_close(
            sample.experience, fake_batch_sequence
        )  # We should have sampled the whole queue


def test_trajectory_queue_does_not_smoke(
    fake_transition: chex.ArrayTree,
    max_length: int,
    add_batch_size: int,
    sample_sequence_length: int,
    add_sequence_length: int,
):
    """Create the trajectoryqueue NamedTuple, and check that it is pmap-able and does not smoke."""
    queue = trajectory_queue.make_trajectory_queue(
        max_length_time_axis=max_length,
        add_batch_size=add_batch_size,
        add_sequence_length=add_sequence_length,
        sample_sequence_length=sample_sequence_length,
    )

    # Initialise the queue's state.
    fake_trajectory_per_device = jax.tree_map(
        lambda x: jnp.stack([x + i for i in range(_DEVICE_COUNT_MOCK)]), fake_transition
    )
    state = jax.pmap(queue.init)(fake_trajectory_per_device)
    assert not queue.can_sample(state).all()

    fake_batch = jax.pmap(get_fake_batch_sequence, static_broadcasted_argnums=(1, 2))(
        fake_trajectory_per_device, add_batch_size, add_sequence_length
    )
    assert jax.pmap(queue.can_add)(state).all()
    state = jax.pmap(queue.add)(state, fake_batch)
    assert queue.can_sample(state).all()

    # Sample from the queue.
    state, batch = jax.pmap(queue.sample)(state)
    chex.assert_tree_shape_prefix(
        batch.experience,
        (_DEVICE_COUNT_MOCK, add_batch_size, sample_sequence_length),
    )
