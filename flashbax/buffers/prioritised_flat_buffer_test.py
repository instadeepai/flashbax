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
import pytest

from flashbax.buffers import prioritised_flat_buffer, sum_tree
from flashbax.buffers.conftest import get_fake_batch
from flashbax.conftest import _DEVICE_COUNT_MOCK


@pytest.fixture()
def priority_exponent() -> float:
    return 0.6


def test_add_and_can_sample(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    sample_batch_size: int,
    priority_exponent: float,
) -> None:
    """Check the `add` function by filling the buffer all
    the way to the max_length and checking that it produces the expected behaviour .
    """
    fake_batch = get_fake_batch(fake_transition, add_batch_size)
    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length,
        min_length,
        sample_batch_size,
        False,
        add_batch_size,
        priority_exponent,
    )
    init_state = buffer.init(fake_transition)
    state = deepcopy(init_state)

    n_batches_to_fill = max_length // add_batch_size
    n_batches_to_sample = min_length // add_batch_size

    for i in range(n_batches_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == ((i + 1) % (max_length // add_batch_size))

        # Check that the `can_sample` function behavior is correct.
        is_ready_to_sample = buffer.can_sample(state)
        if i < n_batches_to_sample:
            assert not is_ready_to_sample
        else:
            assert is_ready_to_sample

    assert state.is_full

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)
        chex.assert_trees_all_close(state.indices, init_state.indices)


def test_sample(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
) -> None:
    """Test the random sampling from the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Fill buffer to the point that we can sample
    fake_batch = get_fake_batch(fake_transition, min_length)

    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length, min_length, sample_batch_size, False, add_batch_size=min_length
    )
    state = buffer.init(fake_transition)

    # Add two items thereby giving a single transition.
    state = buffer.add(state, fake_batch)
    state = buffer.add(state, fake_batch)
    assert buffer.can_sample(state)

    # Sample from the buffer with different keys and check it gives us different batches.
    batch1 = buffer.sample(state, rng_key1)
    batch2 = buffer.sample(state, rng_key2)

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(batch1, batch2)

    # Check that all the corresponding priorities are greater than 0.
    assert (batch1.priorities > 0).all()
    assert (batch2.priorities > 0).all()

    # Check dtypes are correct.
    chex.assert_trees_all_equal_dtypes(
        fake_transition,
        batch1.experience.first,
        batch1.experience.second,
        batch2.experience.first,
        batch2.experience.second,
    )


def test_adjust_priorities(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    priority_exponent: float,
) -> None:
    """Test the adjustment of priorities in the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Fill buffer to the point that we can sample.
    fake_batch = get_fake_batch(fake_transition, min_length)
    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length,
        min_length,
        sample_batch_size,
        False,
        add_batch_size=min_length,
        priority_exponent=priority_exponent,
    )
    state = buffer.init(fake_transition)

    state = buffer.add(state, fake_batch)

    # Sample from the buffer.
    batch = buffer.sample(state, rng_key1)

    # Create fake new priorities, and apply the adjustment.
    new_priorities = jnp.ones_like(batch.priorities) + 10007
    state = buffer.set_priorities(state, batch.indices, new_priorities)

    # Check that this results in the correct changes to the state.
    assert (
        state.priority_state.max_recorded_priority
        == jnp.max(new_priorities) ** priority_exponent
    )
    assert (
        sum_tree.get(state.priority_state, batch.indices)
        == new_priorities**priority_exponent
    ).all()


def test_prioritised_flat_buffer_does_not_smoke(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    priority_exponent: float,
):
    """Create the FlatBuffer NamedTuple, and check that it is pmap-able and does not smoke."""

    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length,
        min_length,
        sample_batch_size,
        False,
        add_batch_size=min_length,
        priority_exponent=priority_exponent,
    )

    # Initialise the buffer's state.
    fake_transition_per_device = jax.tree.map(
        lambda x: jnp.stack([x + i for i in range(_DEVICE_COUNT_MOCK)]), fake_transition
    )
    state = jax.pmap(buffer.init)(fake_transition_per_device)

    # Now fill the buffer above its minimum length.

    fake_batch = jax.pmap(get_fake_batch, static_broadcasted_argnums=1)(
        fake_transition_per_device, min_length
    )
    # Add two items thereby giving a single transition.
    state = jax.pmap(buffer.add)(state, fake_batch)
    state = jax.pmap(buffer.add)(state, fake_batch)
    assert buffer.can_sample(state).all()

    # Sample from the buffer.
    rng_key_per_device = jax.random.split(rng_key, _DEVICE_COUNT_MOCK)
    batch = jax.pmap(buffer.sample)(state, rng_key_per_device)
    chex.assert_tree_shape_prefix(batch, (_DEVICE_COUNT_MOCK, sample_batch_size))

    # Adjust priorities.
    new_priorities = jax.pmap(jnp.ones_like)(batch.priorities) + 10007
    state = jax.pmap(buffer.set_priorities)(state, batch.indices, new_priorities)

    # Check that the priority adjustment produces the correct changes to the state.
    assert (
        state.priority_state.max_recorded_priority
        == jnp.max(new_priorities) ** priority_exponent
    ).all()
    assert (
        jax.pmap(sum_tree.get)(state.priority_state, batch.indices)
        == new_priorities**priority_exponent
    ).all()


def test_add_batch_size_none(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    sample_batch_size: int,
    priority_exponent: float,
):
    # create a fake batch and ensure there is no batch dimension
    fake_batch = jax.tree.map(
        lambda x: jnp.squeeze(x, 0), get_fake_batch(fake_transition, 1)
    )

    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length,
        min_length,
        sample_batch_size,
        False,
        None,
        priority_exponent,
    )
    state = buffer.init(fake_transition)

    init_state = deepcopy(state)  # Save for later checks.

    n_batches_to_fill = max_length
    n_batches_to_sample = min_length

    for i in range(n_batches_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == ((i + 1) % (max_length))

        # Check that the `can_sample` function behavior is correct.
        is_ready_to_sample = buffer.can_sample(state)
        if i < (n_batches_to_sample):
            assert not is_ready_to_sample
        else:
            assert is_ready_to_sample

    assert state.is_full

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)
        chex.assert_trees_all_close(state.indices, init_state.indices)


def test_add_sequences(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    sample_batch_size: int,
    priority_exponent: float,
):
    add_sequence_size = 5
    # create a fake sequence and ensure there is no batch dimension
    fake_batch = jax.tree.map(
        lambda x: x.repeat(add_sequence_size, axis=0),
        get_fake_batch(fake_transition, 1),
    )
    assert fake_batch["obs"].shape[0] == add_sequence_size

    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length,
        min_length,
        sample_batch_size,
        True,
        None,
        priority_exponent,
    )
    state = buffer.init(fake_transition)

    init_state = deepcopy(state)  # Save for later checks.

    n_sequences_to_fill = (max_length // add_sequence_size) + 1

    for i in range(n_sequences_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == (((i + 1) * add_sequence_size) % (max_length))

    assert state.is_full

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)
        chex.assert_trees_all_close(state.indices, init_state.indices)


def test_add_sequences_and_batches(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
    sample_batch_size: int,
    priority_exponent: float,
):
    add_sequence_size = 5
    # create a fake batch and sequence
    fake_batch = jax.tree.map(
        lambda x: x[:, jnp.newaxis].repeat(add_sequence_size, axis=1),
        get_fake_batch(fake_transition, add_batch_size),
    )
    assert fake_batch["obs"].shape[:2] == (add_batch_size, add_sequence_size)

    buffer = prioritised_flat_buffer.make_prioritised_flat_buffer(
        max_length,
        min_length,
        sample_batch_size,
        add_sequences=True,
        add_batch_size=add_batch_size,
        priority_exponent=priority_exponent,
    )

    state = buffer.init(fake_transition)

    init_state = deepcopy(state)  # Save for later checks.

    n_sequences_to_fill = (max_length // add_batch_size // add_sequence_size) + 1

    for i in range(n_sequences_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == (
            ((i + 1) * add_sequence_size) % (max_length // add_batch_size)
        )

    assert state.is_full

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)
        chex.assert_trees_all_close(state.indices, init_state.indices)
