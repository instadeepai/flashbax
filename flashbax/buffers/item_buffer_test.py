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

from flashbax.buffers import item_buffer
from flashbax.buffers.conftest import get_fake_batch
from flashbax.conftest import _DEVICE_COUNT_MOCK


def test_add_and_can_sample(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
) -> None:
    """Check the `add` function by filling the buffer all
    the way to the max_length and checking that it produces the expected behaviour .
    """
    fake_batch = get_fake_batch(fake_transition, add_batch_size)

    buffer = item_buffer.make_item_buffer(max_length, min_length, 4, False, True)
    state = buffer.init(fake_transition)

    init_state = deepcopy(state)  # Save for later checks.

    n_batches_to_fill = max_length // add_batch_size
    n_batches_to_sample = min_length // add_batch_size

    for i in range(n_batches_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == (((i + 1) * add_batch_size) % max_length)

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


def test_sample(
    fake_transition: chex.ArrayTree,
    max_length: int,
    rng_key: chex.PRNGKey,
) -> None:
    """Test the random sampling from the buffer."""

    min_length = 40
    sample_batch_size = 20
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Fill buffer to the point that we can sample
    fake_batch = get_fake_batch(fake_transition, sample_batch_size)

    buffer = item_buffer.make_item_buffer(
        max_length, min_length, sample_batch_size, False, True
    )
    state = buffer.init(fake_transition)

    # Add two batches of items.
    state = buffer.add(state, fake_batch)
    assert not buffer.can_sample(state)
    state = buffer.add(state, fake_batch)
    assert buffer.can_sample(state)

    # Sample from the buffer with different keys and check it gives us different batches.
    batch1 = buffer.sample(state, rng_key1).experience
    batch2 = buffer.sample(state, rng_key2).experience

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(batch1, batch2)

    # Check dtypes are correct.
    chex.assert_trees_all_equal_dtypes(
        fake_transition,
        batch1,
        batch2,
    )


def test_add_batch_size_none(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
):
    # create a fake batch and ensure there is no batch dimension
    fake_batch = jax.tree_map(
        lambda x: jnp.squeeze(x, 0), get_fake_batch(fake_transition, 1)
    )

    buffer = item_buffer.make_item_buffer(max_length, min_length, 4, False, False)
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
        if i < (n_batches_to_sample - 1):
            assert not is_ready_to_sample
        else:
            assert is_ready_to_sample

    assert state.is_full

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)


def test_add_sequences(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
):
    add_sequence_size = 5
    # create a fake sequence and ensure there is no batch dimension
    fake_batch = jax.tree_map(
        lambda x: x.repeat(add_sequence_size, axis=0),
        get_fake_batch(fake_transition, 1),
    )
    assert fake_batch["obs"].shape[0] == add_sequence_size

    buffer = item_buffer.make_item_buffer(
        max_length, min_length, 4, add_sequences=True, add_batches=False
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


def test_add_sequences_and_batches(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    add_batch_size: int,
):
    add_sequence_size = 5
    # create a fake batch and sequence
    fake_batch = jax.tree_map(
        lambda x: x[:, jnp.newaxis].repeat(add_sequence_size, axis=1),
        get_fake_batch(fake_transition, add_batch_size),
    )
    assert fake_batch["obs"].shape[:2] == (add_batch_size, add_sequence_size)

    buffer = item_buffer.make_item_buffer(
        max_length, min_length, 4, add_sequences=True, add_batches=True
    )
    state = buffer.init(fake_transition)

    init_state = deepcopy(state)  # Save for later checks.

    n_sequences_to_fill = (max_length // (add_batch_size * add_sequence_size)) + 1

    for i in range(n_sequences_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == (
            ((i + 1) * add_sequence_size * add_batch_size) % max_length
        )

    assert state.is_full

    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(state.experience, init_state.experience)


def test_item_replay_buffer_does_not_smoke(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
):
    """Create the itemBuffer NamedTuple, and check that it is pmap-able and does not smoke."""

    add_batch_size = int(min_length + 5)
    buffer = item_buffer.make_item_buffer(
        max_length, min_length, sample_batch_size, False, True
    )

    # Initialise the buffer's state.
    fake_transition_per_device = jax.tree_map(
        lambda x: jnp.stack([x + i for i in range(_DEVICE_COUNT_MOCK)]), fake_transition
    )
    state = jax.pmap(buffer.init)(fake_transition_per_device)

    # Now fill the buffer above its minimum length.

    fake_batch = jax.pmap(get_fake_batch, static_broadcasted_argnums=1)(
        fake_transition_per_device, add_batch_size
    )
    # Add two items thereby giving a single transition.
    state = jax.pmap(buffer.add)(state, fake_batch)
    state = jax.pmap(buffer.add)(state, fake_batch)
    assert buffer.can_sample(state).all()

    # Sample from the buffer.
    rng_key_per_device = jax.random.split(rng_key, _DEVICE_COUNT_MOCK)
    batch = jax.pmap(buffer.sample)(state, rng_key_per_device)
    chex.assert_tree_shape_prefix(batch, (_DEVICE_COUNT_MOCK, sample_batch_size))
