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
from typing import NamedTuple

import chex
import flax
import jax
import jax.numpy as jnp
import pytest
from flax import struct

from flashbax.buffers import n_step_buffer
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

    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=4,
        add_sequences=False,
        add_batch_size=add_batch_size,
    )
    state = buffer.init(fake_transition)

    init_state = deepcopy(state)  # Save for later checks.

    n_batches_to_fill = max_length // add_batch_size
    n_batches_to_sample = min_length // add_batch_size

    for i in range(n_batches_to_fill):
        assert not state.is_full
        state = buffer.add(state, fake_batch)
        assert state.current_index == ((i + 1) % (max_length // add_batch_size))

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
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
) -> None:
    """Test the random sampling from the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Fill buffer to the point that we can sample
    fake_batch = get_fake_batch(fake_transition, int(min_length + 10))

    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences=False,
        add_batch_size=int(min_length + 10),
    )
    state = buffer.init(fake_transition)

    # Add two items thereby giving a single transition.
    state = buffer.add(state, fake_batch)
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
        fake_transition, batch1.first, batch1.second, batch2.first, batch2.second
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

    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=4,
        add_sequences=False,
        add_batch_size=None,
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

    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=4,
        add_sequences=True,
        add_batch_size=None,
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

    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=4,
        add_sequences=True,
        add_batch_size=add_batch_size,
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


def test_n_step_replay_buffer_does_not_smoke(
    fake_transition: chex.ArrayTree,
    min_length: int,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
):
    """Create the n_step Buffer NamedTuple, and check that it is pmap-able and does not smoke."""

    add_batch_size = int(min_length + 5)
    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences=False,
        add_batch_size=add_batch_size,
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


def n_step_returns(
    r_t: chex.Array,
    n_step: int,
) -> chex.Array:
    r_t = r_t[:-1]
    n = n_step + 1
    seq_len = r_t.shape[0]
    targets = jnp.zeros(seq_len)
    r_t = jnp.concatenate([r_t, jnp.zeros(n - 1)])

    # Work backwards to compute n-step returns.
    for i in reversed(range(n)):
        r_ = r_t[i : i + seq_len]
        targets = r_ + targets

    return targets


@pytest.mark.parametrize("n_step", [1, 2, 5, 10, 20])
def test_n_step_sample(
    fake_transition: chex.ArrayTree,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    add_batch_size: int,
    n_step: int,
) -> None:
    """Test the random sampling from the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    add_sequence_size = n_step + 10
    # create a fake batch and sequence
    fake_batch = jax.tree_map(
        lambda x: x[:, jnp.newaxis].repeat(add_sequence_size, axis=1) + 1,
        get_fake_batch(fake_transition, add_batch_size),
    )
    assert fake_batch["obs"].shape[:2] == (add_batch_size, add_sequence_size)

    n_step_functional_map = {
        "reward": lambda x: jax.vmap(n_step_returns, in_axes=(0, None))(x, n_step)
    }

    buffer = n_step_buffer.make_n_step_buffer(
        max_length=max_length,
        min_length=n_step,
        sample_batch_size=sample_batch_size,
        add_sequences=True,
        add_batch_size=add_batch_size,
        n_step=n_step,
        n_step_functional_map=n_step_functional_map,
    )
    state = buffer.init(fake_transition)

    # # Add two items thereby giving a single transition.
    state = buffer.add(state, fake_batch)
    state = buffer.add(state, fake_batch)

    assert buffer.can_sample(state)

    # # Sample from the buffer
    batch = buffer.sample(state, rng_key1).experience

    # Since rewards are always the same in a sequence i.e. [1,1,1,1,1] or [2,2,2,2,2] etc
    # The n step return will always be divisible by n_step for e.g.
    # if the rewards are:
    # [1,1] then the n step return will be [2,1]
    # [1,1,1] then the n step return will be [3,2,1]
    # [1,1,1,1] then the n step return will be [4,3,2,1]
    # and since we only take the first and last element of the n step return
    # the first element will always be some multiple of n_step
    assert jnp.all(batch.first["reward"] % n_step == 0)

    for i in range(sample_batch_size):
        # Since we know the n step return of the first element is always divisible by n_step
        # We can work out what the reward sequence value is and that is what the n-step return
        # of the last element should be
        d = batch.first["reward"][i] / n_step
        assert jnp.all(batch.second["reward"][i] == d)


@chex.dataclass
class TestNStepFunctional1:
    obs: chex.Array
    reward: chex.Array
    action: chex.Array


class TestNStepFunctional2(NamedTuple):
    obs: chex.Array
    reward: chex.Array
    action: chex.Array


@struct.dataclass
class TestNStepFunctional3:
    obs: chex.Array
    reward: chex.Array
    action: chex.Array


def test_n_step_functional_with_different_types(
    fake_transition: chex.ArrayTree,
    max_length: int,
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    add_batch_size: int,
) -> None:
    """Test the random sampling from the buffer."""
    rng_key1, rng_key2 = jax.random.split(rng_key)

    n_step = 5
    add_sequence_size = n_step + 10

    orig_fake_transition = fake_transition

    orig_fake_batch = jax.tree_map(
        lambda x: x[:, jnp.newaxis].repeat(add_sequence_size, axis=1) + 1,
        get_fake_batch(orig_fake_transition, add_batch_size),
    )
    assert orig_fake_batch["obs"].shape[:2] == (add_batch_size, add_sequence_size)

    for test_data_type in [
        TestNStepFunctional1,
        TestNStepFunctional2,
        TestNStepFunctional3,
    ]:
        # create a fake batch and sequence
        fake_transition = test_data_type(
            obs=orig_fake_transition["obs"],
            reward=orig_fake_transition["reward"],
            action=orig_fake_transition["action"],
        )
        fake_batch = test_data_type(
            obs=orig_fake_batch["obs"],
            reward=orig_fake_batch["reward"],
            action=orig_fake_batch["action"],
        )

        n_step_functional_map = {
            "reward": lambda x: jax.vmap(n_step_returns, in_axes=(0, None))(x, n_step)
        }

        if test_data_type == TestNStepFunctional2:
            n_step_functional_map["dict_mapper"] = lambda x: x._asdict()

        if test_data_type == TestNStepFunctional3:
            n_step_functional_map[
                "dict_mapper"
            ] = lambda x: flax.serialization.to_state_dict(x)

        buffer = n_step_buffer.make_n_step_buffer(
            max_length=max_length,
            min_length=n_step,
            sample_batch_size=sample_batch_size,
            add_sequences=True,
            add_batch_size=add_batch_size,
            n_step=n_step,
            n_step_functional_map=n_step_functional_map,
        )
        state = buffer.init(fake_transition)

        # # Add two items thereby giving a single transition.
        state = buffer.add(state, fake_batch)
        state = buffer.add(state, fake_batch)

        assert buffer.can_sample(state)

        # # Sample from the buffer
        batch = buffer.sample(state, rng_key1).experience

        assert jnp.all(batch.first.reward % n_step == 0)

        for i in range(sample_batch_size):
            d = batch.first.reward[i] / n_step
            assert jnp.all(batch.second.reward[i] == d)
