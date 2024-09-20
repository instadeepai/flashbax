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


import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flashbax.buffers import (
    flat_buffer,
    prioritised_trajectory_buffer,
    trajectory_buffer,
)
from flashbax.buffers.conftest import get_fake_batch
from flashbax.buffers.mixer import make_mixer


@pytest.fixture
def rng_key() -> chex.PRNGKey:
    return jax.random.PRNGKey(0)


@pytest.fixture
def fake_transition() -> chex.ArrayTree:
    return {
        "obs": jnp.array([0.0, 0.0]),
        "reward": jnp.array(0.0),
        "done": jnp.array(False),
        "next_obs": jnp.array([0.0, 0.0]),
    }


@pytest.fixture
def max_length() -> int:
    return 32


@pytest.fixture
def min_length() -> int:
    return 8


@pytest.fixture
def add_batch_size() -> int:
    return 4


@pytest.fixture
def sample_batch_size() -> int:
    return 100


@pytest.fixture
def sample_sequence_length() -> int:
    return 4


@pytest.fixture
def sample_period() -> int:
    return 1


def get_fake_batch_sequence(
    fake_transition: chex.ArrayTree, batch_size: int, sequence_length: int
) -> chex.ArrayTree:
    return get_fake_batch(get_fake_batch(fake_transition, sequence_length), batch_size)


def test_mixed_trajectory_sample(
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
    add_batch_size: int,
    sample_sequence_length: int,
    fake_transition: chex.ArrayTree,
):
    buffers = []
    buffer_states = []
    for i in range(3):
        buffer = trajectory_buffer.make_trajectory_buffer(
            max_length_time_axis=200 * (i + 1),
            min_length_time_axis=0,
            sample_batch_size=sample_batch_size,
            add_batch_size=add_batch_size,
            sample_sequence_length=sample_sequence_length,
            period=sample_period,
        )
        buffers.append(buffer)

        state = buffer.init(
            jax.tree_map(lambda x, _i=i: jnp.ones_like(x) * _i, fake_transition)
        )
        fake_add_data = get_fake_batch_sequence(fake_transition, add_batch_size, 50)
        fake_add_data = jax.tree_map(
            lambda x, _i=i: jnp.ones_like(x) * _i, fake_add_data
        )
        state = buffer.add(state, fake_add_data)
        buffer_states.append(state)

    proportions = [0.2, 0.2, 0.6]
    mixer = make_mixer(
        buffers=buffers,
        proportions=proportions,
        sample_batch_size=sample_batch_size,
    )
    samples = mixer.sample(buffer_states, rng_key)

    assert samples is not None
    expected_zeros = int(sample_batch_size * proportions[0])
    expected_ones = int(sample_batch_size * proportions[1])
    expected_twos = int(sample_batch_size * proportions[2])
    chex.assert_tree_shape_prefix(samples, (sample_batch_size,))
    dones = samples.experience["done"]
    dones = dones[:, 0]
    assert np.sum(dones == 0) == expected_zeros
    assert np.sum(dones == 1) == expected_ones
    assert np.sum(dones == 2) == expected_twos


def test_mixed_prioritised_trajectory_sample(
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
    add_batch_size: int,
    sample_sequence_length: int,
    fake_transition: chex.ArrayTree,
):
    buffers = []
    buffer_states = []
    for i in range(3):
        buffer = prioritised_trajectory_buffer.make_prioritised_trajectory_buffer(
            max_length_time_axis=200 * (i + 1),
            min_length_time_axis=0,
            sample_batch_size=sample_batch_size,
            add_batch_size=add_batch_size,
            sample_sequence_length=sample_sequence_length,
            period=sample_period,
        )
        buffers.append(buffer)

        state = buffer.init(
            jax.tree_map(lambda x, _i=i: jnp.ones_like(x) * _i, fake_transition)
        )
        fake_add_data = get_fake_batch_sequence(fake_transition, add_batch_size, 50)
        fake_add_data = jax.tree_map(
            lambda x, _i=i: jnp.ones_like(x) * _i, fake_add_data
        )
        state = buffer.add(state, fake_add_data)
        buffer_states.append(state)

    proportions = [0.4, 0.1, 0.5]
    mixer = make_mixer(
        buffers=buffers,
        proportions=proportions,
        sample_batch_size=sample_batch_size,
    )
    samples = mixer.sample(buffer_states, rng_key)

    assert samples is not None
    expected_zeros = int(sample_batch_size * proportions[0])
    expected_ones = int(sample_batch_size * proportions[1])
    expected_twos = int(sample_batch_size * proportions[2])
    chex.assert_tree_shape_prefix(samples, (sample_batch_size,))
    dones = samples.experience["done"]
    dones = dones[:, 0]
    assert np.sum(dones == 0) == expected_zeros
    assert np.sum(dones == 1) == expected_ones
    assert np.sum(dones == 2) == expected_twos


def test_mixed_flat_buffer_sample(
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    add_batch_size: int,
    fake_transition: chex.ArrayTree,
):
    buffers = []
    buffer_states = []
    for i in range(3):
        buffer = flat_buffer.make_flat_buffer(
            max_length=200 * (i + 1),
            min_length=0,
            sample_batch_size=sample_batch_size,
            add_batch_size=add_batch_size,
            add_sequences=True,
        )
        buffers.append(buffer)

        state = buffer.init(
            jax.tree_map(lambda x, _i=i: jnp.ones_like(x) * _i, fake_transition)
        )
        fake_add_data = get_fake_batch_sequence(fake_transition, add_batch_size, 50)
        fake_add_data = jax.tree_map(
            lambda x, _i=i: jnp.ones_like(x) * _i, fake_add_data
        )
        state = buffer.add(state, fake_add_data)
        buffer_states.append(state)

    proportions = [0.1, 0.1, 0.8]
    mixer = make_mixer(
        buffers=buffers,
        proportions=proportions,
        sample_batch_size=sample_batch_size,
    )
    samples = mixer.sample(buffer_states, rng_key)

    assert samples is not None
    expected_zeros = int(sample_batch_size * proportions[0])
    expected_ones = int(sample_batch_size * proportions[1])
    expected_twos = int(sample_batch_size * proportions[2])
    chex.assert_tree_shape_prefix(samples, (sample_batch_size,))
    dones = samples.experience.first["done"]
    assert np.sum(dones == 0) == expected_zeros
    assert np.sum(dones == 1) == expected_ones
    assert np.sum(dones == 2) == expected_twos


def test_mixed_buffer_does_not_smoke(
    rng_key: chex.PRNGKey,
    sample_batch_size: int,
    sample_period: int,
    add_batch_size: int,
    sample_sequence_length: int,
    fake_transition: chex.ArrayTree,
):
    buffers = []
    buffer_states = []
    for i in range(3):
        buffer = trajectory_buffer.make_trajectory_buffer(
            max_length_time_axis=2000 * (i + 1),
            min_length_time_axis=0,
            sample_batch_size=sample_batch_size,
            add_batch_size=add_batch_size,
            sample_sequence_length=sample_sequence_length,
            period=sample_period,
        )
        buffers.append(buffer)

        state = buffer.init(
            jax.tree_map(lambda x, _i=i: jnp.ones_like(x) * _i, fake_transition)
        )
        fake_add_data = get_fake_batch_sequence(fake_transition, add_batch_size, 50)
        fake_add_data = jax.tree_map(
            lambda x, _i=i: jnp.ones_like(x) * _i, fake_add_data
        )
        state = buffer.add(state, fake_add_data)
        buffer_states.append(state)

    proportions = [0.2, 0.2, 0.6]
    mixer = make_mixer(
        buffers=buffers,
        proportions=proportions,
        sample_batch_size=sample_batch_size,
    )

    can_sample = jax.jit(mixer.can_sample)(buffer_states)
    assert can_sample

    samples = jax.jit(mixer.sample)(buffer_states, rng_key)

    assert samples is not None
    expected_zeros = int(sample_batch_size * proportions[0])
    expected_ones = int(sample_batch_size * proportions[1])
    expected_twos = int(sample_batch_size * proportions[2])
    chex.assert_tree_shape_prefix(samples, (sample_batch_size,))
    dones = samples.experience["done"]
    dones = dones[:, 0]
    assert np.sum(dones == 0) == expected_zeros
    assert np.sum(dones == 1) == expected_ones
    assert np.sum(dones == 2) == expected_twos
