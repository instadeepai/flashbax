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


from functools import partial
from tempfile import TemporaryDirectory
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import Array

import flashbax as fbx
from flashbax.vault import Vault


class CustomObservation(NamedTuple):
    x: Array
    y: Array


class FbxTransition(NamedTuple):
    obs: CustomObservation
    act: Array


@pytest.fixture()
def max_length() -> int:
    return 256


@pytest.fixture()
def fake_transition() -> FbxTransition:
    return FbxTransition(
        obs=CustomObservation(
            x=jnp.ones(shape=(1, 2, 3), dtype=jnp.float32),
            y=jnp.ones(shape=(4, 5, 6), dtype=jnp.float32),
        ),
        act=jnp.ones(shape=(7, 8), dtype=jnp.float32),
    )


def test_write_to_vault(
    fake_transition: FbxTransition,
    max_length: int,
):
    with TemporaryDirectory() as temp_dir_path:
        # Get the buffer pure functions
        buffer = fbx.make_flat_buffer(
            max_length=max_length,
            min_length=1,
            sample_batch_size=1,
        )
        buffer_add = jax.jit(buffer.add, donate_argnums=0)
        buffer_state = buffer.init(fake_transition)  # Initialise the state

        # Initialise the vault
        v = Vault(
            vault_name="test_vault",
            experience_structure=buffer_state.experience,
            rel_dir=temp_dir_path,
        )

        # Add to the vault up to the fbx buffer being full
        for i in range(0, max_length):
            assert v.vault_index == i
            buffer_state = buffer_add(
                buffer_state,
                fake_transition,
            )
            v.write(buffer_state)


def test_read_from_vault(
    fake_transition: FbxTransition,
    max_length: int,
):
    with TemporaryDirectory() as temp_dir_path:
        # Get the buffer pure functions
        buffer = fbx.make_flat_buffer(
            max_length=max_length,
            min_length=1,
            sample_batch_size=1,
        )
        buffer_add = jax.jit(buffer.add, donate_argnums=0)
        buffer_state = buffer.init(fake_transition)  # Initialise the state

        # Initialise the vault
        v = Vault(
            vault_name="test_vault",
            experience_structure=buffer_state.experience,
            rel_dir=temp_dir_path,
        )

        for _ in range(0, max_length):
            buffer_state = buffer_add(
                buffer_state,
                fake_transition,
            )
            v.write(buffer_state)

        # Load the state from the vault
        buffer_state_reloaded = v.read()
        # Experience of the two should match
        chex.assert_trees_all_equal(
            buffer_state.experience,
            buffer_state_reloaded.experience,
        )


def test_extend_vault(
    fake_transition: FbxTransition,
    max_length: int,
):
    # Extend the vault more than the buffer size
    n_timesteps = max_length * 10

    with TemporaryDirectory() as temp_dir_path:
        # Get the buffer pure functions
        buffer = fbx.make_flat_buffer(
            max_length=max_length,
            min_length=1,
            sample_batch_size=1,
        )
        buffer_add = jax.jit(buffer.add, donate_argnums=0)
        buffer_state = buffer.init(fake_transition)  # Initialise the state

        # Initialise the vault
        v = Vault(
            vault_name="test_vault",
            experience_structure=buffer_state.experience,
            rel_dir=temp_dir_path,
        )

        # Add to the vault, wrapping around the circular buffer,
        # but writing to the vault each time we add
        for _ in range(0, n_timesteps):
            buffer_state = buffer_add(
                buffer_state,
                fake_transition,
            )
            v.write(buffer_state)

        # Read in the full vault state --> longer than the buffer
        long_buffer_state = v.read()

        # We want to check that all the timesteps are there
        assert long_buffer_state.experience.obs.x.shape[1] == n_timesteps


def test_reload_vault(
    fake_transition: FbxTransition,
    max_length: int,
):
    with TemporaryDirectory() as temp_dir_path:
        # Get the buffer pure functions
        buffer = fbx.make_flat_buffer(
            max_length=max_length,
            min_length=1,
            sample_batch_size=1,
        )
        buffer_add = jax.jit(buffer.add, donate_argnums=0)
        buffer_state = buffer.init(fake_transition)  # Initialise the state

        # Initialise the vault
        v = Vault(
            vault_name="test_vault",
            experience_structure=buffer_state.experience,
            rel_dir=temp_dir_path,
            vault_uid="test_vault_uid",
        )

        def multiplier(x: Array, i: int):
            return x * i

        # Add to the vault
        for i in range(0, max_length):
            buffer_state = buffer_add(
                buffer_state,
                jax.tree_map(partial(multiplier, i=i), fake_transition),
            )
            v.write(buffer_state)

        # Ensure we can't access the vault
        del v

        # Reload the vault
        v_reload = Vault(
            vault_name="test_vault",
            experience_structure=buffer_state.experience,
            rel_dir=temp_dir_path,
            vault_uid="test_vault_uid",  # Need to pass the same UID
        )
        buffer_state_reloaded = v_reload.read()

        # We want to check that all the data is correct
        chex.assert_trees_all_equal(
            buffer_state.experience,
            buffer_state_reloaded.experience,
        )
