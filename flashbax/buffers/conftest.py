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
import pytest


@pytest.fixture
def max_length() -> int:
    return 256


@pytest.fixture
def sample_batch_size() -> int:
    return 64


@pytest.fixture
def min_length(sample_batch_size: int) -> int:
    return int(sample_batch_size + 1)


@pytest.fixture()
def add_batch_size() -> int:
    return 2


@pytest.fixture
def fake_transition() -> chex.ArrayTree:
    return {
        "obs": jnp.array((5, 4), dtype=jnp.float32),
        "action": jnp.ones((2,), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float16),
    }


def get_fake_batch(fake_transition: chex.ArrayTree, batch_size) -> chex.ArrayTree:
    """Create a fake batch with differing values for each transition."""
    return jax.tree_map(
        lambda x: jnp.stack([x + i for i in range(batch_size)]), fake_transition
    )
