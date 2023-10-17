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

from flashbax import utils


@pytest.fixture
def fake_transition() -> chex.ArrayTree:
    return {"obs": jnp.array((5, 4)), "reward": jnp.zeros((3,))}


def get_fake_batch(fake_transition: chex.ArrayTree, batch_size) -> chex.ArrayTree:
    """Create a tree with the same structure as `fake_transition` but with an extra batch axis.
    Each element across the batch dimension is different."""
    return jax.tree_map(
        lambda x: jnp.stack([x + i for i in range(batch_size)]), fake_transition
    )


def test_get_leading_axis_shape(
    fake_transition: chex.ArrayTree,
    batch_size: int = 5,
):
    """Check that get_leading_axis_shape correctly determines the batch size of a tree."""
    fake_batch = get_fake_batch(fake_transition, batch_size)
    batch_size_ = utils.get_tree_shape_prefix(fake_batch, n_axes=1)[0]
    assert batch_size == batch_size_
