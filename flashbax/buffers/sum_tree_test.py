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

from flashbax.buffers import sum_tree


@pytest.fixture
def capacity() -> int:
    return 256


@pytest.fixture
def state(capacity: int) -> sum_tree.SumTreeState:
    """Initial state of the sum-tree."""
    return sum_tree.init(capacity=capacity)


def test_set_non_batched(state: sum_tree.SumTreeState) -> None:
    """Test that `set_non_batched` results in the expected updates to the
    sum-tree state."""

    # Set a node index to a value.
    node_index1 = jnp.int32(2)
    value1 = jnp.float32(10)
    state = sum_tree.set_non_batched(state, node_index1, value1)

    # Check the leaf node is correctly set.
    access_index = sum_tree.get_tree_index(state.tree_depth, node_index1)
    assert state.nodes[access_index] == value1

    # As the sum-tree was initially empty, the root note, and max_recorded_priority
    # match the value assigned to the leaf.
    assert state.nodes[0] == value1
    assert state.max_recorded_priority == value1

    node_index2 = jnp.int32(7)
    value2 = jnp.float32(20)
    state = sum_tree.set_non_batched(state, node_index2, value2)

    # Check the leaf node is correctly set.
    access_index = sum_tree.get_tree_index(state.tree_depth, node_index2)
    assert state.nodes[access_index] == value2
    # Assert that root is the sum of the leaf nodes.
    assert state.nodes[0] == value1 + value2
    # Assert that max_recorded_priority is the maximum of the leaf nodes.
    assert state.max_recorded_priority == jnp.maximum(value1, value2)

    # check overwriting a value works.
    value3 = jnp.float32(30)
    state = sum_tree.set_non_batched(state, node_index2, value3)
    assert state.nodes[access_index] == value3
    assert state.nodes[0] == value1 + value3
    assert state.max_recorded_priority == jnp.maximum(value1, value3)


def test_set_batch_matches_set_non_batched(
    state: sum_tree.SumTreeState,
) -> None:
    """Check that setting a batch of values results in the same behavior
    as setting value sequentially using `set_non_batched`."""
    init_state = deepcopy(state)  # Save for reuse.
    indexes = jnp.arange(10) + 15
    rng_key = jax.random.PRNGKey(0)
    indexes = jax.random.permutation(rng_key, indexes)
    values = jnp.array(11.0) + jnp.arange(10)

    # Create a state by sequentially assigning values to indices.
    for index, value in zip(indexes, values):
        state = sum_tree.set_non_batched(state, index, value)
    state_from_sequential_setting = state  # Save for later comparison.

    # Now create the state by assigning values to indices in two batches.
    # Perform this check for both the scan and bincount implementations.
    state_bincount = init_state  # Re-init the state.
    split_index = 3  # Split index and values into 2 different batches.
    state_bincount = sum_tree.set_batch_bincount(
        state_bincount,
        indexes[:split_index],
        values[:split_index],
    )
    state_bincount = sum_tree.set_batch_bincount(
        state_bincount,
        indexes[split_index:],
        values[split_index:],
    )

    state_scan = init_state  # Re-init the state.
    state_scan = sum_tree.set_batch_scan(
        state_scan,
        indexes[:split_index],
        values[:split_index],
    )
    state_scan = sum_tree.set_batch_scan(
        state_scan,
        indexes[split_index:],
        values[split_index:],
    )

    # Check that the states created the two different ways match.
    chex.assert_trees_all_close(
        state_scan, state_bincount, state_from_sequential_setting
    )


def test_set_batch_clashing_values(state: sum_tree.SumTreeState) -> None:
    """Check that if there is a repeated index that this results in the latest value out of the
    values corresponding to the repeated value being set."""
    num_indices = 10
    # Create set of repeated indices. Each index will be repeated once.
    repeated_indices = jnp.concatenate(
        [jnp.arange(num_indices), jnp.arange(num_indices)]
    )
    # Create a set of values that will be assigned to the repeated indices.
    values = jnp.arange(num_indices * 2) * 100
    # Set the values in the sum-tree using both the scan and bincount implementations.
    state_scan = sum_tree.set_batch_scan(state, repeated_indices, values)
    state_bincount = sum_tree.set_batch_bincount(state, repeated_indices, values)
    # get the values at the repeated indices.
    repeated_vals_scan = sum_tree.get_batch(state_scan, repeated_indices[:num_indices])
    repeated_vals_bincount = sum_tree.get_batch(
        state_bincount, repeated_indices[:num_indices]
    )

    # At the repeated index, the values should match the latest of that from `values`.
    assert jnp.all(repeated_vals_scan == values[num_indices:]) and jnp.all(
        repeated_vals_bincount == values[num_indices:]
    )

    # Check that the root node is the sum of the leaf nodes.
    assert state_scan.nodes[0] == jnp.sum(repeated_vals_scan)
    assert state_bincount.nodes[0] == jnp.sum(repeated_vals_bincount)
    assert state_scan.max_recorded_priority == jnp.max(repeated_vals_scan)
    assert state_bincount.max_recorded_priority == jnp.max(repeated_vals_bincount)

    # Check that the states created the two different ways match.
    chex.assert_trees_all_close(
        state_scan,
        state_bincount,
    )


def test_sample(
    state: sum_tree.SumTreeState,
    rng_key: chex.PRNGKey,
    batch_size: int = 32,
) -> None:
    """Check sampling works by sampling and checking that the result matches the indexes in
    the sumtree, and that it is random."""

    total_size = state.nodes.shape[0]
    bottom_depth = state.tree_depth
    starting_index = 2**bottom_depth - 1
    actual_size = total_size - starting_index
    assert batch_size < actual_size

    # Fill state enough that we can sample.
    indexes = jnp.arange(batch_size, dtype=int)
    # make it uniform so that we can check that the sampling is random.
    values = jnp.ones_like(indexes, dtype=jnp.float32)
    state = sum_tree.set_batch_bincount(state, indexes, values)

    # Sample and check that the result is expected.
    rng_key, rng_key1 = jax.random.split(rng_key)
    index = sum_tree.sample(state, rng_key1)
    assert index in indexes

    # Check that we get different samples when we change the random key.
    rng_key, rng_key2 = jax.random.split(rng_key)
    index_alt = sum_tree.sample(state, rng_key2)
    assert index_alt != index


def test_stratified_sample(
    state: sum_tree.SumTreeState, rng_key: chex.PRNGKey, batch_size: int = 32
) -> None:
    """Check stratified sampling works by sampling and checking that the result matches the
    indexes in the sumtree, and that it is random."""

    # Fill state enough that we can sample.
    indexes = jnp.arange(batch_size + 15, dtype=int)
    values = indexes + 10.0
    state = sum_tree.set_batch_bincount(state, indexes, values)

    # Perform stratified sampling and check results match what is expected.
    rng_key1, rng_key2 = jax.random.split(rng_key)
    sampled_indices = sum_tree.stratified_sample(state, batch_size, rng_key1)
    chex.assert_shape(sampled_indices, (batch_size,))
    for index in sampled_indices:
        assert index in indexes

    # Check that we get different samples when we change the random key.
    sampled_indices_alt = sum_tree.stratified_sample(state, batch_size, rng_key2)
    # Check that the transitions have been updated.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(sampled_indices, sampled_indices_alt)


def test_set_batch_scan_matches_set_batch_bincount(
    state: sum_tree.SumTreeState,
) -> None:
    """Check that setting a batch of values results in the same behavior
    as setting value sequentially using `set_non_batched`."""
    init_state = deepcopy(state)  # Save for reuse.
    indexes = jnp.arange(100)
    rng_key1 = jax.random.PRNGKey(0)
    rng_key2 = jax.random.PRNGKey(42)
    # Shuffle the indexes so that they are not in order.
    # This is to check that the order of the indexes does not matter.
    indexes1 = jax.random.permutation(rng_key1, indexes)
    indexes2 = jax.random.permutation(rng_key2, indexes)
    values = jnp.array(11.0) + jnp.arange(100)
    values1 = jax.random.permutation(rng_key1, values)
    values2 = jax.random.permutation(rng_key2, values)

    assert jnp.all(indexes1 != indexes2)
    assert jnp.all(values1 != values2)

    state_bincount = init_state  # Re-init the state.

    state_bincount = sum_tree.set_batch_bincount(
        state_bincount,
        indexes1,
        values1,
    )

    state_scan = init_state  # Re-init the state.
    state_scan = sum_tree.set_batch_scan(
        state_scan,
        indexes2,
        values2,
    )

    # Check that the states created the two different ways match.
    chex.assert_trees_all_close(state_scan, state_bincount)

    # Check that they handle duplicate values identically
    duplicate_indexes = jnp.array([0, 0, 0, 0, 15, 15, 15, 15, 15, 15])
    values = jnp.arange(duplicate_indexes.shape[0])
    state_scan = sum_tree.set_batch_scan(
        state_scan,
        duplicate_indexes,
        values,
    )
    state_bincount = sum_tree.set_batch_bincount(
        state_bincount,
        duplicate_indexes,
        values,
    )

    chex.assert_trees_all_close(state_scan, state_bincount)


def test_is_jittable(
    capacity: int, rng_key: chex.PRNGKey, sampling_batch_size: int = 8
) -> None:
    """Check that sumtree functions that will be used by the prioritised buffer may be jitted."""
    state = jax.jit(sum_tree.init, static_argnums=0)(capacity=capacity)
    # Fill the sum-tree above the sampling batch_size.
    indexes = jnp.arange(sampling_batch_size + 15, dtype=int)
    values = indexes + 10.0
    state = jax.jit(sum_tree.set_batch_bincount)(
        state,
        indexes,
        values,
    )
    state = jax.jit(sum_tree.set_batch_scan)(
        state,
        indexes,
        values,
    )
    sampled_indices = jax.jit(sum_tree.stratified_sample, static_argnums=(1,))(
        state, sampling_batch_size, rng_key
    )
    chex.assert_shape(sampled_indices, (sampling_batch_size,))
