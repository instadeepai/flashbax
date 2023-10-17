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


"""
 Pure functions defining a sum-tree data structure. The desired use is within a prioritised replay
 buffer, see Prioritized Experience Replay by Schaul et al. (2015) and `prioritised_replay.py`.

 This is an adaption of the sum-tree implementation from
 dopamine: https://github.com/google/dopamine/blob/master/dopamine/replay_memory/sum_tree.py.
 Lots of the code is verbatim copied.
 The key differences between this implementation and the dopamine implementation are (1) This
 implementation is in jax with a functional style, and (2) this implementation focuses on
 vectorised adding (rather sequential adding).
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array


@dataclass
class SumTreeState:
    nodes: Array
    max_recorded_priority: Array
    tree_depth: int = struct.field(pytree_node=False)
    capacity: int = struct.field(pytree_node=False)


def get_tree_depth(capacity: int) -> int:
    """Returns the depth of the sum tree.

    Args:
        capacity: The maximum number of elements that can be stored in this
            data structure.

    Returns:
        The depth of the sum tree.
    """
    return int(np.ceil(np.log2(capacity)))


def init(capacity: int) -> SumTreeState:
    """Creates the sum tree data structure for the given replay capacity.

    Args:
        capacity: The maximum number of elements that can be stored in this
            data structure.
    """
    tree_depth = get_tree_depth(capacity)
    array_size = 2 ** (tree_depth + 1) - 1
    nodes = jnp.zeros(array_size)
    max_recorded_priority = jnp.array(1.0)
    return SumTreeState(
        nodes=nodes,
        max_recorded_priority=max_recorded_priority,
        tree_depth=tree_depth,
        capacity=capacity,
    )


def _total_priority(state: SumTreeState) -> Array:
    """Returns the sum of all priorities stored in this sum tree.

    Returns:
        Sum of priorities stored in this sum tree.
    """
    # The root node contains the sum of all priorities.
    return state.nodes[0]


def get_tree_index(
    depth_level: Union[Array, int], node_index: Union[Array, int]
) -> Array:
    """Returns the index of the node in the sum tree.

    Since the actual nodes are stored in a flat array, we need to map the index of the node at a
    certain depth level to the index of the node in the flat array. This is easy to do as we always
    create trees where the number of nodes at each level is a power of 2. As such we can simply
    take 2^depth_level - 1 to get the off set of the first node at that depth level.
    We then add the node_index to get the index of the node in the flat array.

    Args:
        depth_level: The depth level of the node.
        node_index: The index of the node at that depth level.

    Returns:
        The index of the node in the flat array.
    """
    return jnp.power(2, depth_level) - 1 + node_index


def sample(
    state: SumTreeState,
    rng_key: Optional[chex.PRNGKey] = None,
    query_value: Optional[Array] = None,
) -> Array:
    """Samples an element from the sum tree.

    Each element has probability p_i / sum_j p_j of being picked, where p_i is
    the (positive) value associated with node i (possibly unnormalized).

    Args:
        state: Current state of the sum-tree.
        rng_key: Source of randomness for sampling randomly from the buffer if `query_value`
            is None.
        query_value: float in [0, 1], used as the random value to select a
            sample. If None, will select one randomly in [0, 1).
    Returns:
        A random element from the sum tree.
    """

    # Sample a value in range [0, R), where R is the value stored at the root.
    if query_value is None:
        if rng_key is None:
            raise ValueError(
                "Either the `rng_key` or the `query_value` must be specified."
            )
        query_value = jax.random.uniform(rng_key)
    query_value = query_value * _total_priority(state)

    # Now traverse the sum tree.
    node_index = jnp.array(0, dtype=int)

    # starting at depth 1 (not 0), we traverse the tree until we reach the bottom.
    def get_node_index(
        depth_level: Array, carry: Tuple[Array, Array, Array]
    ) -> Tuple[Array, Array, Array]:
        """Traverses the sum tree to find the index of the leaf node to sample from."""
        nodes, node_index, query_value = carry

        left_child_index = node_index * 2

        mapped_left_child_index = get_tree_index(depth_level, left_child_index)

        left_sum = nodes[mapped_left_child_index]

        query_value, node_index = jax.lax.cond(
            query_value < left_sum,
            lambda query_value, left_child_index, _: (query_value, left_child_index),
            lambda query_value, left_child_index, left_sum: (
                query_value - left_sum,
                left_child_index + 1,
            ),
            query_value,
            left_child_index,
            left_sum,
        )

        return nodes, node_index, query_value

    _, node_index, _ = jax.lax.fori_loop(
        1, state.tree_depth + 1, get_node_index, (state.nodes, node_index, query_value)
    )

    return node_index


def stratified_sample(
    state: SumTreeState,
    batch_size: int,
    rng_key: chex.PRNGKey,
) -> Array:
    """Performs stratified sampling using the sum tree.

    Let R be the value at the root (total value of sum tree). This method will
    divide [0, R) into batch_size segments, pick a random number from each of
    those segments, and use that random number to sample from the sum_tree. This
    is as specified in Schaul et al. (2015).

    Args:
        state: Current state of the sum-tree.
        batch_size: The number of strata to use.
        rng_key: Key to use for random number generation.

    Returns:
        Batch of indices sampled from the sum tree.
    """
    query_keys = jax.random.split(rng_key, batch_size)
    bounds = jnp.linspace(0.0, 1.0, batch_size + 1)

    lower_bounds = bounds[:-1, None]
    upper_bounds = bounds[1:, None]

    query_values = jax.vmap(jax.random.uniform, in_axes=(0, None, None, 0, 0))(
        query_keys, (), jnp.float32, lower_bounds, upper_bounds
    )

    return jax.vmap(sample, in_axes=(None, None, 0))(
        state, None, query_values.squeeze()
    )


def get(state: SumTreeState, node_index: Array) -> Array:
    """Returns the value of the leaf node corresponding to the index.

    Args:
        state: Current state of the sum-tree.
        node_index: The index of the leaf node.

    Returns:
        The value of the leaf node.
    """
    tree_index = get_tree_index(state.tree_depth, node_index)
    return state.nodes[tree_index]


def get_batch(state: SumTreeState, node_indices: Array) -> Array:
    """Returns the value of all the leaf nodes corresponding to the indices.

    Args:
        state: Current state of the sum-tree.
        node_indices: The indices of the leaf nodes.

    Returns:
        The values of the leaf nodes.
    """
    tree_indices = jax.vmap(get_tree_index, in_axes=(None, 0))(
        state.tree_depth, node_indices
    )
    return state.nodes[tree_indices]


def set_non_batched(
    state: SumTreeState,
    node_index: Array,
    value: Array,
) -> SumTreeState:
    """Sets the value of a leaf node and updates internal nodes accordingly.
    This operation takes O(log(capacity)).

    Args:
        state: Current state of the sum-tree.
        node_index: The index of the leaf node to be updated.
        value: The value which we assign to the node. This value must be
            nonnegative. Setting value = 0 will cause the element to never be sampled.

    This is not used in practice within the prioritised replay, as it is not vmap-able. However,
    it is useful for comparisons and testing. See `set_single` and `set_batch` for the functions
    that we use within the prioritised replay in practice.
    """
    # We get the tree index of the node.
    mapped_index = get_tree_index(state.tree_depth, node_index)
    # We get the delta value for the node.
    delta_value = value - state.nodes[mapped_index]

    def update_nodes(
        idx: Array, carry: Tuple[Array, Array, Array, Array]
    ) -> Tuple[Array, Array, Array, Array]:
        """Traverses the sum tree to update the parent nodes and add all the delta values."""
        nodes, node_index, delta_value, tree_depth = carry
        depth_level = tree_depth - idx
        mapped_index = get_tree_index(depth_level, node_index)
        nodes = nodes.at[mapped_index].add(delta_value)
        node_index //= 2

        return (nodes, node_index, delta_value, tree_depth)

    # We propagate the delta value up the tree.
    (nodes, _, _, _) = jax.lax.fori_loop(
        0,
        state.tree_depth + 1,
        update_nodes,
        (state.nodes, node_index, delta_value, state.tree_depth),
    )

    # We get the max priority and replace the current max priority with the maximum of the two.
    new_max_recorded_priority = jnp.maximum(value, state.max_recorded_priority)
    # We then update the state.
    state = state.replace(  # type: ignore
        nodes=nodes, max_recorded_priority=new_max_recorded_priority
    )

    return state


def set_batch_bincount(
    state: SumTreeState, node_indices: Array, values: Array
) -> SumTreeState:
    """Sets the values of a batch of leaf nodes and updates the parent nodes accordingly.

    As sampling sometimes returns repeats of the same index, we deal with duplicates by
    setting the leaf to the "latest" value i.e. the value that has the highest "index" in
    the values array. This method makes use of `jax.numpy.bincount` to deal with duplicates.
    Since "scan" operations can be slow on TPUs, this method is faster than `set_batch_scan`
    on TPUs for practical batch sizes.

    Args:
        state: Current state of the sum-tree.
        node_indices: The indices of the leaf nodes to be updated.
        values: The values which we assign to the nodes. These values must be
            non-negatives. Setting value = 0 will cause the element to never be sampled.

    Returns:
        A buffer state with updates nodes.
    """
    # We get the tree indices of the nodes.
    mapped_indices = get_tree_index(state.tree_depth, node_indices)

    # We perform the first set operation outside of the loop to deal with duplicates.
    # This is because we want to set the value of the leaf nodes to the 'latest' value
    # in the batch of values and we do not want to add the delta values of all duplicate
    # nodes.
    new_nodes = state.nodes.at[mapped_indices].set(values)
    # We calculate the delta values for each node using the original values.
    # We do it like this to deal with duplicates as we want each delta
    # value for duplicates to be identical.
    delta_values = new_nodes[mapped_indices] - state.nodes[mapped_indices]
    # Determine the counts of the duplicate indices
    index_counts = jnp.bincount(node_indices, length=state.capacity)
    # We get the number of duplicates for each leaf node being updated.
    divisor = index_counts[node_indices]
    # We then divide the delta values by the number of duplicates
    # since they are added together in the tree propagation
    # e.g. if two duplicate leafs are being updated - since their delta values are
    # identical (by construction) we can divide by their deltas by 2 to get the
    # correct delta update.
    delta_values = delta_values / divisor

    # We then update the node indices to be the parent nodes.
    node_indices //= 2

    # This function then performs tree traversal to update the parent nodes and add all
    # the delta values.
    def update_nodes(i: Array, carry: Tuple[Array, Array, Array, Array]):
        """Traverses the sum tree to update the parent nodes and add all the delta values."""
        nodes, node_indices, delta_values, tree_depth = carry
        depth_level = tree_depth - i
        mapped_indices = get_tree_index(depth_level, node_indices)
        nodes = nodes.at[mapped_indices].add(delta_values)
        node_indices //= 2

        return (nodes, node_indices, delta_values, tree_depth)

    # We loop from the second level of the tree to the root since we have already updated
    # the leaf nodes.
    (new_nodes, _, _, _) = jax.lax.fori_loop(
        1,
        state.tree_depth + 1,
        update_nodes,
        (new_nodes, node_indices, delta_values, state.tree_depth),
    )

    # We get the max priority and replace the current max priority with the maximum of the two.
    max_recorded_priority = jnp.maximum(jnp.max(values), state.max_recorded_priority)
    state = state.replace(  # type: ignore
        nodes=new_nodes, max_recorded_priority=max_recorded_priority
    )

    return state


def set_batch_scan(
    state: SumTreeState, node_indices: Array, values: Array
) -> SumTreeState:
    """Sets the values of a batch of leaf nodes and updates the parent nodes accordingly.

    As sampling sometimes returns repeats of the same index, we deal with duplicates by setting
    the leaf to the "latest" value i.e. the value that has the highest index in the values array.
    This method makes use of `jax.lax.scan` to deal with duplicates. Since "scan" operations can
    be slow on TPUs, this method is slower than `set_batch_bincount` on TPUs for practical batch
    sizes. However, this method is faster than `set_batch_bincount` on CPUs.

    Args:
        state: Current state of the buffer.
        node_indices: The indices of the leaf nodes to be updated.
        values: The values which we assign to the nodes. These values must be
            non-negatives. Setting value = 0 will cause the element to never be sampled.

    Returns:
        A buffer state with updates nodes.
    """

    def update_node_priority(state: SumTreeState, node_data: Tuple[Array, Array]):
        """Updates the priority of a single node."""
        node_index, node_value = node_data
        state = set_non_batched(state, node_index, node_value)
        return state, None

    # We scan over the nodes and update them one by one which makes the complexity O(BLogN).
    state, _ = jax.lax.scan(update_node_priority, state, (node_indices, values))

    return state
