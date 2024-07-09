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

from typing import Any, Callable, Sequence

import chex
import jax
import jax.numpy as jnp

from flashbax.buffers.trajectory_buffer import BufferState


def mixed_sample(
    buffer_state_list: Sequence[BufferState],
    rng_key: chex.Array,
    buffer_sample_fns: Sequence[Callable[[BufferState], Any]],
    proportions: Sequence[float],
    sample_batch_size: int,
) -> Any:
    """
    Sample from a mixed buffer, which is a list of buffer states, each with its own sample function.

    Each buffer sample needs to be of the same pytree structure, and the samples are concatenated along the first axis i.e. the batch axis.
    For example, if you are sampling trajectories, then all samples need to be sequences of the same sequence length but batch sizes can differ.
    """
    assert len(buffer_state_list) == len(
        buffer_sample_fns
    ), "Number of buffer states and sample functions must match"
    assert len(buffer_state_list) == len(
        proportions
    ), "Number of buffer states and proportions must match"
    assert sum(proportions) == 1.0, "Proportions must sum to 1"
    local_batch_sizes = [int(sample_batch_size * p) for p in proportions]
    if sum(local_batch_sizes) != sample_batch_size:
        local_batch_sizes[-1] += sample_batch_size - sum(local_batch_sizes)
    # Sample from each buffer
    buffer_samples = []
    for buffer_idx, buffer_state in enumerate(buffer_state_list):
        rng_key, sample_key = jax.random.split(rng_key)
        buffer_state = buffer_state_list[buffer_idx]
        buffer_sample_fn = buffer_sample_fns[buffer_idx]
        sampled_data = buffer_sample_fn(buffer_state, sample_key)
        size_to_sample = local_batch_sizes[buffer_idx]
        sampled_data = jax.tree_map(lambda x: x[:size_to_sample], sampled_data)
        buffer_samples.append(sampled_data)

    # Concatenate the samples
    buffer_samples = jax.tree.map(
        lambda *x: jnp.concatenate(x, axis=0), *buffer_samples
    )

    return buffer_samples


def joint_mixed_add(
    buffer_state_list: Sequence[BufferState],
    data: Any,
    buffer_add_fns: Sequence[Callable[[BufferState, Any], BufferState]],
) -> Sequence[BufferState]:
    """
    Add data to a mixed buffer, which is a list of buffer states, each with its own add function.
    """
    assert len(buffer_state_list) == len(
        buffer_add_fns
    ), "Number of buffer states and add functions must match"
    for buffer_idx, buffer_state in enumerate(buffer_state_list):
        buffer_state = buffer_state_list[buffer_idx]
        buffer_add_fn = buffer_add_fns[buffer_idx]
        buffer_state = buffer_add_fn(buffer_state, data)
        buffer_state_list[buffer_idx] = buffer_state
    return buffer_state_list
