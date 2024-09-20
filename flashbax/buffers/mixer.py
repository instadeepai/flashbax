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

import functools
from typing import Callable, Sequence, TypeVar

import chex
import jax
import jax.numpy as jnp
from chex import Numeric, dataclass
from jax import Array
from jax.tree_util import tree_map

from flashbax.buffers.flat_buffer import TransitionSample
from flashbax.buffers.prioritised_trajectory_buffer import (
    PrioritisedTrajectoryBuffer,
    PrioritisedTrajectoryBufferSample,
    PrioritisedTrajectoryBufferState,
)
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
)

# Support for Trajectory, Flat, Item buffers, and prioritised variants
sample_types = [
    TrajectoryBufferSample,
    PrioritisedTrajectoryBufferSample,
    TransitionSample,
]
SampleTypes = TypeVar(
    "SampleTypes",
    TrajectoryBufferSample,
    PrioritisedTrajectoryBufferSample,
    TransitionSample,
)

state_types = [TrajectoryBufferState, PrioritisedTrajectoryBufferState]
StateTypes = TypeVar(
    "StateTypes", TrajectoryBufferState, PrioritisedTrajectoryBufferState
)

BufferTypes = TypeVar("BufferTypes", TrajectoryBuffer, PrioritisedTrajectoryBuffer)


@dataclass(frozen=True)
class Mixer:
    """Pure functions defining the mixer.

    Attributes:
        sample (Callable): function to sample proportionally from all buffers,
            concatenating along the batch axis
        can_sample (Callable): function to check if all buffers can sample
    """

    sample: Callable
    can_sample: Callable


def _batch_slicer(
    sample: SampleTypes,
    batch_start: int,
    batch_end: int,
) -> SampleTypes:
    """Simple utility function to slice a sample along the batch axis.

    Args:
        sample (SampleTypes): incoming sample
        batch_start (int): batch start index
        batch_end (int): batch end index

    Returns:
        SampleTypes: outgoing sliced sample
    """
    return tree_map(lambda x: x[batch_start:batch_end, ...], sample)


def sample_mixer_fn(
    states: Sequence[StateTypes],
    key: chex.PRNGKey,
    prop_batch_sizes: Sequence[int],
    sample_fns: Sequence[Callable[[StateTypes, chex.PRNGKey], SampleTypes]],
) -> SampleTypes:
    """Perform mixed sampling from provided buffer states, according to provided proportions.

    Each buffer sample needs to be of the same pytree structure, and the samples are concatenated
    along the first axis i.e. the batch axis. For example, if you are sampling trajectories, then
    all samples need to be sequences of the same sequence length but batch sizes can differ.

    Args:
        states (Sequence[StateTypes]): list of buffer states
        key (chex.PRNGKey): random key
        prop_batch_sizes (Sequence[Numeric]): list of batch sizes sampled from each buffer,
            calculated according to the proportions of joint sample size
        sample_fns (Sequence[Callable[[StateTypes, chex.PRNGKey], SampleTypes]]): list of pure
            sample functions from each buffer

    Returns:
        SampleTypes: proportionally concatenated samples from all buffers
    """
    keys = jax.random.split(
        key, len(states)
    )  # Split the key for each buffer sampling operation

    # We first sample from each buffer, and get a list of samples
    samples_array = tree_map(
        lambda state, sample, key_in: sample(state, key_in),
        states,
        sample_fns,
        list(keys),
        is_leaf=lambda leaf: type(leaf) in state_types,
    )

    # We then slice the samples according to the proportions
    prop_batch_samples_array = tree_map(
        lambda x, p: _batch_slicer(x, 0, p),
        samples_array,
        prop_batch_sizes,
        is_leaf=lambda leaf: type(leaf) in sample_types,
    )

    # Concatenate the samples along the batch axis
    joint_sample = tree_map(
        lambda *x: jnp.concatenate(x, axis=0),
        *prop_batch_samples_array,
    )

    return joint_sample


def can_sample_mixer_fn(
    states: Sequence[StateTypes],
    can_sample_fns: Sequence[Callable[[StateTypes], Array]],
) -> Array:
    """Check if all buffers can sample.

    Args:
        states (Sequence[StateTypes]): list of buffer states
        can_sample_fns (Sequence[Callable[[StateTypes], Array]]): list of can_sample functions
            from each buffer

    Returns:
        bool: whether all buffers can sample
    """
    each_can_sample = jnp.asarray(
        tree_map(
            lambda state, can_sample: can_sample(state),
            states,
            can_sample_fns,
            is_leaf=lambda leaf: type(leaf) in state_types,
        )
    )
    return jnp.all(each_can_sample)


def make_mixer(
    buffers: Sequence[BufferTypes],
    sample_batch_size: int,
    proportions: Sequence[Numeric],
) -> Mixer:
    """Create the mixer.

    Args:
        buffers (Sequence[BufferTypes]): list of buffers (pure functions)
        sample_batch_size (int): desired batch size of joint sample
        proportions (Sequence[Numeric]):
            Proportions of joint sample size to be sampled from each buffer, given as a ratio.

    Returns:
        Mixer: a mixer
    """
    assert len(buffers) == len(
        proportions
    ), "Number of buffers and proportions must match"
    assert all(
        isinstance(b, type(buffers[0])) for b in buffers
    ), "All buffers must be of the same type"
    assert sample_batch_size > 0, "Sample batch size must be greater than 0"

    sample_fns = [b.sample for b in buffers]
    can_sample_fns = [b.can_sample for b in buffers]

    # Normalize proportions and calculate resulting integer batch sizes
    props_sum = sum(proportions)
    props_norm = [p / props_sum for p in proportions]
    prop_batch_sizes = [int(p * sample_batch_size) for p in props_norm]
    if sum(prop_batch_sizes) != sample_batch_size:
        # In case of rounding errors, add the remainder to the first buffer's proportion
        prop_batch_sizes[0] += sample_batch_size - sum(prop_batch_sizes)

    mixer_sample_fn = functools.partial(
        sample_mixer_fn,
        prop_batch_sizes=prop_batch_sizes,
        sample_fns=sample_fns,
    )

    mixer_can_sample_fn = functools.partial(
        can_sample_mixer_fn,
        can_sample_fns=can_sample_fns,
    )

    return Mixer(
        sample=mixer_sample_fn,
        can_sample=mixer_can_sample_fn,
    )
