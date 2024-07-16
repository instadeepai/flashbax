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
from typing import Callable

import jax.numpy as jnp
from chex import dataclass
from jax.tree_util import tree_map

from flashbax.buffers.trajectory_buffer import (
    TrajectoryBufferSample,
    TrajectoryBufferState,
)


@dataclass(frozen=True)
class Mixer:
    sample: Callable
    can_sample: Callable


def sample_mixer_fn(
    states,
    key,
    prop_batch_sizes,
    sample_fns,
):
    samples_array = tree_map(
        lambda state, sample, key_in: sample(state, key_in),
        states,
        sample_fns,
        [key] * len(sample_fns),  # if key.ndim == 1 else key,
        is_leaf=lambda leaf: type(leaf) == TrajectoryBufferState,
    )

    def _slicer(sample, batch_slice):
        return tree_map(lambda x: x[:batch_slice, ...], sample)

    prop_batch_samples_array = tree_map(
        lambda x, p: _slicer(x, p),
        samples_array,
        prop_batch_sizes,
        is_leaf=lambda leaf: type(leaf) == TrajectoryBufferSample,
    )

    joint_sample = tree_map(
        lambda *x: jnp.concatenate(x, axis=0),
        *prop_batch_samples_array,
    )
    return joint_sample


def can_sample_mixer_fn(
    states,
    can_sample_fns,
):
    each_can_sample = tree_map(
        lambda state, can_sample: can_sample(state),
        states,
        can_sample_fns,
        is_leaf=lambda leaf: type(leaf) == TrajectoryBufferState,
    )
    return all(each_can_sample)


def make_mixer(
    buffers: list,
    sample_batch_size: int,
    proportions: list,
):
    sample_fns = [b.sample for b in buffers]
    can_sample_fns = [b.can_sample for b in buffers]

    props_sum = sum(proportions)
    props_norm = [p / props_sum for p in proportions]
    prop_batch_sizes = [int(p * sample_batch_size) for p in props_norm]
    if sum(prop_batch_sizes) != sample_batch_size:
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
