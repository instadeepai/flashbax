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

import warnings
from typing import TYPE_CHECKING, Generic, Optional

from chex import PRNGKey
from typing_extensions import NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax

from flashbax.buffers.trajectory_buffer import (
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferState,
    make_trajectory_buffer,
)
from flashbax.utils import add_dim_to_args


class ExperiencePair(NamedTuple, Generic[Experience]):
    first: Experience
    second: Experience


@dataclass(frozen=True)
class TransitionSample(Generic[Experience]):
    experience: ExperiencePair[Experience]


def validate_sample_batch_size(sample_batch_size: int, max_length: int):
    if sample_batch_size > max_length:
        raise ValueError("sample_batch_size must be less than or equal to max_length")


def validate_min_length(min_length: int, add_batch_size: int, max_length: int):
    used_min_length = min_length // add_batch_size + 1
    if used_min_length > max_length:
        raise ValueError("min_length used is too large for the buffer size.")


def validate_max_length_add_batch_size(max_length: int, add_batch_size: int):
    if max_length // add_batch_size < 2:
        raise ValueError(
            f"""max_length//add_batch_size must be greater than 2. It is currently
            {max_length}//{add_batch_size} = {max_length//add_batch_size}"""
        )


def validate_flat_buffer_args(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_batch_size: int,
):
    """Validates the arguments for the flat buffer."""

    validate_sample_batch_size(sample_batch_size, max_length)
    validate_min_length(min_length, add_batch_size, max_length)
    validate_max_length_add_batch_size(max_length, add_batch_size)


def create_flat_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool,
    add_batch_size: Optional[int],
) -> TrajectoryBuffer:
    """Creates a trajectory buffer that acts as a flat buffer.

    Args:
        max_length (int): The maximum length of the buffer.
        min_length (int): The minimum length of the buffer.
        sample_batch_size (int): The batch size of the samples.
        add_sequences (Optional[bool], optional): Whether data is being added in sequences
            to the buffer. If False, single transitions are being added each time add
            is called. Defaults to False.
        add_batch_size (Optional[int], optional): If adding data in batches, what is the
            batch size that is being added each time. If None, single transitions or single
            sequences are being added each time add is called. Defaults to None.

    Returns:
        The buffer."""

    if add_batch_size is None:
        # add_batch_size being None implies that we are adding single transitions
        add_batch_size = 1
        add_batches = False
    else:
        add_batches = True

    validate_flat_buffer_args(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting max_size dynamically sets the `max_length_time_axis` to "
            f"be `max_size`//`add_batch_size = {max_length // add_batch_size}`."
            "This allows one to control exactly how many transitions are stored in the buffer."
            "Note that this overrides the `max_length_time_axis` argument.",
        )

        buffer = make_trajectory_buffer(
            max_length_time_axis=None,  # Unused because max_size is specified
            min_length_time_axis=min_length // add_batch_size + 1,
            add_batch_size=add_batch_size,
            sample_batch_size=sample_batch_size,
            sample_sequence_length=2,
            period=1,
            max_size=max_length,
        )

    add_fn = buffer.add

    if not add_batches:
        add_fn = add_dim_to_args(
            add_fn, axis=0, starting_arg_index=1, ending_arg_index=2
        )

    if not add_sequences:
        axis = 1 - int(not add_batches)  # 1 if add_batches else 0
        add_fn = add_dim_to_args(
            add_fn, axis=axis, starting_arg_index=1, ending_arg_index=2
        )

    def sample_fn(state: TrajectoryBufferState, rng_key: PRNGKey) -> TransitionSample:
        """Samples a batch of transitions from the buffer."""
        sampled_batch = buffer.sample(state, rng_key).experience
        first = jax.tree_util.tree_map(lambda x: x[:, 0], sampled_batch)
        second = jax.tree_util.tree_map(lambda x: x[:, 1], sampled_batch)
        return TransitionSample(experience=ExperiencePair(first=first, second=second))

    return buffer.replace(add=add_fn, sample=sample_fn)  # type: ignore


def make_flat_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool = False,
    add_batch_size: Optional[int] = None,
) -> TrajectoryBuffer:
    """Makes a trajectory buffer act as a flat buffer.

    Args:
        max_length (int): The maximum length of the buffer.
        min_length (int): The minimum length of the buffer.
        sample_batch_size (int): The batch size of the samples.
        add_sequences (Optional[bool], optional): Whether data is being added in sequences
            to the buffer. If False, single transitions are being added each time add
            is called. Defaults to False.
        add_batch_size (Optional[int], optional): If adding data in batches, what is the
            batch size that is being added each time. If None, single transitions or single
            sequences are being added each time add is called. Defaults to None.

    Returns:
        The buffer."""

    return create_flat_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences=add_sequences,
        add_batch_size=add_batch_size,
    )
