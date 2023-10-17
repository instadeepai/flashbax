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
import warnings
from typing import TYPE_CHECKING, Callable, Generic, Optional, Tuple, TypeVar

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from jax import Array

from flashbax import utils

Experience = TypeVar("Experience", bound=chex.ArrayTree)


@dataclass(frozen=True)
class TrajectoryQueueState(Generic[Experience]):
    """State of the trajectory queue.

    Attributes:
        experience: Arbitrary pytree containing the experience data, for example a single
            transition (s,a,r,s'). These are stacked along the first axis.
        write_index: Index where the next batch of experience data will be added to.
        read_index: Index where the next batch of experience data will be sampled from.
        is_full: Whether the queue state is completely full with experience (otherwise it will
            have some empty padded values).
    """

    experience: Experience
    write_index: Array
    read_index: Array
    is_full: Array


@dataclass(frozen=True)
class TrajectoryQueueSample(Generic[Experience]):
    """Container for samples from the queue

    Attributes:
        experience: Arbitrary pytree containing a batch of experience data.
    """

    experience: Experience


def init(
    experience: Experience, add_batch_size: int, max_length_time_axis: int
) -> TrajectoryQueueState[Experience]:
    """
    Initialise the queue state.

    Args:
        experience: A single timestep (e.g. (s,a,r)) used for inferring
            the structure of the experience data that will be saved in the queue state.
        add_batch_size: Batch size of experience added to the queue's state using the `add`
            function. I.e. the leading batch size of added experience should have size
            `add_batch_size`.
        max_length_time_axis: Maximum length of the queue along the time axis (second axis of the
            experience data).

    Returns:
        state: Initial state of the replay queue. All values are empty as no experience has
            been added yet.
    """
    # Set experience value to be empty.
    experience = jax.tree_map(jnp.empty_like, experience)

    # Broadcast to [add_batch_size, max_length_time_axis]
    experience = jax.tree_map(
        lambda x: jnp.broadcast_to(
            x[None, None, ...], (add_batch_size, max_length_time_axis, *x.shape)
        ),
        experience,
    )

    state = TrajectoryQueueState(
        experience=experience,
        is_full=jnp.array(False, dtype=bool),
        read_index=jnp.array(0),
        write_index=jnp.array(0),
    )
    return state


def can_add(
    state: TrajectoryQueueState[Experience],
    add_sequence_length: int,
    max_length_time_axis: int,
) -> Array:
    """Check if the queue state can be written to."""

    # We check if the writing index will overtake the reading index in two cases.
    # First if the reading index is less than the writing index, then it is always possible
    # to write unless we are adding beyond the max length.
    # If we are adding beyond the max, then we have to check that the wrapped around index
    # is less than the reading index.
    # Second, if the reading index is greater than the writing index, then we check if the
    # new writing index is greater than the reading index.
    # In this case, we do not modulo by the time length.
    # If either of these cases are true, then we will overwrite the reading index which
    # means we cannot write to the buffer.
    new_write_index = state.write_index + add_sequence_length
    read_index_lt_eq_to_write = state.read_index <= state.write_index
    read_index_greater_than_write = state.read_index > state.write_index
    max_length_reached = new_write_index >= max_length_time_axis

    write_index_circular_overtake = (
        read_index_lt_eq_to_write
        & max_length_reached
        & ((new_write_index % max_length_time_axis) > state.read_index)
    )

    read_index_overtaken = read_index_greater_than_write & (
        new_write_index > state.read_index
    )

    will_overwrite = write_index_circular_overtake | read_index_overtaken

    do_not_write = state.is_full | will_overwrite

    return ~do_not_write


def add(
    state: TrajectoryQueueState[Experience],
    batch: Experience,
) -> TrajectoryQueueState[Experience]:
    """
    Add a batch of experience to the queue state. Assumes that this carries on from the episode
    where the previous added batch of experience ended. For example, if we consider a single
    trajectory within the batch; if the last timestep of the previous added trajectory's was at
    time `t` then the first timestep of the current trajectory will be at time `t + 1`.

    Args:
        state: The queue state.
        batch: A batch of experience. The leading axis of the pytree is the batch dimension.
            This must match `add_batch_size` and the structure of the experience used
            during initialisation of the queue state. This batch is added along the time axis of
            the queue state.


    Returns:
        A new queue state with the batch of experience added.
    """
    chex.assert_tree_shape_prefix(batch, utils.get_tree_shape_prefix(state.experience))
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    num_steps = utils.get_tree_shape_prefix(batch, n_axes=2)[1]
    max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)[1]

    chex.assert_axis_dimension_lteq(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )

    # Calculate index location in the state where we will assign the batch of experience.
    indices = (jnp.arange(num_steps) + state.write_index) % max_length_time_axis

    experience = jax.tree_util.tree_map(
        lambda experience_field, batch_field: experience_field.at[:, indices].set(
            batch_field
        ),
        state.experience,
        batch,
    )

    # update the write index.
    new_index = (state.write_index + num_steps) % max_length_time_axis

    is_full = new_index == state.read_index

    state = state.replace(  # type: ignore
        experience=experience,
        write_index=new_index,
        is_full=is_full,
    )

    return state


def sample(
    state: TrajectoryQueueState[Experience],
    sequence_length: int,
) -> Tuple[TrajectoryQueueState, TrajectoryQueueSample[Experience]]:
    """
    Sample a batch of trajectories from the queue.

    Args:
        state: The queue's state.
        sequence_length: Length of trajectory to sample.

    Returns:
        A batch of experience.
    """
    _, max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)

    # The queue is circular, so we can loop back to the start (`% max_length_time_axis`)
    # if the time index is greater than the length. We then add the sequence length to get
    # the end index of the sequence.
    time_indices = (
        jnp.arange(sequence_length) + state.read_index
    ) % max_length_time_axis

    # Slice the experience in the queue to get a single trajectory of length sequence_length
    trajectory = jax.tree_util.tree_map(lambda x: x[:, time_indices], state.experience)

    # Update the queue state.
    new_read_index = (state.read_index + sequence_length) % max_length_time_axis

    state = state.replace(  # type: ignore
        read_index=new_read_index,
        is_full=jnp.array(False),
    )
    sample = TrajectoryQueueSample(experience=trajectory)

    return state, sample


def can_sample(
    state: TrajectoryQueueState[Experience],
    sample_sequence_length: int,
    max_length_time_axis: int,
) -> Array:
    """Indicates whether the queue has been filled above the minimum length, such that it
    may be sampled from."""

    # Calculate all the conditional expressions
    new_read_index = state.read_index + sample_sequence_length
    read_index_less_than_write = state.read_index < state.write_index
    read_index_greater_than_write = state.read_index > state.write_index
    max_length_reached = new_read_index >= max_length_time_axis

    # If the read index is less than the write index and the new read index is still less than the
    # write index, then we can sample.
    can_sample_read_less = read_index_less_than_write & (
        new_read_index <= state.write_index
    )
    # If the read index is greater than the write index and the new read index has
    # wrapped around thus when taking modulo the length of the buffer, it is less
    # than the write index, then we can sample.
    can_sample_read_greater = (
        read_index_greater_than_write
        & max_length_reached
        & ((new_read_index % max_length_time_axis) <= state.write_index)
    )

    # if the queue is full
    can_sample_if_full = (
        state.is_full & ~max_length_reached & (new_read_index > state.write_index)
    )
    can_sample_if_full_circular = (
        state.is_full
        & max_length_reached
        & ((new_read_index % max_length_time_axis) <= state.write_index)
    )

    # We combine the previous conditions to get the final condition. Note that we
    # apply can_sample_read_greater only if the max length has been reached.
    # Otherwise, if the read index is greater than the write index and the
    # new read index is still greater than the write index without wrapping
    # around the buffer, then we can sample.
    can_sample = (
        can_sample_read_less
        | can_sample_read_greater
        | (read_index_greater_than_write & ~max_length_reached)
        | can_sample_if_full
        | can_sample_if_full_circular
    )

    return can_sample


@dataclass(frozen=True)
class TrajectoryQueue(Generic[Experience]):
    """Pure functions defining the trajectory queue. This queue assumes batches added to the
    queue are a pytree with a shape prefix of (batch_size, trajectory_length). Consecutive batches
    are then concatenated along the second axis (i.e. the time axis). During sampling this allows
    for trajectories to be sampled - by slicing consecutive sequences along the time axis.

    Attributes:
        init: A pure function which may be used to initialise the queue state using a single
            timestep (e.g. (s,a,r)).
        add: A pure function for adding a new batch of experience to the queue state.
        sample: A pure function for sampling a batch of data from the replay queue, with a leading
            axis of size (`sample_batch_size`, `sample_sequence_length`). Note `sample_batch_size`
            is the same as the batch size when adding data to the queue. However,
            `sample_sequence_length` may be different to the sequence length of data added to
            the state using the `add` function.
        can_sample: Whether the queue can be sampled from.
        can_add: Whether the queue can be added to.

    See `make_trajectory_queue` for how this container is instantiated.
    """

    init: Callable[[Experience], TrajectoryQueueState[Experience]]
    add: Callable[
        [TrajectoryQueueState[Experience], Experience],
        TrajectoryQueueState[Experience],
    ]
    sample: Callable[
        [TrajectoryQueueState[Experience]],
        Tuple[TrajectoryQueueState, TrajectoryQueueSample[Experience]],
    ]
    can_sample: Callable[[TrajectoryQueueState[Experience]], Array]
    can_add: Callable[[TrajectoryQueueState[Experience]], Array]


def validate_trajectory_queue_args(
    max_length_time_axis: int,
    add_batch_size: int,
    sample_sequence_length: int,
    max_size: Optional[int],
) -> None:
    if max_size is not None:
        warnings.warn(
            "Setting max_size dynamically sets the `max_length_time_axis` to "
            f"be `max_size`//`add_batch_size = {max_size // add_batch_size}`. "
            "This allows one to control exactly how many timesteps/transitions "
            "are stored in the queue. Note that this overrides the `max_length_time_axis` "
            "argument.",
            stacklevel=1,
        )
        max_length_time_axis = max_size // add_batch_size
    if sample_sequence_length > max_length_time_axis:
        raise ValueError(
            "`sample_sequence_length` must be less than or equal to `max_length_time_axis`."
        )


def make_trajectory_queue(
    max_length_time_axis: int,
    add_batch_size: int,
    add_sequence_length: int,
    sample_sequence_length: int,
    max_size: Optional[int] = None,
) -> TrajectoryQueue:
    """
    Args:
        max_length_time_axis: Maximum length of the queue in terms of time steps within the
            'time axis'. The second axis (the time axis) of the queue state's experience field will
            be of size `max_length_time_axis`.
        add_batch_size: Batch size of experience added to the queue. Used to initialise the leading
            axis of the queue state's experience.
        add_sequence_length: Trajectory length of experience that will be added to the queue
            on every add.
        sample_batch_size: Batch size of experience returned from the `sample` method of the
            queue.
        sample_sequence_length: Trajectory length of experience of sampled batches. Note that this
            may differ from the trajectory length of experience added to the queue.
        max_size: Optional argument to specify the size of the queue based on timesteps.
            This sets the maximum number of timesteps that can be stored in the queue and sets
            the `max_length_time_axis` to be `max_size`//`add_batch_size`. This allows one to
            control exactly how many timesteps/transitions are stored in the queue. Note that this
            overrides the `max_length_time_axis` argument.


    Returns:
        A trajectory queue.
    """

    validate_trajectory_queue_args(
        max_length_time_axis=max_length_time_axis,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        max_size=max_size,
    )

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    init_fn = functools.partial(
        init,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
    )
    add_fn = functools.partial(
        add,
    )
    sample_fn = functools.partial(
        sample,
        sequence_length=sample_sequence_length,
    )
    can_sample_fn = functools.partial(
        can_sample,
        sample_sequence_length=sample_sequence_length,
        max_length_time_axis=max_length_time_axis,
    )
    can_add_fn = functools.partial(
        can_add,
        add_sequence_length=add_sequence_length,
        max_length_time_axis=max_length_time_axis,
    )

    return TrajectoryQueue(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
        can_add=can_add_fn,
    )
