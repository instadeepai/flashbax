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
from typing import TYPE_CHECKING, Callable, Dict, Generic, Optional, Tuple

import chex
from chex import PRNGKey
from typing_extensions import NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax
import jax.numpy as jnp

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


def validate_n_step(n_step: int, max_length: int):
    if n_step >= max_length:
        raise ValueError(
            f"""n_step must be less than max_length. It is currently
            {n_step} >= {max_length}"""
        )


def validate_n_step_buffer_args(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_batch_size: int,
    n_step: int,
):
    """Validates the arguments for the n-step buffer."""

    validate_sample_batch_size(sample_batch_size, max_length)
    validate_min_length(min_length, add_batch_size, max_length)
    validate_max_length_add_batch_size(max_length, add_batch_size)
    validate_n_step(n_step, max_length)


def create_n_step_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    n_step: int,
    add_sequences: bool,
    add_batch_size: Optional[int],
    n_step_functional_map: Optional[
        Dict[Tuple[str, ...], Callable[[chex.Array], chex.Array]]
    ] = None,
) -> TrajectoryBuffer:
    """Creates a trajectory buffer that acts as a n-step buffer.

    Args:
        max_length (int): The maximum length of the buffer.
        min_length (int): The minimum length of the buffer.
        sample_batch_size (int): The batch size of the samples.
        n_step (int): The number of steps to use for the n-step buffer.
        add_sequences (Optional[bool], optional): Whether data is being added in sequences
            to the buffer. If False, single transitions are being added each time add
            is called. Defaults to False.
        add_batch_size (Optional[int], optional): If adding data in batches, what is the
            batch size that is being added each time. If None, single transitions or single
            sequences are being added each time add is called. Defaults to None.
        n_step_functional_map (Optional[Dict[Tuple[str, ...], Callable[[chex.Array], chex.Array]]], optional):
            A dictionary mapping functions to n-step transitions. Keys are tuples of data attribute names;
            values are functions. Tuples format: first position is the data attribute being processed,
            remaining positions are arguments for the function. Functions process n+1 data items, returning
            n+1 items. Only the first and last values of the sequence are used. If key 'dict_mapper' exists,
            its value is a function processing the data batch and returning a dictionary. This function maps
            data types to dictionaries. For dictionaries and chex.dataclasses, setting this is optional.
            For flax structs and named tuples, it must be set. Only data attributes in the dictionary are
            modified; others remain unchanged.


    Returns:
        The buffer."""

    if add_batch_size is None:
        # add_batch_size being None implies that we are adding single transitions
        add_batch_size = 1
        add_batches = False
    else:
        add_batches = True

    validate_n_step_buffer_args(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        n_step=n_step,
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
            sample_sequence_length=n_step + 1,
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
        sampled_n_step_sequence_item = buffer.sample(state, rng_key).experience
        if n_step_functional_map is not None:
            sampled_n_step_sequence_item = perform_n_step_functional_map(
                n_step_functional_map, sampled_n_step_sequence_item
            )
        first = jax.tree_util.tree_map(lambda x: x[:, 0], sampled_n_step_sequence_item)
        second = jax.tree_util.tree_map(
            lambda x: x[:, -1], sampled_n_step_sequence_item
        )
        return TransitionSample(experience=ExperiencePair(first=first, second=second))

    return buffer.replace(add=add_fn, sample=sample_fn)  # type: ignore


def perform_n_step_functional_map(
    n_step_functional_map: Dict[Tuple[str, ...], Callable[[chex.Array], chex.Array]],
    sampled_n_step_sequence_item: Experience,
):
    item_type_dict_mapper = n_step_functional_map.pop("dict_mapper", dict)
    experience_type = type(sampled_n_step_sequence_item)
    sampled_n_step_sequence_dict = item_type_dict_mapper(sampled_n_step_sequence_item)
    mapped_data: Dict[tuple, Callable] = {}
    for key, fun in n_step_functional_map.items():
        data_key = key[0]
        data_func_args = key[1:]
        data_func_args = [sampled_n_step_sequence_dict[k] for k in data_func_args]
        mapped_data[data_key] = jax.tree_util.tree_map(fun, *data_func_args)

    for key, value in mapped_data.items():
        sampled_n_step_sequence_dict[key] = value

    sampled_n_step_sequence_item = experience_type(**sampled_n_step_sequence_dict)
    return sampled_n_step_sequence_item


def make_n_step_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    n_step: int = 1,
    add_sequences: bool = False,
    add_batch_size: Optional[int] = None,
    n_step_functional_map: Optional[
        Dict[Tuple[str, ...], Callable[[chex.Array], chex.Array]]
    ] = None,
) -> TrajectoryBuffer:
    """Makes a trajectory buffer act as a n-step buffer.

    Args:
        max_length (int): The maximum length of the buffer.
        min_length (int): The minimum length of the buffer.
        sample_batch_size (int): The batch size of the samples.
        n_step (int): The number of steps to use for the n-step buffer.
        add_sequences (Optional[bool], optional): Whether data is being added in sequences
            to the buffer. If False, single transitions are being added each time add
            is called. Defaults to False.
        add_batch_size (Optional[int], optional): If adding data in batches, what is the
            batch size that is being added each time. If None, single transitions or single
            sequences are being added each time add is called. Defaults to None.
        n_step_functional_map (Optional[Dict[Tuple[str, ...], Callable[[chex.Array], chex.Array]]], optional):
            A dictionary of functions to apply to the n-step transitions. The keys are tuples of data attribute
            names the values are the functions to apply. The tuples are expected in the format where
            the first position is the name of the data attribute being processed, the rest of the positions
            are data attributes that are used as arguments to the function that processes the sequence of data.
            Each function takes in a sequence of n+1 data items and returns a sequence of n+1 data items.
            However, only the first and last value of the sequence are used and placed into first and second
            respectively. If the dictionary has a key 'dict_mapper' then the value is a function
            that takes in the batch of data and returns a dictionary. This function is used to map the
            data type that to a dictionary. For dictionaries and chex.dataclasses this does not need
            to be set but for flax structs and named tuples this needs to be set accordingly. Only the
            data attributes that are in the dictionary are modified and all other data attributes are
            left unchanged.

    Returns:
        The buffer."""

    return create_n_step_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        n_step=n_step,
        add_sequences=add_sequences,
        add_batch_size=add_batch_size,
        n_step_functional_map=n_step_functional_map,
    )


def n_step_returns(
    r_t: chex.Array,
    discount_t: chex.Array,
    n_step: int,
) -> chex.Array:
    """Computes strided n-step returns over a sequence.

    Args:
        r_t: rewards at times [1, ..., T].
        discount_t: discounts at times [1, ..., T].
        n_step: number of steps over which to accumulate reward.

    Returns:
        n-step returns at times [0, ...., T-1]
    """
    chex.assert_rank([r_t, discount_t], [1, 1])
    chex.assert_type([r_t, discount_t], float)
    chex.assert_equal_shape([r_t, discount_t])
    seq_len = r_t.shape[0]
    targets = jnp.zeros(seq_len)

    # Pad sequences. Shape is now (T + n - 1,).
    r_t = jnp.concatenate([r_t, jnp.zeros(n_step - 1)])
    discount_t = jnp.concatenate([discount_t, jnp.ones(n_step - 1)])

    # Work backwards to compute n-step returns.
    for i in reversed(range(n_step)):
        r_ = r_t[i : i + seq_len]
        discount_ = discount_t[i : i + seq_len]
        targets = r_ + discount_ * targets

    return targets


def n_step_product(
    discount_t: chex.Array,
    n_step: int,
) -> chex.Array:
    """Computes strided n-step products over a sequence. Practically, this is
    useful for computing whether or not one encountered a terminal state in the
    n-step sequence.

    Args:
        discount_t: discounts at times [1, ..., T].
        n_step: number of steps over which to multiply discounts."""

    chex.assert_rank([discount_t], [1])
    chex.assert_type([discount_t], float)
    discount_t = discount_t[:-1]
    seq_len = discount_t.shape[0]
    targets = jnp.ones(seq_len)

    # Pad sequences. Shape is now (T + n - 1,).
    discount_t = jnp.concatenate([discount_t, jnp.ones(n_step - 1)])

    # Work backwards to compute n-step products.
    for i in reversed(range(n_step)):
        discount_ = discount_t[i : i + seq_len]
        targets = discount_ * targets

    return targets
