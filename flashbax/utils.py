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
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp


def get_tree_shape_prefix(tree: chex.ArrayTree, n_axes: int = 1) -> chex.Shape:
    """Get the shape of the leading axes (up to n_axes) of a pytree. This assumes all
    leaves have a common leading axes size (e.g. a common batch size)."""
    flat_tree, tree_def = jax.tree_util.tree_flatten(tree)
    leaf = flat_tree[0]
    leading_axis_shape = leaf.shape[0:n_axes]
    chex.assert_tree_shape_prefix(tree, leading_axis_shape)
    return leading_axis_shape


def add_dim_to_args(
    func: Callable,
    axis: int = 1,
    starting_arg_index: Optional[int] = 1,
    ending_arg_index: Optional[int] = None,
    kwargs_on_device_keys: Optional[list] = None,
):
    """Adds a dimension to the specified arguments of a function.

    Args:
        func (Callable): The function to wrap.
        axis (int, optional): The axis to add the dimension to. Defaults to 1.
        starting_arg_index (Optional[int], optional): The index of the first argument to
            add the dimension to. Defaults to 1.
        ending_arg_index (Optional[int], optional): The index of the last argument to
            add the dimension to. Defaults to None.
        kwargs_on_device_keys (Optional[list], optional): The keys of the kwargs that should
            be added to. Defaults to None.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ending_arg_index is None:
            end_index = len(args)
        else:
            end_index = ending_arg_index

        args = list(args)
        args[starting_arg_index:end_index] = [
            jax.tree.map(lambda x: jnp.expand_dims(x, axis=axis), a)
            for a in args[starting_arg_index:end_index]
        ]
        for k, v in kwargs.items():
            if kwargs_on_device_keys is None or k in kwargs_on_device_keys:
                kwargs[k] = jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), v)
        return func(*args, **kwargs)

    return wrapper


def get_timestep_count(buffer_state: chex.ArrayTree) -> int:
    """Utility to compute the total number of timesteps currently in the buffer state.

    Args:
        buffer_state (BufferStateTypes): the buffer state to compute the total timesteps for.

    Returns:
        int: the total number of timesteps in the buffer state.
    """
    # Ensure the buffer state is a valid buffer state.
    assert hasattr(buffer_state, "experience")
    assert hasattr(buffer_state, "current_index")
    assert hasattr(buffer_state, "is_full")

    b_size, t_size_max = get_tree_shape_prefix(buffer_state.experience, 2)
    t_size = jax.lax.cond(
        buffer_state.is_full,
        lambda: t_size_max,
        lambda: buffer_state.current_index,
    )
    timestep_count: int = b_size * t_size
    return timestep_count


def get_max_index(buffer_state: chex.ArrayTree) -> int:
    """Utility to compute the maximum index of the buffer state.

    Args:
        buffer_state: the buffer state to compute the maximum index for.

    Returns:
        int: the maximum index of the buffer state.
    """
    assert hasattr(buffer_state, "experience")
    return get_tree_shape_prefix(buffer_state.experience, n_axes=2)[1]
