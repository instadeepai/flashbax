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

import asyncio
import json
import os
from ast import literal_eval as make_tuple
from datetime import datetime
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tensorstore as ts  # type: ignore
from chex import Array
from etils import epath  # type: ignore
from jax.tree_util import DictKey, GetAttrKey
from orbax.checkpoint.utils import deserialize_tree, serialize_tree  # type: ignore

from flashbax.buffers.trajectory_buffer import Experience, TrajectoryBufferState
from flashbax.utils import get_tree_shape_prefix

# Constants
DRIVER = "file://"
METADATA_FILE = "metadata.json"
TIME_AXIS_MAX_LENGTH = int(10e12)  # Upper bound on the length of the time axis
COMPRESSION_DEFAULT = {
    "id": "gzip",
    "level": 5,
}
VERSION = 1.2


def _path_to_ds_name(path: Tuple[Union[DictKey, GetAttrKey], ...]) -> str:
    """Utility function to convert a path (yielded by jax.tree_util.tree_map_with_path)
    to a datastore name. The alternative is to use jax.tree_util.keystr(...), but this has
    different behaviour for dictionaries (DictKey) vs. namedtuples (GetAttrKey), which means
    we could not save a vault based on a namedtuple structure but later load it as a dict.
    Instead, this function maps both to a consistent string representation.

    Args:
        path: tuple of DictKeys or GetAttrKeys

    Returns:
        str: standardised string representation of the path
    """
    path_str = ""
    for p in path:
        if isinstance(p, DictKey):
            path_str += str(p.key)
        elif isinstance(p, GetAttrKey):
            path_str += p.name
        path_str += "."
    return path_str


class Vault:
    def __init__(  # noqa: CCR001
        self,
        vault_name: str,
        experience_structure: Optional[Experience] = None,
        rel_dir: str = "vaults",
        vault_uid: Optional[str] = None,
        compression: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Flashbax utility for storing buffers to disk efficiently.

        Args:
            vault_name (str): the upper-level name of this vault.
                Resulting path is <cwd/rel_dir/vault_name/vault_uid>.
            experience_structure (Optional[Experience], optional):
                Structure of the experience data, usually given as `buffer_state.experience`.
                Defaults to None, which can only be done if reading an existing vault.
            rel_dir (str, optional):
                Base directory of all vaults. Defaults to "vaults".
            vault_uid (Optional[str], optional): Unique identifier for this vault.
                Defaults to None, which will use the current timestamp.
            compression (Optional[dict], optional):
                Compression settings used when when creating the vault.
                Defaults to None, which will use the default compression.
            metadata (Optional[dict], optional):
                Any additional metadata to save. Defaults to None.

        Raises:
            ValueError:
                If the targeted vault does not exist, and no experience_structure is provided.

        Returns:
            Vault: a vault object.
        """
        # Get the base path for the vault and the metadata path
        vault_str = vault_uid if vault_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        self._base_path = os.path.join(os.getcwd(), rel_dir, vault_name, vault_str)
        metadata_path = epath.Path(os.path.join(self._base_path, METADATA_FILE))

        # Check if the vault exists, otherwise create the necessary dirs and files
        base_path_exists = os.path.exists(self._base_path)
        if base_path_exists:
            # Vault exists, so we load the metadata to access the structure etc.
            self._metadata = json.loads(metadata_path.read_text())

            # Ensure minor versions match
            assert (self._metadata["version"] // 1) == (VERSION // 1)

            print(f"Loading vault found at {self._base_path}")

            if compression is not None:
                print(
                    "Requested compression settings will be ignored as the vault already exists."
                )

        elif experience_structure is not None:
            # Create the necessary dirs for the vault
            os.makedirs(self._base_path)

            # Ensure provided metadata is json serialisable
            metadata_json_ready = jax.tree_util.tree_map(
                lambda obj: str(obj)
                if not isinstance(obj, (bool, str, int, float, type(None)))
                else obj,
                metadata,
            )

            # We save the structure of the buffer state, storing the shape and dtype of
            # each leaf. We will use this structure to map over the data stores later.
            # (Note: we use `jax.eval_shape` to get shape and dtype of each leaf, without
            # unnecessarily serialising the buffer data itself)
            serialised_experience_structure_shape = jax.tree_map(
                lambda x: str(x.shape),
                serialize_tree(jax.eval_shape(lambda: experience_structure)),
            )
            serialised_experience_structure_dtype = jax.tree_map(
                lambda x: x.dtype.name,
                serialize_tree(jax.eval_shape(lambda: experience_structure)),
            )

            # Construct metadata
            self._metadata = {
                "version": VERSION,
                "structure_shape": serialised_experience_structure_shape,
                "structure_dtype": serialised_experience_structure_dtype,
                **(metadata_json_ready or {}),  # Allow user to save extra metadata
            }
            # Dump metadata to file
            metadata_path.write_text(json.dumps(self._metadata))

            print(f"New vault created at {self._base_path}")

            _fbx_shape = jax.tree_util.tree_leaves(experience_structure)[0].shape
            print(
                f"Since the provided buffer state has a temporal dimension of {_fbx_shape[1]}, "
                f"you must write to the vault at least every {_fbx_shape[1] - 1} "
                "timesteps to avoid data loss."
            )
        else:
            # If the vault does not exist already, and no experience_structure is provided to create
            # a new vault, we cannot proceed.
            raise ValueError(
                "Vault does not exist and no experience_structure was provided."
            )

        # We must now build the tree structure from the metadata, whether the metadata was created
        # here or loaded from file
        if experience_structure is None:
            # Since the experience structure is not provided, we simply use the metadata as is.
            # The result will always be a dictionary.
            self._tree_structure_shape = self._metadata["structure_shape"]
            self._tree_structure_dtype = self._metadata["structure_dtype"]
        else:
            # If experience structure is provided, we try deserialise into that structure
            self._tree_structure_shape = deserialize_tree(
                self._metadata["structure_shape"],
                target=experience_structure,
            )
            self._tree_structure_dtype = deserialize_tree(
                self._metadata["structure_dtype"],
                target=experience_structure,
            )

        # Keep the compression settings, to be used in init_leaf, in case we're creating the vault
        self._compression = compression

        # Each leaf of the fbx_state.experience maps to a data store, so we tree map over the
        # tree structure to create each of the data stores.
        self._all_datastores = jax.tree_util.tree_map_with_path(
            lambda path, shape, dtype: self._init_leaf(
                name=_path_to_ds_name(path),
                shape=make_tuple(
                    shape
                ),  # Must convert to a real tuple from the saved str
                dtype=dtype,
                create_ds=not base_path_exists,
            ),
            self._tree_structure_shape,
            self._tree_structure_dtype,
        )

        # We keep track of the last fbx buffer idx received
        self._last_received_fbx_index = 0

        # We store and load the vault index from a separate datastore
        self._vault_index_ds = ts.open(
            self._get_base_spec("vault_index"),
            dtype=jnp.int32,
            shape=(1,),
            create=not base_path_exists,
        ).result()
        # Just read synchronously as it's one number
        self.vault_index = int(self._vault_index_ds.read().result()[0])

    def _get_base_spec(self, name: str) -> dict:
        """Simple common specs for all datastores.

        Args:
            name (str): name of the datastore

        Returns:
            dict: config for the datastore
        """
        return {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": f"{DRIVER}{self._base_path}",
                "path": name,
            },
        }

    def _init_leaf(
        self, name: str, shape: Tuple[int, ...], dtype: str, create_ds: bool = False
    ) -> ts.TensorStore:
        """Initialise a datastore for a leaf of the experience tree.

        Args:
            name (str): datastore name
            shape (Tuple[int, ...]): shape of the data for this leaf
            dtype (str): dtype of the data for this leaf
            create_ds (bool, optional): whether to create the datastore. Defaults to False.

        Returns:
            ts.TensorStore: the datastore object
        """
        spec = self._get_base_spec(name)

        leaf_shape, leaf_dtype = None, None
        if create_ds:
            # Only specify dtype, shape, and compression if we are creating a vault
            # (i.e. don't impose these fields if we are _loading_ a vault)
            leaf_shape = (
                shape[0],  # Batch dim
                TIME_AXIS_MAX_LENGTH,  # Time dim, which we extend
                *shape[2:],  # Experience dim(s)
            )
            leaf_dtype = dtype
            spec["metadata"] = {
                "compressor": COMPRESSION_DEFAULT
                if self._compression is None
                else self._compression
            }

        leaf_ds = ts.open(
            spec,
            shape=leaf_shape,
            dtype=leaf_dtype,
            # Only create datastore if we are creating the vault:
            create=create_ds,
        ).result()  # Do this synchronously

        return leaf_ds

    async def _write_leaf(
        self,
        source_leaf: jax.Array,
        dest_leaf: ts.TensorStore,
        source_interval: Tuple[int, int],
        dest_start: int,
    ) -> None:
        """Asychronously write a chunk of data to a leaf's datastore.

        Args:
            source_leaf (jax.Array): the input fbx_state.experience array
            dest_leaf (ts.TensorStore): the destination datastore
            source_interval (Tuple[int, int]): read interval from the source leaf
            dest_start (int): write start index in the destination leaf
        """
        dest_interval = (
            dest_start,
            dest_start + (source_interval[1] - source_interval[0]),
        )
        # Write to the datastore along the time axis
        await dest_leaf[:, slice(*dest_interval), ...].write(
            source_leaf[:, slice(*source_interval), ...],
        )

    async def _write_chunk(
        self,
        fbx_state: TrajectoryBufferState,
        source_interval: Tuple[int, int],
        dest_start: int,
    ) -> None:
        """Asynchronous method for writing to all the datastores.

        Args:
            fbx_state (TrajectoryBufferState): input buffer state
            source_interval (Tuple[int, int]): read interval from the buffer state
            dest_start (int): write start index in the vault
        """
        # Collect futures for writing to each datastore
        futures_tree = jax.tree_util.tree_map(
            lambda x, ds: self._write_leaf(
                source_leaf=x,
                dest_leaf=ds,
                source_interval=source_interval,
                dest_start=dest_start,
            ),
            fbx_state.experience,  # x = experience
            self._all_datastores,  # ds = data stores
        )
        # Write to all datastores asynchronously
        futures, _ = jax.tree_util.tree_flatten(futures_tree)
        await asyncio.gather(*futures)

    def write(
        self,
        fbx_state: TrajectoryBufferState,
        source_interval: Tuple[int, int] = (0, 0),
        dest_start: Optional[int] = None,
    ) -> int:
        """Write any new data from the fbx buffer state to the vault.

        Args:
            fbx_state (TrajectoryBufferState): input buffer state
            source_interval (Tuple[int, int], optional): from where to read in the buffer.
                Defaults to (0, 0), which reads from the last received index up to the
                current buffer state's index.
            dest_start (Optional[int], optional): where to write in the vault.
                Defaults to None, which writes from the current vault index.

        Returns:
            int: how many elements along the time-axis were written to the vault
        """
        fbx_current_index = int(fbx_state.current_index)

        # By default, we read from `last received` to `current index`
        if source_interval == (0, 0):
            source_interval = (self._last_received_fbx_index, fbx_current_index)

        # By default, we continue writing from the current vault index
        dest_start = self.vault_index if dest_start is None else dest_start

        if source_interval[1] == source_interval[0]:
            # Nothing to write
            return 0

        elif source_interval[1] > source_interval[0]:
            # Vanilla write, no wrap around in the buffer state
            asyncio.run(
                self._write_chunk(
                    fbx_state=fbx_state,
                    source_interval=source_interval,
                    dest_start=dest_start,
                )
            )
            written_length = source_interval[1] - source_interval[0]

        elif source_interval[1] < source_interval[0]:
            # Wrap around in the buffer state!

            # Get seq dim (i.e. the length of the time axis in the fbx buffer state)
            fbx_max_index = get_tree_shape_prefix(fbx_state.experience, n_axes=2)[1]

            # Read from last received fbx index to max index
            source_interval_a = (source_interval[0], fbx_max_index)
            time_length_a = source_interval_a[1] - source_interval_a[0]

            asyncio.run(
                self._write_chunk(
                    fbx_state=fbx_state,
                    source_interval=source_interval_a,
                    dest_start=dest_start,
                )
            )

            # Read from the start of the fbx buffer state to the current fbx index
            source_interval_b = (0, source_interval[1])
            time_length_b = source_interval_b[1] - source_interval_b[0]

            asyncio.run(
                self._write_chunk(
                    fbx_state=fbx_state,
                    source_interval=source_interval_b,
                    dest_start=dest_start + time_length_a,
                )
            )

            written_length = time_length_a + time_length_b

        # Update vault index, and write this to its datastore too
        self.vault_index += written_length
        self._vault_index_ds.write(self.vault_index).result()

        # Keep track of the last fbx buffer idx received
        self._last_received_fbx_index = fbx_current_index

        return written_length

    def _read_leaf(
        self,
        read_leaf: ts.TensorStore,
        read_interval: Tuple[int, int],
    ) -> Array:
        """Read from a leaf of the experience tree.

        Args:
            read_leaf (ts.TensorStore): the datastore from which to read
            read_interval (Tuple[int, int]): the interval on the time-axis to read

        Returns:
            Array: the read data, as a jax array
        """
        return jnp.asarray(read_leaf[:, slice(*read_interval), ...].read().result())

    def read(
        self,
        timesteps: Optional[int] = None,
        percentiles: Optional[Tuple[int, int]] = None,
    ) -> TrajectoryBufferState:
        """Read synchronously from the vault.

        Args:
            timesteps (Optional[int], optional):
                If provided, we read the last `timesteps` count of elements.
                Defaults to None.
            percentiles (Optional[Tuple[int, int]], optional):
                If provided (and timesteps is None) we read the corresponding interval.
                Defaults to None.

        Returns:
            TrajectoryBufferState: the read data as a fbx buffer state
        """

        # By default we read the entire vault
        if timesteps is None and percentiles is None:
            read_interval = (0, self.vault_index)
        # If time steps are provided, we read the last `timesteps` count of elements
        elif timesteps is not None:
            read_interval = (self.vault_index - timesteps, self.vault_index)
        # If percentiles are provided, we read the corresponding interval
        elif percentiles is not None:
            assert (
                percentiles[0] < percentiles[1]
            ), "Percentiles must be in ascending order."
            read_interval = (
                int(self.vault_index * (percentiles[0] / 100)),
                int(self.vault_index * (percentiles[1] / 100)),
            )

        read_result = jax.tree_util.tree_map(
            lambda _, ds: self._read_leaf(
                read_leaf=ds,
                read_interval=read_interval,
            ),
            self._tree_structure_shape,  # Just used to return a valid tree structure
            self._all_datastores,  # The vault data stores
        )

        # Return the read result as a fbx buffer state
        return TrajectoryBufferState(
            experience=read_result,
            current_index=jnp.array(0, dtype=int),
            is_full=jnp.array(True, dtype=bool),
        )
