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
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import tensorstore as ts  # type: ignore
from chex import Array
from etils import epath  # type: ignore
from orbax.checkpoint.utils import deserialize_tree, serialize_tree

from flashbax.buffers.trajectory_buffer import Experience, TrajectoryBufferState
from flashbax.utils import get_tree_shape_prefix

# CURRENT LIMITATIONS / TODO LIST
# - Only tested with flat buffers

DRIVER = "file://"
METADATA_FILE = "metadata.json"
TIME_AXIS_MAX_LENGTH = int(10e12)  # Upper bound on the length of the time axis
VERSION = 0.1


def _path_to_ds_name(path: str) -> str:
    path_str = ""
    for p in path:
        if isinstance(p, jax.tree_util.DictKey):
            path_str += str(p.key)
        elif isinstance(p, jax.tree_util.GetAttrKey):
            path_str += p.name
        path_str += "."
    return path_str


class Vault:
    def __init__(
        self,
        vault_name: str,
        experience_structure: Optional[Experience] = None,
        rel_dir: str = "vaults",
        vault_uid: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:

        ## ---
        # Get the base path for the vault and the metadata path
        vault_str = vault_uid if vault_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        self._base_path = os.path.join(os.getcwd(), rel_dir, vault_name, vault_str)
        metadata_path = epath.Path(os.path.join(self._base_path, METADATA_FILE))

        # TODO: logging at each step
        # Check if the vault exists, otherwise create the necessary dirs and files
        base_path_exists = os.path.exists(self._base_path)
        if base_path_exists:
            # Vault exists, so we load the metadata to access the structure etc.
            self._metadata = json.loads(metadata_path.read_text())

            # Ensure minor versions match
            assert (self._metadata["version"] // 1) == (VERSION // 1)

        elif experience_structure is not None:
            # Create the necessary dirs for the vault
            os.makedirs(self._base_path)

            # TODO with serialize_tree?
            def get_json_ready(obj: Any) -> Any:
                """Ensure that the object is json serializable. Convert to string if not.

                Args:
                    obj (Any): Object to be considered

                Returns:
                    Any: json serializable object
                """
                if not isinstance(obj, (bool, str, int, float, type(None))):
                    return str(obj)
                else:
                    return obj

            metadata_json_ready = jax.tree_util.tree_map(get_json_ready, metadata)

            # We save the structure of the buffer state
            #  e.g. [(128, 100, 4), jnp.int32]
            # We will use this structure to map over the data stores later
            serialised_experience_structure = jax.tree_map(
                lambda x: [str(x.shape), x.dtype.name],
                serialize_tree(
                    # Get shape and dtype of each leaf, without serialising the structure itself
                    jax.eval_shape( 
                        lambda: experience_structure,
                    ),
                )
            )

            # Construct metadata
            self._metadata = {
                "version": VERSION,
                "structure": serialised_experience_structure,
                **(metadata_json_ready or {}),  # Allow user to save extra metadata
            }
            # Dump metadata to file
            metadata_path.write_text(json.dumps(self._metadata))
        else:
            raise ValueError("Vault does not exist and no init_fbx_state provided.")

        ## ---
        # We must now build the tree structure from the metadata, whether created here or loaded from file
        if experience_structure is None:
            # If an example state is not provided, we simply load from the metadata
            #  and the result will be a dictionary.
            self._tree_structure = self._metadata["structure"]
        else:
            # If an example state is provided, we try deserialise into that structure
            self._tree_structure = deserialize_tree(
                self._metadata["structure"],
                target=experience_structure,
            )

        # Each leaf of the fbx_state.experience maps to a data store
        self._all_ds = jax.tree_util.tree_map_with_path(
            lambda path, x: self._init_leaf(
                name=_path_to_ds_name(path),
                leaf=x,
                create_ds=not base_path_exists,
            ),
            self._tree_structure,
            is_leaf=lambda x: isinstance(x, list),  # The list [shape, dtype] is a leaf
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
        return {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": f"{DRIVER}{self._base_path}",
                "path": name,
            },
        }

    def _init_leaf(
        self, name: str, leaf: list, create_ds: bool = False
    ) -> ts.TensorStore:
        spec = self._get_base_spec(name)
        leaf_shape = make_tuple(leaf[0])
        leaf_dtype = leaf[1]
        leaf_ds = ts.open(
            spec,
            # Only specify dtype and shape if we are creating a checkpoint
            dtype=leaf_dtype if create_ds else None,
            shape=(
                leaf_shape[0],  # Batch dim
                TIME_AXIS_MAX_LENGTH,  # Time dim
                *leaf_shape[2:],  # Experience dim
            )
            if create_ds
            else None,  # Don't impose shape if we are loading a vault
            # Only create datastore if we are creating the vault
            create=create_ds,
        ).result()  # Synchronous
        return leaf_ds

    async def _write_leaf(
        self,
        source_leaf: jax.Array,
        dest_leaf: ts.TensorStore,
        source_interval: Tuple[int, int],
        dest_start: int,
    ) -> None:
        dest_interval = (
            dest_start,
            dest_start + (source_interval[1] - source_interval[0]),  # type: ignore
        )
        await dest_leaf[:, slice(*dest_interval), ...].write(
            source_leaf[:, slice(*source_interval), ...],
        )

    async def _write_chunk(
        self,
        fbx_state: TrajectoryBufferState,
        source_interval: Tuple[int, int],
        dest_start: int,
    ) -> None:
        # Write to each ds
        futures_tree = jax.tree_util.tree_map(
            lambda x, ds: self._write_leaf(
                source_leaf=x,
                dest_leaf=ds,
                source_interval=source_interval,
                dest_start=dest_start,
            ),
            fbx_state.experience,  # x = experience
            self._all_ds,  # ds = data stores
        )
        futures, _ = jax.tree_util.tree_flatten(futures_tree)
        await asyncio.gather(*futures)

    def write(
        self,
        fbx_state: TrajectoryBufferState,
        source_interval: Tuple[int, int] = (0, 0),
        dest_start: Optional[int] = None,
    ) -> int:
        # TODO: more than one current_index if B > 1
        fbx_current_index = int(fbx_state.current_index)

        # By default, we write from `last received` to `current index` [CI]
        if source_interval == (0, 0):
            source_interval = (self._last_received_fbx_index, fbx_current_index)

        if source_interval[1] == source_interval[0]:
            # Nothing to write
            return 0

        elif source_interval[1] > source_interval[0]:
            # Vanilla write, no wrap around
            dest_start = self.vault_index if dest_start is None else dest_start
            asyncio.run(
                self._write_chunk(
                    fbx_state=fbx_state,
                    source_interval=source_interval,
                    dest_start=dest_start,
                )
            )
            written_length = source_interval[1] - source_interval[0]

        elif source_interval[1] < source_interval[0]:
            # Wrap around!

            # Get dest start
            dest_start = self.vault_index if dest_start is None else dest_start
            # Get seq dim
            fbx_max_index = get_tree_shape_prefix(fbx_state.experience, n_axes=2)[1]

            # From last received to max
            source_interval_a = (source_interval[0], fbx_max_index)
            time_length_a = source_interval_a[1] - source_interval_a[0]

            asyncio.run(
                self._write_chunk(
                    fbx_state=fbx_state,
                    source_interval=source_interval_a,
                    dest_start=dest_start,
                )
            )

            # From 0 (wrapped) to CI
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

        # Update vault index, and write this to the ds too
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
        return jnp.asarray(read_leaf[:, slice(*read_interval), ...].read().result())

    def read(
        self,
        timesteps: Optional[int] = None,
        percentiles: Optional[Tuple[int, int]] = None,
    ) -> TrajectoryBufferState:
        """Read from the vault."""

        if timesteps is None and percentiles is None:
            read_interval = (0, self.vault_index)
        elif timesteps is not None:
            read_interval = (self.vault_index - timesteps, self.vault_index)
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
            self._tree_structure,
            self._all_ds,  # data stores
            is_leaf=lambda x: isinstance(x, list),
        )

        return TrajectoryBufferState(
            experience=read_result,
            current_index=jnp.array(self.vault_index, dtype=int),
            is_full=jnp.array(True, dtype=bool),
        )
