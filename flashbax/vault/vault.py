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

from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.utils import get_tree_shape_prefix

# CURRENT LIMITATIONS / TODO LIST
# - Only tested with flat buffers
# - Reloading must be with dicts, not namedtuples

DRIVER = "file://"
METADATA_FILE = "metadata.json"
TIME_AXIS_MAX_LENGTH = int(10e12)  # Upper bound on the length of the time axis
VERSION = 0.1


class Vault:
    def __init__(
        self,
        vault_name: str,
        init_fbx_state: Optional[TrajectoryBufferState] = None,
        rel_dir: str = "vaults",
        vault_uid: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        vault_str = vault_uid if vault_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        self._base_path = os.path.join(os.getcwd(), rel_dir, vault_name, vault_str)

        # We use epath for metadata
        metadata_path = epath.Path(os.path.join(self._base_path, METADATA_FILE))

        # Check if the vault exists, otherwise create the necessary dirs and files
        base_path_exists = os.path.exists(self._base_path)
        if base_path_exists:
            self._metadata = json.loads(metadata_path.read_text())

            # Ensure minor versions match
            assert (self._metadata["version"] // 1) == (VERSION // 1)

        elif init_fbx_state is not None:
            # init_fbx_state must be a TrajectoryBufferState
            assert isinstance(init_fbx_state, TrajectoryBufferState)

            # Create the necessary dirs for the vault
            os.makedirs(self._base_path)

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
            experience_structure = jax.tree_map(
                lambda x: [str(x.shape), str(x.dtype)],
                init_fbx_state.experience,
            )
            self._metadata = {
                "version": VERSION,
                "structure": experience_structure,
                **(metadata_json_ready or {}),  # Allow user to save extra metadata
            }
            metadata_path.write_text(json.dumps(self._metadata))
        else:
            raise ValueError("Vault does not exist and no init_fbx_state provided.")

        # Keep a data store for the vault index
        self._vault_index_ds = ts.open(
            self._get_base_spec("vault_index"),
            dtype=jnp.int32,
            shape=(1,),
            create=not base_path_exists,
        ).result()
        self.vault_index = int(self._vault_index_ds.read().result()[0])

        # Each leaf of the fbx_state.experience is a data store
        self._all_ds = jax.tree_util.tree_map_with_path(
            lambda path, x: self._init_leaf(
                name=jax.tree_util.keystr(path),  # Use the path as the name
                leaf=x,
                create_checkpoint=not base_path_exists,
            ),
            self._metadata["structure"],
            is_leaf=lambda x: isinstance(x, list),
        )

        self._last_received_fbx_index = 0

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
        self, name: str, leaf: list, create_checkpoint: bool = False
    ) -> ts.TensorStore:
        spec = self._get_base_spec(name)
        leaf_shape = make_tuple(leaf[0])
        leaf_dtype = leaf[1]
        leaf_ds = ts.open(
            spec,
            # Only specify dtype and shape if we are creating a checkpoint
            dtype=leaf_dtype if create_checkpoint else None,
            shape=(
                leaf_shape[0],  # Batch dim
                TIME_AXIS_MAX_LENGTH,  # Time dim
                *leaf_shape[2:],  # Experience dim
            )
            if create_checkpoint
            else None,
            # Only create directory if we are creating a checkpoint
            create=create_checkpoint,
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
        return read_leaf[:, slice(*read_interval), ...].read().result()

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
            self._metadata["structure"],  # just for structure
            self._all_ds,  # data stores
            is_leaf=lambda x: isinstance(x, list),
        )

        return TrajectoryBufferState(
            experience=read_result,
            current_index=jnp.array(self.vault_index, dtype=int),
            is_full=jnp.array(True, dtype=bool),
        )
