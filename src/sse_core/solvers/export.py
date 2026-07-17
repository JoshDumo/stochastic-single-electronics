# src/sse_core/solvers/export.py
from typing import Any

import h5py
from sse_core.compiler.builder import CompiledAssembly


class TelemetryExporter:
    """
    Handles high-performance serialization of stochastic trajectories
    to standard HDF5 structures compatible with SPICE-style raw data viewers.
    """

    @staticmethod
    def export_to_hdf5(
        filepath: str, history: dict[str, Any], assembly: CompiledAssembly, v_th: float
    ) -> None:
        """
        Serializes the simulation history dictionary directly to an HDF5 container.
        """
        with h5py.File(filepath, "w") as h5f:
            # 1. Populate /meta attributes
            meta = h5f.create_group("meta")
            meta.attrs["v_th"] = v_th
            meta.attrs["free_nodes"] = [
                name.encode("utf-8") for name in assembly.free_names
            ]
            meta.attrs["regulated_nodes"] = [
                name.encode("utf-8") for name in assembly.regulated_names
            ]

            # 2. Populate /data group
            data = h5f.create_group("data")

            # Write trajectories
            data.create_dataset(
                "time", data=history["time"], compression="gzip", chunks=True
            )
            data.create_dataset(
                "charges", data=history["charge"], compression="gzip", chunks=True
            )

            # Map node voltages over time
            potentials_group = data.create_group("voltages")
            for node_name, voltage_trajectory in history["potentials"].items():
                potentials_group.create_dataset(
                    node_name, data=voltage_trajectory, compression="gzip", chunks=True
                )
