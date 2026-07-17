# tests/test_solvers/test_export.py
import os

import h5py
import numpy as np
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
from sse_core.solvers.export import TelemetryExporter
from sse_core.solvers.gillespie import GillespieSolver


def test_hdf5_serialization_integrity(tmp_path):
    """
    Run a simulation trajectory, serialize the output to HDF5,
    and assert that the recovered data matches the in-memory states.
    """
    yaml_circuit = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1.0e-9, seed: 42}
    nodes:
      free: [{"name": "island"}]
      regulated: [{"name": "gnd", "type": "ground"}]
    components:
      - type: "capacitor"
        name: "C1"
        terminals: ["island", "gnd"]
        specs: {capacitance: 1.0e-15}
      - type: "tunnel_junction"
        name: "TJ1"
        terminals: ["island", "gnd"]
        specs: {resistance: 1.0e5}
    """
    parsed_netlist = SSEParser.parse_string(yaml_circuit)
    assembly = SSECompiler.compile_string(yaml_circuit)
    solver = GillespieSolver(parsed_netlist, assembly)

    # Run trajectory
    q_init = np.array([0])
    vr = np.array([0.0])
    history = solver.simulate(q_init, vr, max_steps=10)

    # Export to temporary path
    hdf5_file = os.path.join(tmp_path, "simulation_run.h5")
    TelemetryExporter.export_to_hdf5(hdf5_file, history, assembly, solver.v_th)

    # Re-open and verify structural integrity
    with h5py.File(hdf5_file, "r") as h5f:
        # Verify metadata
        assert "meta" in h5f

        # Safe decode check: converts bytes to strings if necessary, handling both types elegantly
        read_free = [
            n.decode("utf-8") if isinstance(n, bytes) else n
            for n in h5f["meta"].attrs["free_nodes"]
        ]
        read_regulated = [
            n.decode("utf-8") if isinstance(n, bytes) else n
            for n in h5f["meta"].attrs["regulated_nodes"]
        ]

        assert read_free == ["island"]
        assert read_regulated == ["gnd"]

        # Verify time dataset
        assert "data/time" in h5f
        assert np.allclose(h5f["data/time"][:], history["time"])

        # Verify charges dataset
        assert "data/charges" in h5f
        assert np.allclose(h5f["data/charges"][:], history["charge"])

        # Verify dynamic voltages
        assert "data/voltages/island" in h5f
        assert np.allclose(
            h5f["data/voltages/island"][:], history["potentials"]["island"]
        )
