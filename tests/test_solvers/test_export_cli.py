# tests/test_solvers/test_export_cli.py
import os

import numpy as np
from sse_core.compiler.builder import SSECompiler
from sse_core.compiler.parser import SSEParser
from sse_core.solvers.export import TelemetryExporter
from sse_core.solvers.export_cli import export_hdf5_to_tabular
from sse_core.solvers.gillespie import GillespieSolver


def test_sse_export_cli_writes_correct_tabular_files(tmp_path):
    """
    Run a simulation, export to HDF5, run the renamed sse-export CLI logic,
    and verify that readable plain-text tabular files are created with correct headers.
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
    assembly = SSECompiler.compile_string(yaml_circuit)
    parsed_netlist = SSEParser.parse_string(yaml_circuit)
    solver = GillespieSolver(parsed_netlist, assembly)

    # Run and write master HDF5 database
    history = solver.simulate(np.array([0]), np.array([0.0]), max_steps=5)
    h5_filepath = os.path.join(tmp_path, "run.h5")
    TelemetryExporter.export_to_hdf5(h5_filepath, history, assembly, solver.v_th)

    # Invoke our renamed export processing subroutine
    export_dir = os.path.join(tmp_path, "txt_plots")
    export_hdf5_to_tabular(h5_filepath, export_dir, delimiter=",")

    # Verify plain-text output files exist
    charge_file = os.path.join(export_dir, "trajectory_charges.csv")
    voltage_file = os.path.join(export_dir, "trajectory_voltages.csv")

    assert os.path.exists(charge_file)
    assert os.path.exists(voltage_file)

    # Read back a line from CSV to verify header structure
    with open(charge_file, "r") as f:
        first_line = f.readline().strip()
        assert first_line == "time,q_island"
