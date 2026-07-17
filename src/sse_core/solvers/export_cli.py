# src/sse_core/solvers/export_cli.py
import argparse
import os
import sys

import h5py
import numpy as np


def export_hdf5_to_tabular(h5_path: str, output_dir: str, delimiter: str = ",") -> None:
    """
    Parses an SSE HDF5 simulation output file and exports its datasets
    to plottable, delimited text files (CSV/TSV/DAT)[cite: 231].
    """
    if not os.path.exists(h5_path):
        print(f"Error: Simulation file '{h5_path}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as h5f:
        if "data" not in h5f:
            print(
                "Error: Invalid file format. No 'data' group found in HDF5 file.",
                file=sys.stderr,
            )
            sys.exit(1)

        data = h5f["data"]
        time_arr = data["time"][:]
        charges_arr = data["charges"][:]

        # Extract names of free nodes from metadata
        free_nodes = [
            n.decode("utf-8") if isinstance(n, bytes) else n
            for n in h5f["meta"].attrs["free_nodes"]
        ]

        # 1. Export Time & Charges combined
        charges_out_path = os.path.join(output_dir, "trajectory_charges.csv")
        header = delimiter.join(["time"] + [f"q_{node}" for node in free_nodes])

        combined_charges = np.column_stack((time_arr, charges_arr))
        np.savetxt(
            charges_out_path,
            combined_charges,
            delimiter=delimiter,
            header=header,
            comments="",
        )
        print(f" -> Exported charge trajectories to: {charges_out_path}")

        # 2. Export Voltages if they exist in the dataset [cite: 235]
        if "voltages" in data:
            voltages_gp = data["voltages"]
            voltages_out_path = os.path.join(output_dir, "trajectory_voltages.csv")

            node_names = list(voltages_gp.keys())
            voltage_columns = [time_arr]
            for name in node_names:
                voltage_columns.append(voltages_gp[name][:])

            combined_voltages = np.column_stack(voltage_columns)
            v_header = delimiter.join(["time"] + [f"v_{name}" for name in node_names])
            np.savetxt(
                voltages_out_path,
                combined_voltages,
                delimiter=delimiter,
                header=v_header,
                comments="",
            )
            print(f" -> Exported voltage trajectories to: {voltages_out_path}")


def run_export_cli() -> None:
    """
    Main entry point for the sse-export command-line utility[cite: 250, 251].
    """
    parser = argparse.ArgumentParser(
        prog="sse-export",
        description="SSE Telemetry Exporter: Convert simulation HDF5 databases to plottable CSV/TSV formats[cite: 231, 252].",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input .h5 simulation run database[cite: 234].",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="exports",
        help="Directory to write plain-text plottable output files (default: 'exports').",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        type=str,
        default=",",
        help="Field delimiter for output text files (default: ',')[cite: 236].",
    )

    args = parser.parse_args()
    export_hdf5_to_tabular(args.input, args.output_dir, args.delimiter)
