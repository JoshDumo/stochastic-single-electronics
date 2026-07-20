# src/sse_core/compiler/builder.py
from dataclasses import dataclass

import numpy as np
from sse_core.compiler.linter import SSETopologyLinter
from sse_core.compiler.parser import CircuitNetlist, SSEParser


@dataclass
class CompiledAssembly:
    """
    A statically compiled numerical representation of the circuit,
    ready for execution by high-performance solvers.
    """

    free_names: list[str]
    regulated_names: list[str]
    C_inv: np.ndarray  # Inverse of free capacitance matrix (Nf x Nf)
    Cx: np.ndarray  # Free-to-regulated mutual capacitance matrix (Nf x Nr)
    Cr: np.ndarray  # Regulated-to-regulated capacitance matrix (Nr x Nr)
    free_Delta: np.ndarray  # Reduced incidence matrix for active devices (Nf x Nd)
    dV_precomputed: np.ndarray  # Voltage change offset across devices per jump (Nd,)
    device_terminals: list[
        tuple[int, int]
    ]  # Index pair (node_a, node_b) for each active device


class SSEMatrixBuilder:
    """
    Translates a validated CircuitNetlist AST into partitioned numerical matrices,
    evaluating matrix solvability and assembling active device charge transition graphs.
    """

    def __init__(self, netlist):
        self.netlist = netlist

        # Index lookup maps
        self.free_names = [node.name for node in netlist.nodes.free]
        self.regulated_names = [node.name for node in netlist.nodes.regulated]

        self.all_names = self.free_names + self.regulated_names
        self.name_to_idx = {name: idx for idx, name in enumerate(self.all_names)}

        self.N = len(self.all_names)
        self.Nf = len(self.free_names)
        self.Nr = len(self.regulated_names)

    def assemble(self) -> CompiledAssembly:
        """
        Assemble the full Maxwell capacitance matrix and the active-device
        incidence matrix.

        Every capacitor between nodes a and b contributes the standard stamp

            +C at (a, a)
            +C at (b, b)
            -C at (a, b)
            -C at (b, a)

        The full matrix is then partitioned as

            M = [[C,  Cx],
                [Cx.T, Cr]]

        where C corresponds to free nodes and Cr to regulated nodes.
        """

        # =====================================================================
        # 1. Assemble the full Maxwell capacitance matrix
        # =====================================================================

        M = np.zeros((self.N, self.N), dtype=np.float64)

        for comp in self.netlist.components:
            if comp.type != "capacitor":
                continue

            node_a, node_b = comp.terminals
            idx_a = self.name_to_idx[node_a]
            idx_b = self.name_to_idx[node_b]
            capacitance = float(comp.specs["capacitance"])

            if capacitance < 0.0:
                raise ValueError(
                    f"ERR_MATH_202: Capacitor '{comp.name}' has negative "
                    f"capacitance {capacitance}."
                )

            # Complete symmetric capacitor stamp.
            M[idx_a, idx_a] += capacitance
            M[idx_b, idx_b] += capacitance
            M[idx_a, idx_b] -= capacitance
            M[idx_b, idx_a] -= capacitance

        # This should be guaranteed by the capacitor-stamping operation.
        if not np.allclose(M, M.T, rtol=1e-13, atol=1e-30):
            raise RuntimeError(
                "ERR_MATH_203: The assembled capacitance matrix is not symmetric."
            )

        # Partition the full matrix:
        #
        #     M = [[C,    Cx],
        #          [Cx.T, Cr]]
        #
        C = M[: self.Nf, : self.Nf].copy()
        Cx = M[: self.Nf, self.Nf :].copy()
        Cr = M[self.Nf :, self.Nf :].copy()

        # Invert the free-node capacitance block.
        if self.Nf == 0:
            C_inv = np.empty((0, 0), dtype=np.float64)
        else:
            try:
                condition_number = np.linalg.cond(C)

                if (
                    not np.isfinite(condition_number)
                    or condition_number >= 1.0 / np.finfo(float).eps
                ):
                    raise np.linalg.LinAlgError("Singular or ill-conditioned matrix")

                C_inv = np.linalg.inv(C)

            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "ERR_MATH_201: The free-node capacitance matrix C is "
                    "singular or uninvertible. Ensure every free-node island "
                    "has a capacitive connection to the rest of the circuit."
                ) from exc

        # =====================================================================
        # 2. Compile the active-device incidence matrix
        # =====================================================================

        active_comps = [
            comp
            for comp in self.netlist.components
            if comp.type
            in [
                "tunnel_junction",
                "n_channel_mosfet",
                "p_channel_mosfet",
                "diode",
            ]
        ]

        number_of_devices = len(active_comps)

        free_Delta = np.zeros(
            (self.Nf, number_of_devices),
            dtype=np.float64,
        )

        device_terminals: list[tuple[int, int]] = []

        dV_precomputed = np.zeros(
            number_of_devices,
            dtype=np.float64,
        )

        for device_index, comp in enumerate(active_comps):
            if hasattr(comp.terminals, "drain"):
                drain_name = comp.terminals.drain
                source_name = comp.terminals.source
            else:
                drain_name = comp.terminals[0]
                source_name = comp.terminals[1]

            idx_drain = self.name_to_idx[drain_name]
            idx_source = self.name_to_idx[source_name]

            device_terminals.append((idx_drain, idx_source))

            # Native simulator convention:
            # a positive transition moves one electron from drain to source.
            if idx_drain < self.Nf:
                free_Delta[idx_drain, device_index] += 1.0

            if idx_source < self.Nf:
                free_Delta[idx_source, device_index] -= 1.0

            # Preserve the existing precomputation behaviour in this patch.
            if self.Nf > 0:
                dV_free = C_inv @ free_Delta[:, device_index]

                v_drain = dV_free[idx_drain] if idx_drain < self.Nf else 0.0

                v_source = dV_free[idx_source] if idx_source < self.Nf else 0.0

                dV_precomputed[device_index] = v_drain - v_source

        return CompiledAssembly(
            free_names=self.free_names,
            regulated_names=self.regulated_names,
            C_inv=C_inv,
            Cx=Cx,
            Cr=Cr,
            free_Delta=free_Delta,
            dV_precomputed=dV_precomputed,
            device_terminals=device_terminals,
        )


# Orchestrator
# -------------


class SSECompiler:
    """
    The main user-facing compiler API. Coordinates parsing,
    topological linting, and matrix assembly in a single call.
    """

    @staticmethod
    def compile_string(yaml_content: str) -> CompiledAssembly:
        """
        Compiles a raw YAML string into a numerical circuit assembly.

        Raises:
            ValidationError: If the YAML schema is invalid.
            ValueError: If topological lint checks or matrix calculations fail.
        """
        # 1. Parse YAML to AST
        netlist: CircuitNetlist = SSEParser.parse_string(yaml_content)

        # 2. Run Topological Linter
        linter = SSETopologyLinter(netlist)
        errors = linter.lint()
        if errors:
            formatted_errors = "\n".join(f"  - {err}" for err in errors)
            raise ValueError(
                f"ERR_NET_100: Topological linting failed with the following errors:\n{formatted_errors}"
            )

        # 3. Assemble and return numerical matrices
        builder = SSEMatrixBuilder(netlist)
        return builder.assemble()

    @classmethod
    def compile_file(cls, file_path: str) -> CompiledAssembly:
        """
        Reads a YAML file from disk and compiles it.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.compile_string(content)
