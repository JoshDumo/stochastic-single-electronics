# src/sse_core/compiler/builder.py
from dataclasses import dataclass

import numpy as np
from sse_core.compiler.linter import SSETopologyLinter
from sse_core.compiler.models import MOSFETTerminals
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
        Builds, partitions, and compiles the system matrices and active device incidence graphs.

        Raises:
            ValueError: If the free capacitance matrix C is singular (ERR_MATH_201).
        """
        # =====================================================================
        # 1. Compile Capacitance Matrices
        # =====================================================================
        M = np.zeros((self.N, self.N))

        for comp in self.netlist.components:
            if comp.type == "capacitor":
                node_a_name, node_b_name = comp.terminals
                idx_a = self.name_to_idx[node_a_name]
                idx_b = self.name_to_idx[node_b_name]
                cap_val = comp.specs["capacitance"]

                M[idx_a, idx_a] += cap_val
                M[idx_b, idx_b] += cap_val
                M[idx_a, idx_b] -= cap_val
                M[idx_b, idx_a] -= cap_val

        C = M[0 : self.Nf, 0 : self.Nf]
        Cx = M[0 : self.Nf, self.Nf : self.N]
        Cr = M[self.Nf : self.N, self.Nf : self.N]  # <--- Extract Cr block

        try:
            if np.linalg.cond(C) > 1 / np.finfo(float).eps:
                raise np.linalg.LinAlgError("Singular matrix")
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "ERR_MATH_201: The free capacitance matrix C is singular or uninvertible. "
                "Ensure your circuit does not contain completely isolated node islands."
            ) from e

        # =====================================================================
        # 2. Compile Active Device Incidence Matrix (free_Delta)
        # =====================================================================
        # Identify active components
        active_comps = [
            comp
            for comp in self.netlist.components
            if comp.type
            in ["tunnel_junction", "n_channel_mosfet", "p_channel_mosfet", "diode"]
        ]
        Nd = len(active_comps)

        free_Delta = np.zeros((self.Nf, Nd))
        device_terminals: list[tuple[int, int]] = []
        dV_precomputed = np.zeros(Nd)

        for d, comp in enumerate(active_comps):
            # Extract charge-transfer terminals (A -> target, B -> source)
            if comp.type in ["tunnel_junction", "diode"]:
                node_a_name, node_b_name = comp.terminals
            else:  # MOSFETs: charge transfer happens between drain and source
                terminals: MOSFETTerminals = comp.terminals
                node_a_name = terminals.drain
                node_b_name = terminals.source

            idx_a = self.name_to_idx[node_a_name]
            idx_b = self.name_to_idx[node_b_name]
            device_terminals.append((idx_a, idx_b))

            # Populate the reduced incidence matrix columns (if nodes are free)
            if idx_a < self.Nf:
                free_Delta[idx_a, d] = 1.0
            if idx_b < self.Nf:
                free_Delta[idx_b, d] = -1.0

            # Compute the precalculated voltage step: delta_V = C_inv * delta_free
            # Represents the voltage jump across this device's terminals when it fires.
            if self.Nf > 0:
                delta_column = free_Delta[:, d]
                dV_free = C_inv @ delta_column

                # Get potentials at terminal nodes (treating regulated potentials as fixed 0V for delta check)
                v_a = dV_free[idx_a] if idx_a < self.Nf else 0.0
                v_b = dV_free[idx_b] if idx_b < self.Nf else 0.0
                dV_precomputed[d] = v_a - v_b

        return CompiledAssembly(
            free_names=self.free_names,
            regulated_names=self.regulated_names,
            C_inv=C_inv,
            Cx=Cx,
            Cr=Cr,  # <--- Added to assembly
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
