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
        Builds the capacitance matrix using MNA rules.
        It isolates regulated nodes and filters the ground reference to
        ensure the capacitance matrix C is non-singular and physically grounded.
        """

        # Initialize full system matrix (N x N)
        M = np.zeros((self.N, self.N))

        for comp in self.netlist.components:
            if comp.type == "capacitor":
                idx_a = self.name_to_idx[comp.terminals[0]]
                idx_b = self.name_to_idx[comp.terminals[1]]
                cap_val = comp.specs["capacitance"]

                # Stamp diagonals
                if idx_a < self.Nf:
                    M[idx_a, idx_a] += cap_val
                if idx_b < self.Nf:
                    M[idx_b, idx_b] += cap_val

                # Stamp coupling (Works for Free-Free AND Free-Regulated)
                # We only care about the top-left (C) and top-right (Cx) blocks
                if idx_a < self.Nf and idx_b < self.Nf:
                    M[idx_a, idx_b] -= cap_val
                    M[idx_b, idx_a] -= cap_val
                elif idx_a < self.Nf and idx_b >= self.Nf:
                    M[idx_a, idx_b] -= cap_val  # Fills Cx block
                elif idx_b < self.Nf and idx_a >= self.Nf:
                    M[idx_b, idx_a] -= cap_val  # Fills Cx block (transpose)

        # C = Floating Nodes Only (index 0 to Nf-1)
        C = M[0 : self.Nf, 0 : self.Nf]
        Cx = M[0 : self.Nf, self.Nf : self.N]
        Cr = M[self.Nf : self.N, self.Nf : self.N]

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

        # 4. Compile Active Device Incidence Matrix (free_Delta)
        active_comps = [
            c
            for c in self.netlist.components
            if c.type
            in ["tunnel_junction", "n_channel_mosfet", "p_channel_mosfet", "diode"]
        ]
        Nd = len(active_comps)
        free_Delta = np.zeros((self.Nf, Nd))
        device_terminals = []
        dV_precomputed = np.zeros(Nd)

        for d, comp in enumerate(active_comps):
            idx_drain = self.name_to_idx[
                comp.terminals.drain
                if hasattr(comp.terminals, "drain")
                else comp.terminals[0]
            ]
            idx_source = self.name_to_idx[
                comp.terminals.source
                if hasattr(comp.terminals, "source")
                else comp.terminals[1]
            ]

            device_terminals.append((idx_drain, idx_source))

            # Electron tunneling direction (standard positive = A to B)
            if idx_drain < self.Nf:
                free_Delta[idx_drain, d] += 1.0
            if idx_source < self.Nf:
                free_Delta[idx_source, d] -= 1.0

            # Compute dV jump across the device terminals
            if self.Nf > 0:
                dV_free = C_inv @ free_Delta[:, d]
                v_a = dV_free[idx_drain] if idx_drain < self.Nf else 0.0
                v_b = dV_free[idx_source] if idx_source < self.Nf else 0.0
                dV_precomputed[d] = v_a - v_b

        return CompiledAssembly(
            self.free_names,
            self.regulated_names,
            C_inv,
            Cx,
            Cr,
            free_Delta,
            dV_precomputed,
            device_terminals,
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
