# src/sse_core/compiler/builder.py
from dataclasses import dataclass

import numpy as np


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


class SSEMatrixBuilder:
    """
    Translates a validated CircuitNetlist AST into partitioned numerical matrices,
    evaluating matrix solvability (invertibility) and topological parameters.
    """

    def __init__(self, netlist):
        self.netlist = netlist

        # Build strict lookup indexing maps to convert string names to matrix dimensions
        self.free_names = [node.name for node in netlist.nodes.free]
        self.regulated_names = [node.name for node in netlist.nodes.regulated]

        self.all_names = self.free_names + self.regulated_names
        self.name_to_idx = {name: idx for idx, name in enumerate(self.all_names)}

        self.N = len(self.all_names)
        self.Nf = len(self.free_names)
        self.Nr = len(self.regulated_names)

    def assemble(self) -> CompiledAssembly:
        """
        Builds, partitions, and compiles the system capacitance matrices.

        Raises:
            ValueError: If the free capacitance matrix C is singular (ERR_MATH_201).
        """
        # 1. Initialize global Maxwell matrix with zeros
        M = np.zeros((self.N, self.N))

        # 2. Populate Maxwell elements from capacitor specs
        for comp in self.netlist.components:
            if comp.type == "capacitor":
                node_a_name, node_b_name = comp.terminals
                idx_a = self.name_to_idx[node_a_name]
                idx_b = self.name_to_idx[node_b_name]
                cap_val = comp.specs["capacitance"]

                # Accumulate self-capacitances (diagonals)
                M[idx_a, idx_a] += cap_val
                M[idx_b, idx_b] += cap_val

                # Accumulate mutual coupling (off-diagonals)
                M[idx_a, idx_b] -= cap_val
                M[idx_b, idx_a] -= cap_val

        # 3. Slice global matrix into free (C) and regulated (Cx) blocks
        C = M[0 : self.Nf, 0 : self.Nf]
        Cx = M[0 : self.Nf, self.Nf : self.N]

        # 4. Invert the free matrix C to get C_inv
        try:
            # Check condition number to explicitly catch singular or ill-conditioned matrices
            if np.linalg.cond(C) > 1 / np.finfo(float).eps:
                raise np.linalg.LinAlgError("Singular matrix")

            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "ERR_MATH_201: The free capacitance matrix C is singular or uninvertible. "
                "Ensure your circuit does not contain completely isolated node islands."
            ) from e

        return CompiledAssembly(
            free_names=self.free_names,
            regulated_names=self.regulated_names,
            C_inv=C_inv,
            Cx=Cx,
        )
