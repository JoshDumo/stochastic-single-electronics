# src/sse_core/compiler/linter.py
from collections import defaultdict

from sse_core.compiler.models import MOSFETTerminals
from sse_core.compiler.parser import CircuitNetlist


class SSETopologyLinter:
    """
    Analyzes a parsed CircuitNetlist AST to verify physical and graph-level
    topological correctness before mathematical matrix compilation begins.
    """

    def __init__(self, netlist: CircuitNetlist):
        self.netlist = netlist

        # Build set of all registered node names for O(1) lookup
        self.all_nodes: set[str] = {node.name for node in self.netlist.nodes.free} | {
            node.name for node in self.netlist.nodes.regulated
        }

    def lint(self) -> list[str]:
        """
        Executes all topological checks. Returns a list of error messages.
        If the list is empty, the topology is fully valid.
        """
        errors: list[str] = []

        # Run individual rule checks

        self._check_reference_exists(errors)
        self._check_terminals_and_connectivity(errors)

        return errors

    def _check_reference_exists(self, errors: list[str]) -> None:
        """Rule ERR_NET_104: Enforce at least one regulated (fixed potential) node."""
        if not self.netlist.nodes.regulated:
            errors.append(
                "ERR_NET_104: No regulated node detected. "
                "The circuit must declare at least one regulated node to provide a voltage reference."
            )

    def _check_terminals_and_connectivity(self, errors: list[str]) -> None:
        """
        Rule ERR_NET_101 & ERR_NET_102:
        Checks for dangling terminals and identifies isolated/floating free nodes.
        """
        # Track how many times each registered node is connected to a component
        connection_counts: dict[str, int] = defaultdict(int)

        for comp in self.netlist.components:
            # Flatten active and control terminals depending on component representation
            terminals: list[str] = []
            if isinstance(comp.terminals, list):
                terminals = comp.terminals
            elif isinstance(comp.terminals, MOSFETTerminals):
                terminals = [
                    comp.terminals.drain,
                    comp.terminals.gate,
                    comp.terminals.source,
                    comp.terminals.bulk,
                ]

            for term in terminals:
                # Check ERR_NET_101: Dangling terminal
                if term not in self.all_nodes:
                    errors.append(
                        f"ERR_NET_101: Component '{comp.name}' references undefined "
                        f"terminal node '{term}'."
                    )
                else:
                    connection_counts[term] += 1

        # Check ERR_NET_102: Isolated/floating nodes (connectivity degree < 2)
        # CRITICAL FIX: Only enforce this rule for FREE (internal) nodes!
        free_node_names = {node.name for node in self.netlist.nodes.free}
        for node_name in free_node_names:
            count = connection_counts[node_name]
            if count < 2:
                errors.append(
                    f"ERR_NET_102: Isolated node '{node_name}' detected. "
                    f"It has only {count} connection(s). Nodes must connect to at least 2 components."
                )
