# src/sse_core/compiler/parser.py
import yaml
from pydantic import BaseModel, Field
from sse_core.compiler.models import (
    ComponentConfig,
    NodeDirectory,
    SimulationConfig,
)


class CircuitNetlist(BaseModel):
    """
    The master AST container representing a validated, complete circuit netlist.
    """

    schema_version: str = Field(
        ..., description="Standardized schema version identifier."
    )
    simulation: SimulationConfig = Field(
        ..., description="Global simulation execution parameters."
    )
    nodes: NodeDirectory = Field(
        ..., description="The partitioned node directory (free vs regulated)."
    )
    components: list[ComponentConfig] = Field(
        default_factory=list, description="List of all circuit components."
    )


class SSEParser:
    """
    Responsible for loading raw YAML configurations and instantiating
    the validated CircuitNetlist AST.
    """

    @staticmethod
    def parse_string(yaml_content: str) -> CircuitNetlist:
        """
        Parses a raw YAML configuration string.

        Raises:
            ValueError: If the YAML syntax is malformed.
            ValidationError: If any schema validation or physical bound checks fail.
        """
        try:
            raw_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Malformed YAML syntax: {e}") from e

        if not isinstance(raw_data, dict):
            raise ValueError(
                "Invalid netlist format: root element must be a dictionary."
            )

        # Let Pydantic orchestrate all validation checks across our model hierarchy
        return CircuitNetlist(**raw_data)

    @classmethod
    def parse_file(cls, file_path: str) -> CircuitNetlist:
        """
        Reads a netlist file from disk and parses its content.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.parse_string(content)
