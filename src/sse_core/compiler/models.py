# src/sse_core/compiler/models.py
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


# Simulation Block
# -----------------------
class SimulationConfig(BaseModel):
    """
    Validates global simulation execution parameters, timing limits,
    and physical thermal voltage thresholds.
    """

    solver: Literal["gillespie", "tau_leaping"] = Field(
        ..., description="The mathematical stochastic solver to execute."
    )
    t_finish: float = Field(
        ..., gt=0.0, description="Total simulation runtime in seconds."
    )
    t_start: float = Field(default=0.0, ge=0.0, description="Initial simulation time.")
    time_step: float | None = Field(
        default=None,
        gt=0.0,
        description="Fixed interval time-step. Mandatory only for tau-leaping.",
    )
    seed: int | None = Field(
        default=None, ge=0, description="Optional seed for reproducibility."
    )
    v_th: float = Field(
        default=0.0259,
        gt=0.0,
        description="Physical thermal voltage (V_th = kT/e) in Volts.",
    )

    @model_validator(mode="after")
    def validate_time_step_necessity(self) -> "SimulationConfig":
        """Enforces that time_step is provided if the tau_leaping solver is selected."""
        if self.solver == "tau_leaping" and self.time_step is None:
            raise ValueError(
                "ERR_CFG_002: Parameter 'time_step' is strictly required when using the 'tau_leaping' solver."
            )
        return self


# Node Directory Schema
# -----------------------
class FreeNodeConfig(BaseModel):
    """Configuration for a stochastically fluctuating free node."""

    name: str = Field(..., description="Unique name identifier of the node.")
    initial_charge: int = Field(
        default=0,
        description="Initial excess charge on the node (in units of qe).",
    )


class SinusoidalSpecs(BaseModel):
    """Specification parameters for an AC sinusoidal voltage source."""

    offset: float = Field(default=0.0, description="DC offset in Volts.")
    amplitude: float = Field(..., gt=0.0, description="Peak amplitude in Volts.")
    frequency: float = Field(..., gt=0.0, description="Frequency in Hertz.")
    phase: float = Field(default=0.0, description="Phase offset in Radians.")


class RegulatedNodeConfig(BaseModel):
    """
    Configuration for an external regulated voltage source or ground reference.
    """

    name: str = Field(..., description="Unique name identifier of the regulated node.")
    type: Literal["ground", "constant", "sinusoidal"] = Field(
        ..., description="The voltage behavior category."
    )
    value: float | None = Field(
        default=None,
        description="Static DC potential in Volts. Required if type is 'constant'.",
    )
    specs: SinusoidalSpecs | None = Field(
        default=None,
        description="AC parameters. Required if type is 'sinusoidal'.",
    )

    @model_validator(mode="after")
    def validate_type_requirements(self) -> "RegulatedNodeConfig":
        """Enforces field constraints based on the regulated node type."""
        if self.type == "ground":
            if self.value is not None or self.specs is not None:
                raise ValueError(
                    f"ERR_CFG_003: Regulated node '{self.name}' of type 'ground' cannot accept custom values or specs."
                )
        elif self.type == "constant":
            if self.value is None:
                raise ValueError(
                    f"ERR_CFG_002: Constant source '{self.name}' requires a 'value' field."
                )
            if self.specs is not None:
                raise ValueError(
                    f"ERR_CFG_003: Constant source '{self.name}' cannot accept 'specs' fields."
                )
        elif self.type == "sinusoidal":
            if self.specs is None:
                raise ValueError(
                    f" ERR_CFG_002: Sinusoidal source '{self.name}' requires a 'specs' configuration block."
                )
            if self.value is not None:
                raise ValueError(
                    f"ERR_CFG_003: Sinusoidal source '{self.name}' cannot accept a static 'value' field."
                )
        return self


class NodeDirectory(BaseModel):
    """
    Top-level catalog verifying the strict separation of free and regulated node spaces.
    """

    free: list[FreeNodeConfig] = Field(
        default_factory=list, description="List of internal free nodes."
    )
    regulated: list[RegulatedNodeConfig] = Field(
        default_factory=list,
        description="List of external regulated voltage/ground reference nodes.",
    )

    @model_validator(mode="after")
    def validate_no_name_collisions(self) -> "NodeDirectory":
        """Ensures that no node name is duplicated across free and regulated sets."""
        free_names = {node.name for node in self.free}
        regulated_names = {node.name for node in self.regulated}

        intersection = free_names.intersection(regulated_names)
        if intersection:
            raise ValueError(
                f"ERR_NET_103: Name collision detected! The following node name(s) are defined in "
                f"both free and regulated directories: {intersection}"
            )
        return self


# Component Schema
# -----------------------

# 1. Component Specs Models


class CapacitorSpecs(BaseModel):
    """Specific parameters required for basic electrostatic capacitors."""

    capacitance: float = Field(
        ..., gt=0.0, description="Capacitance value in Farads (F)."
    )


class TunnelJunctionSpecs(BaseModel):
    """Specific parameters required for discrete tunnel barriers."""

    resistance: float = Field(
        ..., gt=0.0, description="Tunneling resistance value in Ohms (Ω)."
    )


class MOSFETSpecs(BaseModel):
    """Physical parameters characterizing the subthreshold behavior of MOSFETs."""

    I0: float = Field(
        ..., gt=0.0, description="Saturation current coefficient in Amperes (A)."
    )
    VT: float = Field(..., description="Threshold voltage parameter in Volts (V).")
    n: float = Field(
        ..., gt=0.0, description="Subthreshold swing non-ideality coefficient."
    )


class DiodeSpecs(BaseModel):
    """Physical parameters characterizing a classical Shockley diode barrier."""

    I0: float = Field(
        ..., gt=0.0, description="Reverse saturation current in Amperes (A)."
    )
    n: float = Field(
        ..., gt=0.0, description="Diode ideality factor (typically 1.0 to 2.0)."
    )


# 2. Terminal Mapping Models


class MOSFETTerminals(BaseModel):
    """Strict name mappings for transistor channel and gate control conductors."""

    drain: str = Field(..., description="Name of the drain node connection.")
    gate: str = Field(..., description="Name of the gate node connection.")
    source: str = Field(..., description="Name of the source node connection.")
    bulk: str = Field(..., description="Name of the body/bulk substrate connection.")


# 3. Master Component Packaging Model


class ComponentConfig(BaseModel):
    """
    Unified component entry wrapper establishing connection mappings and specs verification.
    """

    type: Literal[
        "capacitor", "tunnel_junction", "n_channel_mosfet", "p_channel_mosfet", "diode"
    ] = Field(..., description="The physical structural category of the component.")
    name: str = Field(
        ..., description="Unique label identifying this component instance."
    )

    # Capacitors, Tunnel Junctions, and Diodes use simple 2-terminal listing arrays.
    # MOSFETs map using an explicit structural terminal block.
    terminals: list[str] | MOSFETTerminals = Field(
        ..., description="Node boundaries linking this device into the circuit netlist."
    )

    specs: dict = Field(
        ...,
        description="Raw dictionary containing parameter blocks mapped by device type.",
    )

    @model_validator(mode="after")
    def validate_component_matching(self) -> "ComponentConfig":
        """
        Enforces structural terminal shapes and parses/re-validates specs
        dictionaries against the designated typed Pydantic models.
        """
        c_type = self.type

        # --- Handle Two-Terminal Configurations ---
        if c_type in ["capacitor", "tunnel_junction", "diode"]:
            if not isinstance(self.terminals, list) or len(self.terminals) != 2:
                raise ValueError(
                    f"ERR_CFG_003: Component '{self.name}' [type={c_type}] must provide a list of exactly 2 terminals."
                )

            # Wrap specs initialization in try/except to prepend diagnostic codes
            try:
                if c_type == "capacitor":
                    self.specs = CapacitorSpecs(**self.specs).model_dump()
                elif c_type == "tunnel_junction":
                    self.specs = TunnelJunctionSpecs(**self.specs).model_dump()
                elif c_type == "diode":
                    self.specs = DiodeSpecs(**self.specs).model_dump()
            except ValidationError as e:
                # Catch internal constraints (like gt=0.0) and map to our system code
                raise ValueError(
                    f"ERR_CFG_001: Invalid spec parameters for component '{self.name}' [type={c_type}]. Details: {e}"
                ) from e

        # --- Handle Four-Terminal Configurations (MOSFETs) ---
        elif c_type in ["n_channel_mosfet", "p_channel_mosfet"]:
            if not isinstance(self.terminals, MOSFETTerminals):
                raise ValueError(
                    f"ERR_CFG_003: Component '{self.name}' [type={c_type}] requires a structured terminal mapping block."
                )

            try:
                parsed_mos = MOSFETSpecs(**self.specs)
                self.specs = parsed_mos.model_dump()
            except ValidationError as e:
                raise ValueError(
                    f"ERR_CFG_001: Invalid spec parameters for MOSFET '{self.name}'. Details: {e}"
                ) from e

            if c_type == "n_channel_mosfet" and parsed_mos.VT < 0.0:
                raise ValueError(
                    f"ERR_CFG_004: nMOSFET '{self.name}' has an unphysical negative threshold voltage (VT={parsed_mos.VT})."
                )
            if c_type == "p_channel_mosfet" and parsed_mos.VT > 0.0:
                raise ValueError(
                    f"ERR_CFG_004: pMOSFET '{self.name}' has an unphysical positive threshold voltage (VT={parsed_mos.VT})."
                )

        return self
