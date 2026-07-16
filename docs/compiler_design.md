# Stochastic Single Electronics (SSE) 
##  Simulation Netlist and Compiler Specification
Document Version: 1.0.0

Target Audience: Circuit Designers, Hardware Researchers, and System Integrators
## 1. Introduction: 
Working with the SSE SimulatorThe Stochastic Single Electronics (SSE) Simulator is a high-performance backend designed to model physical charge fluctuations and thermodynamic behaviors in mesoscopic circuits. Unlike traditional SPICE solvers that rely on continuous differential equations, the SSE simulator tracks the discrete movements of individual electrons (shot noise) and thermal fluctuations ($k_B T$) using stochastic execution methods (e.g., the Gillespie Direct Method or Tau-Leaping).

To run a simulation, the circuit configuration, devices, and solver parameters are written in a structured YAML netlist file. This file is parsed, linted, and compiled by the SSE Static Compiler, which transforms the textual description into mathematical matrices before handing execution over to the numerical solvers.

### Quickstart: Compiling a Netlist in Python
```python
from sse_compiler import SSECompiler

# 1. Initialize the Compiler
compiler = SSECompiler()

# 2. Load and Compile your Netlist file
assembly, diagnostics = compiler.compile_file("my_circuit.yaml")

# 3. Check for Errors/Warnings
if diagnostics.has_errors():
    print("Compilation failed with the following errors:")
    diagnostics.print_report()
else:
    print("Compilation successful!")
    # Proceed to pass the compiled assembly directly to the simulation engine
    # solver = SSESolver(assembly)
    # solver.run()
```
## 2. YAML Input Configuration Specification (User Guide)
The YAML file is divided into four primary top-level blocks:
* schema_version: Standardized version identifier.
* simulation: Numerical execution settings and physical global parameters.
* nodes: Declarations of all circuit nodes partitioned by physical boundaries.
* components: Structural connectivity map and device parameters.

#### Complete Reference Template
```YAML
schema_version: "1.0.0"

metadata:
  name: "CMOS Inverter Benchmark"
  description: "Standard complementary p-bit inverter operating around thermal voltage limit"

simulation:
  solver: "gillespie"         # Options: [gillespie, tau_leaping]
  t_finish: 1.0e-6            # Total simulation time in seconds
  time_step: 1.0e-9           # Only required if solver is "tau_leaping"
  seed: 123456                # Optional integer for reproducible stochastic runs
  v_th: 0.0259                # Thermal voltage (V_th) in Volts. Defaults to 0.0259 (approx. 300K)

nodes:
  # ---------------------------------------------------------------------------
  # FREE NODES: Conductors whose excess charge states fluctuate stochastically.
  # These represent the internal system state variables.
  # ---------------------------------------------------------------------------
  free:
    - name: "out"
      initial_charge: 0       # Initial excess charge (in units of elementary charge qe)
    - name: "internal"
      initial_charge: 0

  # ---------------------------------------------------------------------------
  # REGULATED NODES: Voltage sources or grounds that act as boundary conditions.
  # Their potentials are controlled and do not fluctuate based on single electron transfers.
  # ---------------------------------------------------------------------------
  regulated:
    - name: "gnd"
      type: "ground"          # Hard-coded to 0.0 V. No auxiliary specs allowed.

    - name: "vdd"
      type: "constant"        # Static DC voltage
      value: 1.0              # Volts

    - name: "v_in"
      type: "sinusoidal"      # Time-varying AC signal
      specs:
        offset: 0.0           # Volts
        amplitude: 0.5        # Volts (peak)
        frequency: 1.0e6      # Hertz (Hz)
        phase: 0.0            # Radians

components:
  # ---------------------------------------------------------------------------
  # 1. CAPACITORS: Elements establishing electrostatic coupling between nodes.
  # ---------------------------------------------------------------------------
  - type: "capacitor"
    name: "C_load"
    terminals: ["out", "gnd"]
    specs:
      capacitance: 2.0e-15    # Farads (F)

  # ---------------------------------------------------------------------------
  # 2. TUNNEL JUNCTIONS: Quantized charge passage barriers.
  # ---------------------------------------------------------------------------
  - type: "tunnel_junction"
    name: "TJ_1"
    terminals: ["out", "gnd"]
    specs:
      resistance: 1.0e5       # Ohms (Ω)

  # ---------------------------------------------------------------------------
  # 3. N-CHANNEL MOSFET (nMOS): Gate-controlled tunnel junction.
  # ---------------------------------------------------------------------------
  - type: "n_channel_mosfet"
    name: "M_pulldown"
    terminals:
      drain: "out"
      gate: "v_in"
      source: "gnd"
      bulk: "gnd"
    specs:
      I0: 1.0e-6              # Saturation current multiplier (Amps)
      VT: 0.4                 # Positive threshold voltage (Volts)
      n: 1.1                  # Subthreshold swing factor

  # ---------------------------------------------------------------------------
  # 4. P-CHANNEL MOSFET (pMOS): Complementary gate-controlled tunnel junction.
  # ---------------------------------------------------------------------------
  - type: "p_channel_mosfet"
    name: "M_pullup"
    terminals:
      drain: "out"
      gate: "v_in"
      source: "vdd"
      bulk: "vdd"
    specs:
      I0: 1.0e-6              # Saturation current multiplier (Amps)
      VT: -0.4                # Negative threshold voltage for pMOS (Volts)
      n: 1.1                  # Subthreshold swing factor
```

## 3. Compiler Diagnostic Catalog (Errors and Warnings)
The compiler checks configuration parsing, topology validation, and mathematical sanity. If issues are found, they are reported using a formal classification schema.
### Diagnostic Codes 
| Diagnostic Code | Classification| Phase | Meaning |
|---------|---------|------|-----------------------|
|ERR_CFG_001| Fatal Error | Schema Parsing | Negative physical parameter value (e.g. negative capacitance or resistance) where only positive values are physically real.|
| ERR_CFG_002| Fatal Error | Schema Parsing | Missing required parameter field within a component's specs dictionary.|
| ERR_CFG_003 | Fatal Error | Schema Parsing | Illegal parameter configuration (e.g., providing custom voltage specs to a node of type "ground").|
| ERR_NET_101 | Fatal Error | Topology Check | Dangling terminal path. A component references a terminal name not declared in the nodes directory. | ERR_NET_102 | Fatal Error | Topology Check | Floating node detected. A node is declared but connects to fewer than two component terminals. | 
| ERR_NET_103 | Fatal Error | Topology Check | Name collision. A node name is duplicated inside the nodes. | 
| ERR_NET_104 | Fatal Error | Topology Check | No ground reference. The circuit does not declare at least one node of type "ground". | 
| ERR_NET_105 | Warning | Topology Check | Suspicious MOSFET polarity. For example, a p_channel_mosfet source node is tied directly to a lower static DC potential than its drain terminal. | 
| ERR_MATH_201 | Fatal Error | Matrix Assembly | Singular Maxwell capacitance matrix. The matrix cannot be inverted, meaning part of the circuit is electrostatically floating or isolated.|

## 4. Architectural Design of the Compiler & Linter
The compiling process is organized as a pipeline:

```
[YAML String]
     │
     ▼
[Phase 1: AST Parser & Schema Validator] 
     │  - Parses YAML syntax into typed models
     │  - Validates types, bounds, and specs
     │  - Catch: ERR_CFG_*
     ▼
[Phase 2: Topological Graph Analyzer]
     │  - Builds a graph where Nodes = Vertices, Components = Edges
     │  - Trace connectivity degrees & validate terminal bounds
     │  - Catch: ERR_NET_*
     ▼
[Phase 3: Electrostatics Assembler]
     │  - Builds lumped/Maxwell capacitance matrices (C, Cx)
     │  - Computes inverse capacitance (invC) and delta transitions (Delta)
     │  - Catch: ERR_MATH_*
     ▼
[Compiled Assembly Object] -> (Ready for C++/Numba Solver)
```
## 5. Compiler Implementation Pseudocode
The following pseudocode details the architectural design and operations of the compiler pipeline.
### Data Structures & Diagnostics Container
```python
Pythonclass Diagnostic:
    code: String        # e.g., "ERR_NET_101"
    severity: String    # "ERROR" or "WARNING"
    message: String     # Human-readable context

class DiagnosticReport:
    diagnostics: List[Diagnostic]
    
    def add_error(code, message):
        self.diagnostics.append(Diagnostic(code, "ERROR", message))
        
    def add_warning(code, message):
        self.diagnostics.append(Diagnostic(code, "WARNING", message))
        
    def has_errors():
        return any(d.severity == "ERROR" for d in self.diagnostics)
```
### Compiler Pipeline Orchestration
```python
Pythonclass SSECompiler:
    def compile(self, yaml_string: String) -> (Assembly, DiagnosticReport):
        report = DiagnosticReport()
        
        # --- Phase 1: AST Parsing & Schema Parsing ---
        raw_ast = ParseYaml(yaml_string)  # Standard YAML Parser
        if raw_ast is Null:
            report.add_error("ERR_CFG_000", "Malformed YAML syntax")
            return Null, report
            
        config = self.validate_schema(raw_ast, report)
        if report.has_errors():
            return Null, report
            
        # --- Phase 2: Topological Graph Analysis ---
        self.validate_topology(config, report)
        if report.has_errors():
            return Null, report
            
        # --- Phase 3: Mathematical Matrix Assembly ---
        assembly = self.assemble_mathematics(config, report)
        if report.has_errors():
            return Null, report
            
        return assembly, report

    # -------------------------------------------------------------------------
    # PHASE 1: Schema Validation
    # -------------------------------------------------------------------------
    def validate_schema(self, ast, report) -> ConfigModel:
        # 1. Validate Simulation Block
        sim = ast.get("simulation")
        if sim.get("v_th") <= 0:
            report.add_error("ERR_CFG_001", "v_th must be strictly positive")
            
        # 2. Check Node Names for collisions
        free_names = [n.get("name") for n in ast.get("nodes", {}).get("free", [])]
        reg_names = [n.get("name") for n in ast.get("nodes", {}).get("regulated", [])]
        
        for name in free_names:
            if name in reg_names:
                report.add_error("ERR_NET_103", f"Collision: Node name '{name}' used as both free and regulated")
                
        # 3. Validate components and their distinct specs
        for comp in ast.get("components", []):
            name = comp.get("name")
            c_type = comp.get("type")
            specs = comp.get("specs", {})
            
            if c_type == "capacitor":
                if "capacitance" not in specs:
                    report.add_error("ERR_CFG_002", f"Capacitor '{name}' missing spec field: capacitance")
                elif specs.get("capacitance") <= 0:
                    report.add_error("ERR_CFG_001", f"Capacitor '{name}' has non-positive capacitance: {specs.get('capacitance')}")
                    
            elif c_type == "tunnel_junction":
                if "resistance" not in specs:
                    report.add_error("ERR_CFG_002", f"Tunnel Junction '{name}' missing spec field: resistance")
                elif specs.get("resistance") <= 0:
                    report.add_error("ERR_CFG_001", f"Tunnel Junction '{name}' has non-positive resistance: {specs.get('resistance')}")
                    
            elif c_type in ["n_channel_mosfet", "p_channel_mosfet"]:
                for param in ["I0", "VT", "n"]:
                    if param not in specs:
                        report.add_error("ERR_CFG_002", f"MOSFET '{name}' missing spec field: {param}")
                if specs.get("I0", 0) <= 0:
                    report.add_error("ERR_CFG_001", f"MOSFET '{name}' has non-positive scaling factor I0")
                if specs.get("n", 0) <= 0:
                    report.add_error("ERR_CFG_001", f"MOSFET '{name}' subthreshold swing factor 'n' must be positive")
                    
                # Specific threshold polarity validation
                vt = specs.get("VT", 0)
                if c_type == "n_channel_mosfet" and vt < 0:
                    report.add_warning("ERR_CFG_004", f"nMOSFET '{name}' usually requires a positive threshold voltage VT (Found: {vt})")
                elif c_type == "p_channel_mosfet" and vt > 0:
                    report.add_warning("ERR_CFG_004", f"pMOSFET '{name}' usually requires a negative threshold voltage VT (Found: {vt})")
        
        return ConfigModel(ast)

    # -------------------------------------------------------------------------
    # PHASE 2: Topological Validation
    # -------------------------------------------------------------------------
    def validate_topology(self, config, report):
        all_nodes = config.get_all_node_names() # includes ground and sources
        
        # Check ground presence
        gnd_nodes = [n for n in config.regulated_nodes if n.type == "ground"]
        if len(gnd_nodes) == 0:
            report.add_error("ERR_NET_104", "No ground reference node detected in 'nodes.regulated'")
            
        # Initialize connection degree counter
        connection_counts = {name: 0 for name in all_nodes}
        
        for comp in config.components:
            # Flatten terminals regardless of terminal dict style (lists or structured)
            terminals = comp.get_flat_terminals()
            for t in terminals:
                if t not in all_nodes:
                    report.add_error("ERR_NET_101", f"Component '{comp.name}' references undefined terminal node: '{t}'")
                else:
                    connection_counts[t] += 1
                    
        # Check for isolated/floating nodes (degree < 2)
        for node_name, count in connection_counts.items():
            if count < 2:
                report.add_error("ERR_NET_102", f"Isolated node detected: '{node_name}' has only {count} connection(s)")

    # -------------------------------------------------------------------------
    # PHASE 3: Mathematical Assembly
    # -------------------------------------------------------------------------
    def assemble_mathematics(self, config, report) -> Assembly:
        # Index Mapping
        # Generate stable, 0-indexed vectors for calculation routines
        free_nodes = config.get_free_node_names()
        reg_nodes = config.get_regulated_node_names()
        all_nodes = free_nodes + reg_nodes  # Order is highly important here
        
        N = len(all_nodes)
        Nf = len(free_nodes)
        Nr = len(reg_nodes)
        
        # Build index mapping helper dictionaries
        node_idx = {name: i for i, name in enumerate(all_nodes)}
        
        # 1. Build Maxwell Capacitance Matrices (C, Cx)
        lumped_C = Matrix.zeros(N, N)
        for comp in config.components:
            if comp.type == "capacitor":
                i = node_idx[comp.terminals[0]]
                j = node_idx[comp.terminals[1]]
                val = comp.specs.capacitance
                lumped_C[i, j] += val
                lumped_C[j, i] += val
                
        # Convert lumped to Maxwell capacitance matrix
        maxwell_C = Matrix.zeros(N, N)
        for r in range(N):
            for c in range(N):
                if r == c:
                    maxwell_C[r, c] = sum(lumped_C[r, :])
                else:
                    maxwell_C[r, c] = -lumped_C[r, c]
                    
        # Slice into blocks (free vs regulated partition)
        # C: free-to-free coupling, Cx: free-to-regulated coupling
        C_block = maxwell_C[0:Nf, 0:Nf]
        Cx_block = maxwell_C[0:Nf, Nf:N]
        
        # 2. Invert core capacitance matrix
        try:
            invC = C_block.invert()
        except MatrixSingularException:
            report.add_error("ERR_MATH_201", "The compiled free-node Maxwell Capacitance matrix is singular. Check for floating subsystems.")
            return Null
            
        # 3. Build Incident Matrix Delta for all devices
        # Non-capacitive active components (Tunnel Junctions, MOSFETs)
        devices = [c for c in config.components if c.type != "capacitor"]
        Ndev = len(devices)
        Delta = Matrix.zeros(N, Ndev)
        
        for k, dev in enumerate(devices):
            # Resolve physical terminal roles to source/drain connections
            terminals = dev.get_active_terminals() # returns [terminal_from, terminal_to]
            from_idx = node_idx[terminals[0]]
            to_idx = node_idx[terminals[1]]
            
            # An electron jump shifts charge: -1 qe from 'from_idx' node, +1 qe to 'to_idx' node
            Delta[from_idx, k] = -1
            Delta[to_idx, k] = 1
            
        # Split Delta for free nodes
        free_Delta = Delta[0:Nf, :]
        reg_Delta = Delta[Nf:N, :]
        
        # Return complete verified, statically assembled package
        return Assembly(
            Nf=Nf, Nr=Nr, N=N,
            C=C_block, Cx=Cx_block, invC=invC,
            Delta=Delta, free_Delta=free_Delta, reg_Delta=reg_Delta,
            devices=devices, node_mapping=node_idx
        )
```
## 6. Verification Test Cases for the Compiler Pipeline
These pytest test plans verify that both positive (successful compile) and negative (proper error detection) pathways behave exactly as designed.
```python
import pytest
from sse_compiler import SSECompiler

def test_compiler_success_on_basic_inverter():
    """Verify compile succeeds with zero diagnostics on a valid complementary CMOS topology."""
    valid_yaml = """
    schema_version: "1.0.0"
    simulation:
      solver: "gillespie"
      t_finish: 1.0e-6
      v_th: 0.0259
    nodes:
      free:
        - name: "out"
          initial_charge: 0
      regulated:
        - name: "gnd"
          type: "ground"
        - name: "vdd"
          type: "constant"
          value: 1.2
        - name: "v_in"
          type: "constant"
          value: 0.6
    components:
      - type: "capacitor"
        name: "CL"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1.0e-15}
      - type: "n_channel_mosfet"
        name: "M1"
        terminals: {drain: "out", gate: "v_in", source: "gnd", bulk: "gnd"}
        specs: {I0: 1e-6, VT: 0.35, n: 1.1}
    """
    compiler = SSECompiler()
    assembly, report = compiler.compile(valid_yaml)
    
    assert report.has_errors() is False
    assert assembly is not None
    assert assembly.Nf == 1
    assert assembly.Nr == 3

def test_compiler_detects_isolated_floating_node_err_net_102():
    """Verify floating node detector identifies isolated nodes."""
    invalid_yaml = """
    schema_version: "1.0.0"
    simulation: {solver: "gillespie", t_finish: 1e-6}
    nodes:
      free:
        - name: "isolated_node"  # <--- Connected to absolutely nothing!
          initial_charge: 0
        - name: "out"
          initial_charge: 0
      regulated:
        - name: "gnd"
          type: "ground"
    components:
      - type: "capacitor"
        name: "CL"
        terminals: ["out", "gnd"]
        specs: {capacitance: 1e-15}
    """
    compiler = SSECompiler()
    assembly, report = compiler.compile(invalid_yaml)
    
    assert report.has_errors() is True
    assert any(diag.code == "ERR_NET_102" for diag in report.diagnostics)
    assert "isolated_node" in report.diagnostics[0].message
    ```