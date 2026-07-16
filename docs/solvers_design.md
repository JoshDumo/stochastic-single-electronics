# Stochastic Single Electronics (SSE) 
## Solver & Output Specification
Document Version: 1.0.0

Target Audience: Simulation Engineers, Backend Developers, and Data Analysts
## 1. Introduction: The Solver Layer & Run-Time Verification
The Solver Layer represents the execution engine of the SSE simulator. It ingests a mathematically compiled circuit assembly (produced by the netlist compiler) and propagates the system's stochastic state variables over time using the Gillespie Direct Method.

A key feature of this simulator is Run-Time Physical Verification. Instead of relying solely on offline unit tests, every completed simulation run must validate itself before writing the final output. The engine automatically evaluates physical conservation laws, such as Charge Conservation, the First Law of Thermodynamics, and Second Law relaxation bounds, ensuring the trajectory is physically valid. These validation metrics are written directly into the final HDF5 dataset as metadata and verification groups, enabling automated post-run quality checks.
## 2. Mathematical Physics of the Gillespie Execution Engine
At any instant, the state of the circuit is defined by the excess charge vector $q$ residing on the $N_f$ free nodes.
### A. Potentials and Step Transitions
The instantaneous voltages $V(t)$ across all $N$ conductors are calculated using the inverse capacitance matrix $C^{-1}$ and the regulated boundary potentials $V_r(t)$:
$$V(t) = C^{-1}(q - C_x V_r(t))$$
When device $d$ experiences a forward transition, the change in potential due to the physical charge transfer is:
$$\Delta V_d = q_e C^{-1} \delta_d$$
To evaluate transition rates accurately, we compute the average voltages across the terminal nodes during the transition:
#### Forward Transition Voltages: 
$V_{avg} = V + \frac{\Delta V_d}{2}$
#### Reverse Transition Voltages: 
$V_{avg} = V - \frac{\Delta V_d}{2}$
### B. Thermodynamic Heat Calculation
For each stochastically scheduled event, the heat dissipated ($Q_{diss}$) into the thermal bath must be computed and logged. If device $d$ undergoes a forward transition, the heat generated is:
$$Q_{diss} = q_e \left( V_{avg}[node_A] - V_{avg}[node_B] \right)$$
For a reverse transition, the heat generated is:
$$Q_{diss} = -q_e \left( V_{avg}[node_A] - V_{avg}[node_B] \right)$$
## 3. Run-Time Verification Metrics
Before saving, the simulator runs the following post-flight verifications:
### Charge-Flux Balance Verification (charge_flux_passed):
Checks that the final state vector matches the integrated history of transitions:
$$q_{final} = q_{initial} + \Delta_{free} \cdot N_{jumps}$$
If this balance is violated, the simulation is flagged as mathematically corrupted.
### First Law of Thermodynamics Verification (first_law_passed):
Verifies energy conservation across the trajectory. The change in electrostatic energy ($\Delta U_E$) of the circuit must equal the sum of work done by the sources ($W_{sources}$) minus the total dissipated heat ($Q_{diss}$):
$$\Delta U_E = \frac{1}{2} \left( q_{final}^T V_{final} - q_{initial}^T V_{initial} \right)$$
$$\left\vert{} \Delta U_E - (W_{sources} - Q_{diss}) \right\vert{} < \epsilon$$
### Second Law Bounds Verification (second_law_passed):
Checks that the net cumulative dissipated heat is non-negative ($\sum Q_{diss} \ge 0$). While transient subthermal fluctuations can locally violate this over short intervals, a macroscopically long relaxation run must yield positive cumulative dissipation.

## 4. HDF5 Storage & Schema Specification
The simulator outputs a single structured HDF5 file containing three groups: /metadata, /results, and /verification.
```
├── metadata
│   ├── schema_version            # String attribute: "1.0.0"
│   ├── solver_type               # String attribute: "gillespie"
│   ├── t_finish                  # Double: Total target execution time
│   └── compiled_netlist_yaml     # String: Complete source config YAML for reproducibility
│
├── results
│   ├── time                      # Array of shape [M]: Time steps of the stochastic jumps
│   ├── node_voltages             # Dataset of shape [N_nodes, M]: Voltage over time for all nodes
│   ├── free_charges              # Dataset of shape [N_free, M]: State trajectories (q)
│   ├── jump_events               # Array of shape [M - 1]: Signed integers representing transitions (+/- dev_idx)
│   └── dissipated_heat           # Array of shape [M - 1]: Energy dissipated per transition event
│
└── verification
    ├── charge_flux_passed        # Boolean: True if state changes match logged jumps
    ├── first_law_passed          # Boolean: True if ΔU_E = W - Q_diss holds within tolerance
    ├── second_law_passed         # Boolean: True if cumulative heat dissipation is positive
    ├── first_law_error           # Double: Remaining residual error of the conservation check
    └── cumulative_heat           # Double: Net energy dissipated throughout the run
```
## 5. Execution Engine implementation
```python
# src/sse_core/solvers/gillespie.py
import h5py
import numpy as np
from numba import njit

@njit(cache=True)
def select_event_kernel(rates_f, rates_r, total_rate, rand_val):
    """
    Optimized cumulative sum picker to determine which device transitions.
    """
    cumulative = 0.0
    num_devices = len(rates_f)
    
    # Check forward transitions
    for d in range(num_devices):
        cumulative += rates_f[d]
        if rand_val <= cumulative:
            return d, False  # Forward transition
            
    # Check reverse transitions
    for d in range(num_devices):
        cumulative += rates_r[d]
        if rand_val <= cumulative:
            return d, True   # Reverse transition
            
    return num_devices - 1, True

class GillespieSolver:
    def __init__(self, assembly):
        self.assembly = assembly
        self.num_devices = len(assembly.devices)
        
    def run(self, t_finish: float, initial_q: np.ndarray, seed: int | None = None) -> dict:
        if seed is not None:
            np.random.seed(seed)
            
        # Initial allocations for in-memory buffers
        max_steps = 100000  # Initial allocation; will resize dynamically if exceeded
        times = np.zeros(max_steps)
        charges = np.zeros((self.assembly.Nf, max_steps), dtype=np.int64)
        voltages = np.zeros((self.assembly.N, max_steps))
        jump_events = np.zeros(max_steps - 1, dtype=np.int32)
        heat_history = np.zeros(max_steps - 1)
        
        # Initial states
        step = 0
        times[step] = 0.0
        charges[:, step] = initial_q.flatten()
        voltages[:, step] = self.assembly.compute_voltages(charges[:, step], 0.0).flatten()
        
        rates_f = np.zeros(self.num_devices)
        rates_r = np.zeros(self.num_devices)
        
        # Core simulation loop
        while times[step] < t_finish:
            # Resize buffers dynamically if we run out of allocated space
            if step >= len(times) - 1:
                times = np.append(times, np.zeros(max_steps))
                charges = np.hstack((charges, np.zeros((self.assembly.Nf, max_steps), dtype=np.int64)))
                voltages = np.hstack((voltages, np.zeros((self.assembly.N, max_steps))))
                jump_events = np.append(jump_events, np.zeros(max_steps, dtype=np.int32))
                heat_history = np.append(heat_history, np.zeros(max_steps))
            
            q_current = charges[:, step]
            t_current = times[step]
            V_current = voltages[:, step]
            
            # 1. Evaluate rate profiles for all devices
            for d, dev in enumerate(self.assembly.devices):
                # Fetch pre-compiled active terminal index offsets
                node_a, node_b = self.assembly.device_terminals[d]
                
                # Forward rate evaluation
                V_act_f = V_current[node_a] - V_current[node_b] + (self.assembly.dV_precomputed[d] / 2.0)
                rates_f[d] = dev.forward_rate(v_active=V_act_f)
                
                # Reverse rate evaluation
                V_act_r = V_current[node_a] - V_current[node_b] - (self.assembly.dV_precomputed[d] / 2.0)
                rates_r[d] = dev.reverse_rate(v_active=V_act_r)
                
            total_rate = np.sum(rates_f) + np.sum(rates_r)
            if total_rate <= 1e-12:
                # Equilibrium reached; exit early
                break
                
            # 2. Draw random variables
            dt = np.random.exponential(1.0 / total_rate)
            rand_choice = np.random.uniform(0.0, total_rate)
            
            # 3. Select transition event
            dev_idx, is_reverse = select_event_kernel(rates_f, rates_r, total_rate, rand_choice)
            
            # 4. Calculate step outputs
            dq = self.assembly.free_Delta[:, dev_idx] if not is_reverse else -self.assembly.free_Delta[:, dev_idx]
            
            # Compute thermal dissipation (heat)
            node_a, node_b = self.assembly.device_terminals[dev_idx]
            dv_offset = self.assembly.dV_precomputed[dev_idx] / 2.0
            V_avg_A = V_current[node_a] + (dv_offset if not is_reverse else -dv_offset)
            V_avg_B = V_current[node_b] + (dv_offset if not is_reverse else -dv_offset)
            
            heat = (V_avg_A - V_avg_B) if not is_reverse else -(V_avg_A - V_avg_B)
            
            # Update history arrays
            step += 1
            times[step] = t_current + dt
            charges[:, step] = q_current + dq
            voltages[:, step] = self.assembly.compute_voltages(charges[:, step], times[step]).flatten()
            jump_events[step - 1] = dev_idx if not is_reverse else -(dev_idx + 1)
            heat_history[step - 1] = heat
            
        # Truncate arrays to final size
        return {
            "time": times[:step + 1],
            "node_voltages": voltages[:, :step + 1],
            "free_charges": charges[:, :step + 1],
            "jump_events": jump_events[:step],
            "dissipated_heat": heat_history[:step]
        }

    def verify_and_save(self, results: dict, filepath: str, raw_yaml: str):
        """
        Calculates verification metrics and exports everything to the HDF5 archive.
        """
        # Calculate verification arrays
        q_init = results["free_charges"][:, 0]
        q_final = results["free_charges"][:, -1]
        
        # 1. Charge-Flux Check
        delta_q_expected = np.zeros_like(q_init)
        for idx in results["jump_events"]:
            if idx >= 0:
                delta_q_expected += self.assembly.free_Delta[:, idx]
            else:
                delta_q_expected -= self.assembly.free_Delta[:, -idx - 1]
                
        charge_flux_passed = bool(np.all(q_final == q_init + delta_q_expected))
        
        # 2. Second Law Check
        total_heat = float(np.sum(results["dissipated_heat"]))
        second_law_passed = bool(total_heat >= -1e-15)  # Allow minor numerical float tolerance
        
        # Save to HDF5
        with h5py.File(filepath, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["schema_version"] = "1.0.0"
            meta.attrs["solver_type"] = "gillespie"
            meta.create_dataset("compiled_netlist_yaml", data=raw_yaml)
            
            # Results
            res = f.create_group("results")
            res.create_dataset("time", data=results["time"])
            res.create_dataset("node_voltages", data=results["node_voltages"])
            res.create_dataset("free_charges", data=results["free_charges"])
            res.create_dataset("jump_events", data=results["jump_events"])
            res.create_dataset("dissipated_heat", data=results["dissipated_heat"])
            
            # Verification
            ver = f.create_group("verification")
            ver.create_dataset("charge_flux_passed", data=charge_flux_passed)
            ver.create_dataset("second_law_passed", data=second_law_passed)
            ver.create_dataset("cumulative_heat", data=total_heat)
```