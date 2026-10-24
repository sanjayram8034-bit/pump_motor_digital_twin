"""
Dynamically Coupled Simulation Engine for Industrial Pump & Motor Digital Twin.

Derives deterministic, physically grounded coupled telemetry for testing and simulation.

[SIMULATION DISCLAIMER & MODEL ASSUMPTIONS]
The readings and equations in this module represent idealized numerical simulation
models designed for demonstration, algorithm validation, and fault-injection testing.
They MUST NOT be cited or interpreted as certified physical measurements, empirical field
sensor data, or validated engineering design calculations.

Coupled Physics Assumptions:
1. Direct Drive Coupling:
   - Motor and centrifugal pump shafts are directly coupled (rigid or flexible element).
   - Speed identity: RPM_pump == RPM_motor.
2. Centrifugal Pump Affinity Laws (Simplified):
   - Discharge flow rate scales linearly with shaft speed: Q / Q_rated ≈ (N / N_rated).
   - Differential pressure head scales quadratically with speed: P / P_rated ≈ (N / N_rated)^2.
3. Vibration Structural Transmission:
   - Baseplate and shaft couplings transmit mechanical vibration between machines.
   - Modeled via a configurable transmissibility factor (transmission_ratio ≈ 0.08 - 0.12).
   - This is an empirical engineering approximation, not a universal physical constant.
4. Hydraulic Load to Motor Current Coupling:
   - Pump hydraulic power P_hyd = rho * g * Q * H.
   - When fluid prime is lost (dry run / severe cavitation), hydraulic load vanishes.
   - Motor current drops from rated load (~14.2 A) down to induction motor no-load/magnetizing
     current (~5.5 - 6.5 A).
5. Thermal Time-Constant & Decoupling:
   - Typical 15 kW motor stator core has a thermal time constant of 15-30 minutes.
   - During a pump dry-run event, motor electrical losses (I^2 * R) decrease; hence motor
     temperature remains near normal baseline and does NOT spike instantaneously.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from backend.twin.state import TelemetryPayload, MotorTelemetry, PumpTelemetry


@dataclass(frozen=True)
class SimulationConfig:
    """Configurable baseline parameters and coupling coefficients."""
    # Baseline nominal operating point (Normal BEP operation)
    rated_rpm: float = 1485.0
    rated_flow_rate: float = 88.5        # m³/h
    rated_pressure: float = 4.8          # Bar
    rated_motor_current: float = 14.2    # Amperes
    rated_pump_current: float = 11.8     # Amperes (apportioned load current)
    rated_motor_temp: float = 56.4       # °C
    rated_pump_temp: float = 46.2        # °C
    baseline_motor_vib: float = 1.6      # mm/s RMS
    baseline_pump_vib: float = 1.3       # mm/s RMS

    # Idle / Stopped conditions
    ambient_temp: float = 26.0           # °C
    idle_vibration: float = 0.02         # mm/s

    # Coupling transmission assumptions
    vibration_transmission_ratio: float = 0.08  # Configurable structural vibration transmission ratio
    motor_no_load_current: float = 5.8          # Amperes (motor current when pump is dry/unloaded)


DEFAULT_SIM_CONFIG = SimulationConfig()


def generate_coupled_telemetry(
    scenario: str,
    config: Optional[SimulationConfig] = None
) -> TelemetryPayload:
    """
    Generate deterministic, dynamically coupled telemetry for the requested scenario.

    Supported Scenarios:
    - 'normal': Resets both motor and pump to nominal rated baseline.
    - 'idle': Stopped system (0 RPM, 0 current, 0 flow, ambient conditions).
    - 'motor_warning': Motor bearing/thermal stress with coupled pump speed derating,
                       Affinity-law pressure/flow reduction, and transmitted vibration.
    - 'pump_critical': Pump suction loss / dry-run with collapsed pressure and flow,
                       severely reduced motor electrical current (unloaded), and
                       transmitted cavitation shockwaves to motor vibration.
    """
    cfg = config or DEFAULT_SIM_CONFIG
    s = scenario.strip().lower()

    if s == "idle":
        # 1. Stopped / Idle State
        motor = MotorTelemetry(
            temperature=cfg.ambient_temp,
            vibration=cfg.idle_vibration,
            current=0.0,
            rpm=0.0
        )
        pump = PumpTelemetry(
            temperature=cfg.ambient_temp,
            vibration=cfg.idle_vibration,
            current=0.0,
            rpm=0.0,
            pressure=0.0,
            flow_rate=0.0
        )
        return TelemetryPayload(motor=motor, pump=pump)

    elif s == "motor_warning":
        # 2. Motor Vibration & Thermal Stress (Primary Motor Fault)
        # Primary Motor Fault Assumptions:
        # - Winding / bearing thermal rise: 78.5 °C
        # - Elevated mechanical vibration (ISO Zone C): 5.2 mm/s
        # - Increased electrical current draw due to mechanical drag: 23.2 A
        # - Reduced shaft speed due to slip increase: 1445.0 RPM
        motor_temp = 78.5
        motor_vib = 5.2
        motor_curr = 23.2
        motor_rpm = 1445.0

        # Coupled Pump Response (Derived Physics):
        # a) Direct coupling: pump shaft speed matches motor speed exactly
        pump_rpm = motor_rpm

        # b) Simplified Pump Affinity Laws: Q ∝ N, P ∝ N^2
        speed_ratio = pump_rpm / cfg.rated_rpm
        derived_flow = round(cfg.rated_flow_rate * speed_ratio, 1)          # ~86.1 m³/h
        derived_pressure = round(cfg.rated_pressure * (speed_ratio ** 2), 2) # ~4.55 Bar

        # c) Vibration structural coupling (transmissibility assumption)
        excess_motor_vib = max(0.0, motor_vib - cfg.baseline_motor_vib)
        derived_pump_vib = round(cfg.baseline_pump_vib + (cfg.vibration_transmission_ratio * excess_motor_vib), 2)

        # d) Pump thermal and electrical load
        # Mild conductive heat transfer down shaft and slight hydraulic power adjustment
        pump_temp = round(cfg.rated_pump_temp + 2.5, 1)                      # ~48.7 °C
        pump_curr = round(cfg.rated_pump_current * (speed_ratio ** 3), 1)    # ~10.9 A

        motor = MotorTelemetry(
            temperature=motor_temp,
            vibration=motor_vib,
            current=motor_curr,
            rpm=motor_rpm
        )
        pump = PumpTelemetry(
            temperature=pump_temp,
            vibration=derived_pump_vib,
            current=pump_curr,
            rpm=pump_rpm,
            pressure=derived_pressure,
            flow_rate=derived_flow
        )
        return TelemetryPayload(motor=motor, pump=pump)

    elif s == "pump_critical":
        # 3. Pump Suction Loss / Dry-Run (Primary Pump Fault)
        # Primary Pump Fault Assumptions:
        # - Fluid evacuation: delivery flow collapses to near zero (5.0 m³/h)
        # - Discharge pressure collapses (0.30 Bar)
        # - Dry mechanical seal and bearing friction heating: 92.0 °C
        # - Severe cavitation / turbulent gas pocket vibration: 9.5 mm/s
        # - Pump effective hydraulic current: collapses to 2.5 A
        pump_flow = 5.0
        pump_press = 0.30
        pump_temp = 92.0
        pump_vib = 9.5
        pump_curr = 2.5

        # Coupled Motor Response (Derived Physics):
        # a) Loss of hydraulic load:
        #    Because water fluid mass is removed, motor does no hydraulic work!
        #    Motor current collapses from rated 14.2 A down to no-load magnetizing current (~6.0 A)
        motor_curr = round(cfg.motor_no_load_current + 0.3, 1)               # ~6.1 A

        # b) Slip reduction:
        #    With rotor unloaded, induction motor speed rises towards synchronous speed (1500 RPM)
        motor_rpm = 1493.0
        pump_rpm = motor_rpm  # Direct shaft coupling

        # c) Motor Thermal Behavior:
        #    Because motor electrical I^2*R losses have dropped and thermal time-constant is large,
        #    motor temperature does NOT spike; remains near normal running baseline
        motor_temp = round(cfg.rated_motor_temp - 0.4, 1)                    # ~56.0 °C

        # d) Transmitted Cavitation Shockwaves:
        #    Violent cavitation vibrations transmit across shaft coupling and baseplate to motor
        excess_pump_vib = max(0.0, pump_vib - cfg.baseline_pump_vib)
        derived_motor_vib = round(cfg.baseline_motor_vib + (cfg.vibration_transmission_ratio * excess_pump_vib), 2) # ~2.26 mm/s

        motor = MotorTelemetry(
            temperature=motor_temp,
            vibration=derived_motor_vib,
            current=motor_curr,
            rpm=motor_rpm
        )
        pump = PumpTelemetry(
            temperature=pump_temp,
            vibration=pump_vib,
            current=pump_curr,
            rpm=pump_rpm,
            pressure=pump_press,
            flow_rate=pump_flow
        )
        return TelemetryPayload(motor=motor, pump=pump)

    else:
        # 4. Default: Normal Baseline Operation
        motor = MotorTelemetry(
            temperature=cfg.rated_motor_temp,
            vibration=cfg.baseline_motor_vib,
            current=cfg.rated_motor_current,
            rpm=cfg.rated_rpm
        )
        pump = PumpTelemetry(
            temperature=cfg.rated_pump_temp,
            vibration=cfg.baseline_pump_vib,
            current=cfg.rated_pump_current,
            rpm=cfg.rated_rpm,
            pressure=cfg.rated_pressure,
            flow_rate=cfg.rated_flow_rate
        )
        return TelemetryPayload(motor=motor, pump=pump)
