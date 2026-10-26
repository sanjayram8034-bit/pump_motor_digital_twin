"""
Component-Level Diagnostic Engine for Industrial Pump & Motor Digital Twin.

Provides explainable, evidence-based component-level fault diagnosis by analyzing
real-time telemetry and twin state. Adheres to industrial condition monitoring
principles (ISO 10816 vibration guidelines, NEMA/IEC motor thermal standards,
and Hydraulic Institute pump standards).

DISCLAIMER & LIMITATIONS:
Thresholds and heuristics in this module are provisional engineering models
calibrated for algorithm demonstration and digital twin simulation. They must not
be interpreted as certified OEM, IEEE, NEMA, or ISO safety limits.
In real industrial facilities, diagnosis requires multi-axis high-frequency
accelerometers, envelope spectrum analysis (FFT), motor current signature
analysis (MCSA), oil tribology, and embedded RTD sensors.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field as PydanticField


# -----------------------------------------------------------------------------
# Component Identifiers (Mapped 1:1 with 3D Visual Twin Models)
# -----------------------------------------------------------------------------
# Motor 3D Components:
COMPONENT_MOTOR_HOUSING = "motor_housing"
COMPONENT_MOTOR_DE_BEARING = "de_bearing"
COMPONENT_MOTOR_NDE_BEARING = "nde_bearing"
COMPONENT_MOTOR_SHAFT = "motor_shaft"
COMPONENT_MOTOR_ROTOR = "motor_rotor"
COMPONENT_MOTOR_STATOR = "stator_windings"

# Pump 3D Components:
COMPONENT_PUMP_CASING = "pump_casing"
COMPONENT_PUMP_IMPELLER = "pump_impeller"
COMPONENT_PUMP_SHAFT = "pump_shaft"
COMPONENT_PUMP_BEARINGS = "pump_bearings"
COMPONENT_PUMP_SEAL = "mechanical_seal"

# Unidentifiable / Generic:
COMPONENT_UNIDENTIFIABLE = "unidentifiable"
COMPONENT_SENSORS = "sensors"

VALID_3D_COMPONENTS = {
    "motor": [
        COMPONENT_MOTOR_HOUSING,
        COMPONENT_MOTOR_DE_BEARING,
        COMPONENT_MOTOR_NDE_BEARING,
        COMPONENT_MOTOR_SHAFT,
        COMPONENT_MOTOR_ROTOR,
        COMPONENT_MOTOR_STATOR,
    ],
    "pump": [
        COMPONENT_PUMP_CASING,
        COMPONENT_PUMP_IMPELLER,
        COMPONENT_PUMP_SHAFT,
        COMPONENT_PUMP_BEARINGS,
        COMPONENT_PUMP_SEAL,
    ]
}


# -----------------------------------------------------------------------------
# Configurable Diagnostic Thresholds
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosticThresholds:
    """Configurable thresholds for component-level fault isolation."""
    # Motor Thresholds
    motor_temp_warning: float = 72.0     # °C (Stator / Casing thermal stress)
    motor_temp_critical: float = 90.0    # °C (Severe thermal overload)
    motor_vib_warning: float = 3.0       # mm/s RMS (ISO 10816-3 Class II Alert)
    motor_vib_critical: float = 7.1      # mm/s RMS (ISO 10816-3 Unacceptable)
    motor_current_overload: float = 22.0 # Amperes (Rated: ~14.2A)
    motor_current_unloaded: float = 7.0  # Amperes (Dry run / loss of load)
    motor_rated_rpm: float = 1485.0
    motor_min_running_rpm: float = 50.0

    # Pump Thresholds
    pump_temp_warning: float = 65.0      # °C (Bearing/Seal housing warm)
    pump_temp_critical: float = 85.0     # °C (Seal face / bearing friction)
    pump_vib_warning: float = 3.0        # mm/s RMS
    pump_vib_critical: float = 7.1       # mm/s RMS
    pump_pressure_dry_crit: float = 0.8  # Bar (Suction head loss / dry run)
    pump_flow_dry_crit: float = 15.0     # m³/h (Dry run flow collapse)
    pump_pressure_deadhead: float = 8.5  # Bar (Discharge valve closed)
    pump_rated_pressure: float = 4.8     # Bar
    pump_rated_flow: float = 88.5        # m³/h

    # Telemetry Health & Staleness
    stale_timeout_seconds: float = 10.0  # Seconds before marking sensor data stale


DEFAULT_THRESHOLDS = DiagnosticThresholds()


# -----------------------------------------------------------------------------
# Pydantic Schemas for Structured Diagnostic Output
# -----------------------------------------------------------------------------
class ComponentDiagnosticEntry(BaseModel):
    """Structured, explainable diagnostic evaluation for a single machine or component."""
    machine_type: str = PydanticField(..., description="'motor', 'pump', or 'system'")
    suspected_component: Optional[str] = PydanticField(
        None,
        description="3D component ID if uniquely identifiable, or 'unidentifiable' if ambiguous"
    )
    component_name: str = PydanticField(..., description="Human-readable component label")
    fault_category: str = PydanticField(..., description="Suspected fault or operating mode")
    severity: str = PydanticField(..., description="'NORMAL', 'WARNING', or 'CRITICAL'")
    confidence: str = PydanticField(..., description="'High', 'Moderate', 'Low', or 'Uncertain'")
    supporting_evidence: Dict[str, Any] = PydanticField(
        default_factory=dict,
        description="Key sensor readings and comparison thresholds justifying diagnosis"
    )
    explanation: str = PydanticField(..., description="Clear engineering explanation of the diagnosis")
    missing_measurements: List[str] = PydanticField(
        default_factory=list,
        description="Sensor limitations preventing definitive physical confirmation"
    )
    recommended_action: str = PydanticField(..., description="Prescribed physical inspection action")


class ComponentDiagnosticsResult(BaseModel):
    """Aggregate diagnostics result for the digital twin system."""
    timestamp: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    diagnoses: List[ComponentDiagnosticEntry] = PydanticField(default_factory=list)
    highlighted_components: Dict[str, str] = PydanticField(
        default_factory=dict,
        description="Mapping of component_id -> severity ('NORMAL', 'WARNING', 'CRITICAL') for 3D visual twin"
    )
    summary: str = PydanticField(..., description="Executive diagnostic summary")


# -----------------------------------------------------------------------------
# Core Diagnostic Rule Engine
# -----------------------------------------------------------------------------
class DiagnosticEngine:
    """
    Evidence-based condition diagnostic engine.
    Separates machine-level classification from component-level physical isolation.
    """

    def __init__(self, thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS):
        self.cfg = thresholds

    def evaluate(
        self,
        motor_telem: Optional[Any],
        pump_telem: Optional[Any],
        last_updated: Optional[datetime] = None,
        data_source: str = "initialized"
    ) -> ComponentDiagnosticsResult:
        """
        Evaluate current telemetry and generate structured component diagnostics.
        """
        diagnoses: List[ComponentDiagnosticEntry] = []
        highlighted: Dict[str, str] = {}

        # Initialize all known 3D components as NORMAL by default
        for comp in VALID_3D_COMPONENTS["motor"]:
            highlighted[comp] = "NORMAL"
        for comp in VALID_3D_COMPONENTS["pump"]:
            highlighted[comp] = "NORMAL"

        now = datetime.now(timezone.utc)

        # ---------------------------------------------------------------------
        # 1. Telemetry Staleness and Missing Sensor Checks
        # ---------------------------------------------------------------------
        is_stale = False
        if last_updated is not None:
            # Ensure last_updated is timezone-aware for comparison
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)
            age = (now - last_updated).total_seconds()
            if age > self.cfg.stale_timeout_seconds and data_source == "live_sensor":
                is_stale = True

        if motor_telem is None and pump_telem is None:
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="system",
                suspected_component=COMPONENT_SENSORS,
                component_name="IoT Sensor Network",
                fault_category="Telemetry Missing / Disconnected",
                severity="WARNING",
                confidence="High",
                supporting_evidence={"motor_telemetry": None, "pump_telemetry": None},
                explanation="No telemetry packets received for either motor or pump.",
                missing_measurements=["All sensor data missing"],
                recommended_action="Inspect ESP32 microcontroller, Wi-Fi connectivity, and gateway telemetry stream."
            ))
            return ComponentDiagnosticsResult(
                timestamp=now,
                diagnoses=diagnoses,
                highlighted_components=highlighted,
                summary="Telemetry missing: Waiting for sensor stream connection."
            )

        if is_stale:
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="system",
                suspected_component=COMPONENT_SENSORS,
                component_name="IoT Sensor Gateway",
                fault_category="Sensor Telemetry Stale",
                severity="WARNING",
                confidence="High",
                supporting_evidence={"age_seconds": round(age, 1), "timeout": self.cfg.stale_timeout_seconds},
                explanation=f"Telemetry data has not updated for {age:.1f}s (threshold: {self.cfg.stale_timeout_seconds}s).",
                missing_measurements=["Live heartbeat packet lost"],
                recommended_action="Verify field bus / MQTT gateway and ESP32 Wokwi network connection."
            ))

        # ---------------------------------------------------------------------
        # 2. Motor Component Diagnostics
        # ---------------------------------------------------------------------
        if motor_telem is not None:
            self._diagnose_motor(motor_telem, diagnoses, highlighted)
        else:
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="motor",
                suspected_component=COMPONENT_SENSORS,
                component_name="Motor Sensor Package",
                fault_category="Motor Telemetry Missing",
                severity="WARNING",
                confidence="High",
                supporting_evidence={"motor_telemetry": "None"},
                explanation="Motor sensor data is missing from the received telemetry payload.",
                missing_measurements=["Motor temperature, vibration, current, speed"],
                recommended_action="Check DS18B20 and MPU6050 wiring on motor drive end."
            ))

        # ---------------------------------------------------------------------
        # 3. Pump Component Diagnostics
        # ---------------------------------------------------------------------
        if pump_telem is not None:
            self._diagnose_pump(pump_telem, motor_telem, diagnoses, highlighted)
        else:
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=COMPONENT_SENSORS,
                component_name="Pump Sensor Package",
                fault_category="Pump Telemetry Missing",
                severity="WARNING",
                confidence="High",
                supporting_evidence={"pump_telemetry": "None"},
                explanation="Pump sensor data is missing from the received telemetry payload.",
                missing_measurements=["Pump pressure, flow, temperature, vibration"],
                recommended_action="Check pump pressure transducer and flow meter telemetry."
            ))

        # ---------------------------------------------------------------------
        # 4. Overall Executive Summary
        # ---------------------------------------------------------------------
        crit_count = sum(1 for d in diagnoses if d.severity == "CRITICAL")
        warn_count = sum(1 for d in diagnoses if d.severity == "WARNING")
        
        if crit_count > 0:
            summary = f"CRITICAL FAULT DETECTED: {crit_count} critical and {warn_count} warning component alert(s) requiring immediate intervention."
        elif warn_count > 0:
            summary = f"MAINTENANCE WARNING: {warn_count} warning condition(s) detected. Component inspection recommended."
        else:
            summary = "NOMINAL OPERATION: All motor and pump components operating within established baselines."

        return ComponentDiagnosticsResult(
            timestamp=now,
            diagnoses=diagnoses,
            highlighted_components=highlighted,
            summary=summary
        )

    def _diagnose_motor(
        self,
        m: Any,
        diagnoses: List[ComponentDiagnosticEntry],
        highlighted: Dict[str, str]
    ) -> None:
        """Evaluate motor conditions with physical evidence correlation."""
        temp = float(m.temperature)
        vib = float(m.vibration)
        curr = float(m.current)
        rpm = float(m.rpm)

        # A. Standby / Idle Check
        if rpm < self.cfg.motor_min_running_rpm:
            # When stopped, check if there is an abnormal thermal condition while offline
            if temp >= self.cfg.motor_temp_critical:
                highlighted[COMPONENT_MOTOR_STATOR] = "CRITICAL"
                highlighted[COMPONENT_MOTOR_HOUSING] = "WARNING"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=COMPONENT_MOTOR_STATOR,
                    component_name="Stator Windings",
                    fault_category="Severe Thermal Overheat in Standby",
                    severity="CRITICAL",
                    confidence="Moderate",
                    supporting_evidence={
                        "temperature_c": temp,
                        "threshold_c": self.cfg.motor_temp_critical,
                        "rpm": rpm,
                        "current_a": curr
                    },
                    explanation=(
                        f"Motor temperature is {temp:.1f}°C (>= {self.cfg.motor_temp_critical:.1f}°C) "
                        "despite shaft being stationary. Indicates external fire, heat soak, or locked rotor prior to trip."
                    ),
                    missing_measurements=["Embedded winding RTD Pt100", "Infrared thermography scan"],
                    recommended_action="DO NOT restart. De-energize breaker and perform IR thermography scan of terminal box."
                ))
            else:
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=None,
                    component_name="Electric Motor",
                    fault_category="Normal Standby / Idle",
                    severity="NORMAL",
                    confidence="High",
                    supporting_evidence={"rpm": rpm, "current_a": curr, "temperature_c": temp},
                    explanation="Motor is de-energized or in standby mode (RPM < 50). All idle parameters normal.",
                    missing_measurements=[],
                    recommended_action="No action required. Machine ready for dispatch."
                ))
            return

        # B. Thermal Overheating Scenarios
        is_temp_crit = temp >= self.cfg.motor_temp_critical
        is_temp_warn = temp >= self.cfg.motor_temp_warning

        if is_temp_crit or is_temp_warn:
            severity = "CRITICAL" if is_temp_crit else "WARNING"
            thresh = self.cfg.motor_temp_critical if is_temp_crit else self.cfg.motor_temp_warning

            # If current is also high: electrical I²R stator winding losses
            if curr >= self.cfg.motor_current_overload:
                highlighted[COMPONENT_MOTOR_STATOR] = severity
                highlighted[COMPONENT_MOTOR_HOUSING] = "WARNING" if severity == "CRITICAL" else "NORMAL"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=COMPONENT_MOTOR_STATOR,
                    component_name="Stator Windings",
                    fault_category="Electrical Overcurrent & Stator Thermal Overload",
                    severity=severity,
                    confidence="High",
                    supporting_evidence={
                        "temperature_c": temp,
                        "temp_threshold": thresh,
                        "current_a": curr,
                        "rated_current_a": 14.2
                    },
                    explanation=(
                        f"Motor temperature ({temp:.1f}°C) exceeds threshold ({thresh:.1f}°C) "
                        f"correlated with elevated current draw ({curr:.1f}A vs rated ~14.2A). "
                        "Directly indicates excessive stator winding resistive dissipation (I²R losses)."
                    ),
                    missing_measurements=[
                        "Phase-by-phase current unbalance",
                        "Slot RTD temperature distribution",
                        "Motor insulation resistance (Megger test)"
                    ],
                    recommended_action="Inspect motor load, verify 3-phase voltage balance, and measure winding insulation resistance."
                ))
            else:
                # Temperature high without excessive current: cooling airflow blocked or casing heat trap
                highlighted[COMPONENT_MOTOR_HOUSING] = severity
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=COMPONENT_MOTOR_HOUSING,
                    component_name="Motor Housing & Cooling Fins",
                    fault_category="Cooling Degradation / Frame Thermal Dissipation Issue",
                    severity=severity,
                    confidence="Moderate",
                    supporting_evidence={
                        "temperature_c": temp,
                        "temp_threshold": thresh,
                        "current_a": curr,
                        "rated_current_a": 14.2
                    },
                    explanation=(
                        f"Motor frame temperature is elevated ({temp:.1f}°C) while electrical current ({curr:.1f}A) "
                        "remains within reasonable bounds. Likely cause is clogged fan cowl, dirt accumulation on cooling fins, "
                        "or high ambient temperature."
                    ),
                    missing_measurements=["Cooling air inlet/outlet delta-T", "Airflow velocity anemometer"],
                    recommended_action="Clean external motor cooling fins and verify external cooling fan cowl is unobstructed."
                ))

        # C. Vibration & Mechanical Fault Scenarios
        is_vib_crit = vib >= self.cfg.motor_vib_critical
        is_vib_warn = vib >= self.cfg.motor_vib_warning

        if is_vib_crit or is_vib_warn:
            severity = "CRITICAL" if is_vib_crit else "WARNING"
            thresh = self.cfg.motor_vib_critical if is_vib_crit else self.cfg.motor_vib_warning

            # Differentiate: Rotor / Shaft issue vs Bearing issue vs Ambiguous
            # Rotor/Shaft evidence: High vibration combined with high current or significant slip
            if curr >= self.cfg.motor_current_overload and rpm < 1440.0:
                highlighted[COMPONENT_MOTOR_ROTOR] = severity
                highlighted[COMPONENT_MOTOR_SHAFT] = "WARNING"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=COMPONENT_MOTOR_ROTOR,
                    component_name="Motor Rotor & Shaft Assembly",
                    fault_category="Possible Rotor Bar Defect or Dynamic Eccentricity",
                    severity=severity,
                    confidence="Moderate",
                    supporting_evidence={
                        "vibration_rms": vib,
                        "vib_threshold": thresh,
                        "current_a": curr,
                        "rpm": rpm
                    },
                    explanation=(
                        f"Elevated vibration ({vib:.2f} mm/s) accompanied by increased slip (RPM {rpm:.0f} < 1440) "
                        f"and heavy current ({curr:.1f}A). Suggests broken rotor bar, dynamic air-gap eccentricity, or rotor unbalance."
                    ),
                    missing_measurements=[
                        "Motor Current Signature Analysis (MCSA) sideband FFT",
                        "High-resolution encoder speed ripple"
                    ],
                    recommended_action="Perform MCSA spectrum test for twice-slip frequency sidebands around fundamental line frequency."
                ))
            elif is_temp_warn or is_temp_crit:
                # High vibration accompanied by thermal rise at bearing/housing: Drive-end bearing stress
                highlighted[COMPONENT_MOTOR_DE_BEARING] = severity
                highlighted[COMPONENT_MOTOR_SHAFT] = "WARNING"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=COMPONENT_MOTOR_DE_BEARING,
                    component_name="Drive-End (DE) Bearing",
                    fault_category="Mechanical Bearing Degradation & Frictional Heat",
                    severity=severity,
                    confidence="Moderate",
                    supporting_evidence={
                        "vibration_rms": vib,
                        "vib_threshold": thresh,
                        "temperature_c": temp,
                        "temp_threshold": self.cfg.motor_temp_warning
                    },
                    explanation=(
                        f"Vibration ({vib:.2f} mm/s) coupled with thermal elevation ({temp:.1f}°C). "
                        "Drive-end bearing bears highest radial coupling loads; symptoms match raceway spalling or lubrication breakdown."
                    ),
                    missing_measurements=[
                        "High-frequency acceleration envelope (PeakVue/gE)",
                        "DE vs NDE individual accelerometers",
                        "Ultrasound acoustic emission"
                    ],
                    recommended_action="Perform high-frequency demodulation analysis (FFT) and replenish grease or replace DE bearing."
                ))
            else:
                # Elevated vibration alone with nominal current and nominal temperature
                # Prompt rule: Do not falsely assign a specific component when single accelerometer is ambiguous!
                highlighted[COMPONENT_MOTOR_DE_BEARING] = "WARNING"
                highlighted[COMPONENT_MOTOR_NDE_BEARING] = "WARNING"
                highlighted[COMPONENT_MOTOR_SHAFT] = "WARNING"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="motor",
                    suspected_component=COMPONENT_UNIDENTIFIABLE,
                    component_name="Motor Mechanical Drive Train",
                    fault_category="Elevated Vibration — Component Ambiguous",
                    severity=severity,
                    confidence="Low",
                    supporting_evidence={"vibration_rms": vib, "vib_threshold": thresh, "current_a": curr, "rpm": rpm},
                    explanation=(
                        f"Abnormal vibration ({vib:.2f} mm/s) exceeds threshold ({thresh:.1f} mm/s) while temperature and current "
                        "remain nominal. Current single casing accelerometer cannot distinguish between DE/NDE bearing flaw, "
                        "shaft unbalance, coupling misalignment, or baseplate looseness."
                    ),
                    missing_measurements=[
                        "Multi-axis triaxial accelerometer (Horizontal, Vertical, Axial)",
                        "1X, 2X, 3X vibration harmonic spectral analysis (FFT)",
                        "Phase angle analysis across coupling"
                    ],
                    recommended_action="Perform multi-point vibration spectrum survey and check shaft alignment with laser tool."
                ))

        # D. If no fault detected on motor
        if not (is_temp_crit or is_temp_warn or is_vib_crit or is_vib_warn):
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="motor",
                suspected_component=None,
                component_name="Induction Motor Subsystem",
                fault_category="Normal Operating Condition",
                severity="NORMAL",
                confidence="High",
                supporting_evidence={"temperature_c": temp, "vibration_rms": vib, "current_a": curr, "rpm": rpm},
                explanation="Motor operating steadily within nominal thermal, vibration, current, and speed boundaries.",
                missing_measurements=[],
                recommended_action="Continue routine condition monitoring."
            ))

    def _diagnose_pump(
        self,
        p: Any,
        m: Optional[Any],
        diagnoses: List[ComponentDiagnosticEntry],
        highlighted: Dict[str, str]
    ) -> None:
        """Evaluate centrifugal pump conditions with physical evidence correlation."""
        temp = float(p.temperature)
        vib = float(p.vibration)
        curr = float(p.current)
        rpm = float(p.rpm)
        press = float(p.pressure)
        flow = float(p.flow_rate)

        # A. Standby / Idle Check
        if rpm < self.cfg.motor_min_running_rpm:
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=None,
                component_name="Centrifugal Pump",
                fault_category="Normal Standby / Idle",
                severity="NORMAL",
                confidence="High",
                supporting_evidence={"rpm": rpm, "pressure_bar": press, "flow_m3h": flow},
                explanation="Pump is stationary (RPM < 50). Zero pressure and flow expected.",
                missing_measurements=[],
                recommended_action="No action required. Ready for startup."
            ))
            return

        # B. Hydraulic Loss: Suction Loss / Dry-Running / Severe Cavitation
        is_dry_press = press <= self.cfg.pump_pressure_dry_crit
        is_dry_flow = flow < self.cfg.pump_flow_dry_crit

        if is_dry_press and is_dry_flow:
            # Mechanical seal and impeller are directly threatened
            highlighted[COMPONENT_PUMP_SEAL] = "CRITICAL"
            highlighted[COMPONENT_PUMP_IMPELLER] = "CRITICAL"
            highlighted[COMPONENT_PUMP_CASING] = "WARNING"

            # Check if thermal rise is occurring at seal face
            seal_evidence = {
                "pressure_bar": press,
                "pressure_threshold": self.cfg.pump_pressure_dry_crit,
                "flow_m3h": flow,
                "flow_threshold": self.cfg.pump_flow_dry_crit,
                "pump_temp_c": temp,
                "pump_vib_rms": vib
            }
            if m is not None:
                seal_evidence["motor_current_a"] = float(m.current)

            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=COMPONENT_PUMP_SEAL,
                component_name="Mechanical Seal & Impeller",
                fault_category="Pump Dry-Run & Severe Cavitation (Loss of Prime)",
                severity="CRITICAL",
                confidence="High",
                supporting_evidence=seal_evidence,
                explanation=(
                    f"Discharge pressure collapsed to {press:.2f} Bar (<= {self.cfg.pump_pressure_dry_crit:.1f} Bar) "
                    f"and flow rate dropped to {flow:.1f} m³/h (< {self.cfg.pump_flow_dry_crit:.1f} m³/h) while shaft is rotating at {rpm:.0f} RPM. "
                    "Operating without liquid film destroys mechanical seal carbon/silicon carbide faces through friction heat "
                    "and induces vapor cavity collapse damage on impeller blades."
                ),
                missing_measurements=["Suction line NPSHa pressure transmitter", "Seal flush plan barrier fluid pressure"],
                recommended_action="TRIP PUMP IMMEDIATELY to prevent catastrophic mechanical seal destruction and impeller seizure."
            ))
            return

        # C. Deadheading Condition (High pressure, near-zero flow)
        if press >= self.cfg.pump_pressure_deadhead and flow < self.cfg.pump_flow_dry_crit:
            highlighted[COMPONENT_PUMP_CASING] = "CRITICAL"
            highlighted[COMPONENT_PUMP_IMPELLER] = "WARNING"
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=COMPONENT_PUMP_CASING,
                component_name="Pump Volute Casing",
                fault_category="Deadhead / Closed Discharge Valve",
                severity="CRITICAL",
                confidence="High",
                supporting_evidence={
                    "pressure_bar": press,
                    "pressure_threshold": self.cfg.pump_pressure_deadhead,
                    "flow_m3h": flow,
                    "flow_threshold": self.cfg.pump_flow_dry_crit
                },
                explanation=(
                    f"Discharge pressure surged to {press:.2f} Bar (>= {self.cfg.pump_pressure_deadhead:.1f} Bar) "
                    f"with near-zero flow ({flow:.1f} m³/h). Trapped fluid inside casing will boil and overpressurize casing."
                ),
                missing_measurements=["Discharge valve limit switch state"],
                recommended_action="Open discharge valve or divert through minimum flow bypass loop immediately."
            ))
            return

        # D. Hydraulic Impeller Degradation (Reduced head or flow while speed is rated)
        if press < 3.2 and flow < 60.0 and rpm >= 1400.0:
            highlighted[COMPONENT_PUMP_IMPELLER] = "WARNING"
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=COMPONENT_PUMP_IMPELLER,
                component_name="Pump Impeller",
                fault_category="Hydraulic Degradation / Partial Impeller Vane Erosion",
                severity="WARNING",
                confidence="Moderate",
                supporting_evidence={
                    "pressure_bar": press,
                    "flow_m3h": flow,
                    "rpm": rpm,
                    "expected_rated_pressure": self.cfg.pump_rated_pressure,
                    "expected_rated_flow": self.cfg.pump_rated_flow
                },
                explanation=(
                    f"Delivered pressure ({press:.2f} Bar) and flow ({flow:.1f} m³/h) are significantly below "
                    f"rated pump curve at {rpm:.0f} RPM. Suggests internal recirculation, wear ring clearance widening, or impeller vane erosion."
                ),
                missing_measurements=["Wear ring clearance gauge", "Suction strainer differential pressure"],
                recommended_action="Inspect pump suction strainer for blockage and check impeller wear ring clearances."
            ))

        # E. Pump Mechanical Vibration / Bearing Fault Scenarios
        is_pvib_crit = vib >= self.cfg.pump_vib_critical
        is_pvib_warn = vib >= self.cfg.pump_vib_warning
        is_ptemp_crit = temp >= self.cfg.pump_temp_critical
        is_ptemp_warn = temp >= self.cfg.pump_temp_warning

        if is_pvib_crit or is_pvib_warn:
            severity = "CRITICAL" if is_pvib_crit else "WARNING"
            thresh = self.cfg.pump_vib_critical if is_pvib_crit else self.cfg.pump_vib_warning

            if is_ptemp_warn or is_ptemp_crit:
                # Vibration + Temperature: Bearing issue
                highlighted[COMPONENT_PUMP_BEARINGS] = severity
                highlighted[COMPONENT_PUMP_SHAFT] = "WARNING"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="pump",
                    suspected_component=COMPONENT_PUMP_BEARINGS,
                    component_name="Pump Bearings",
                    fault_category="Pump Bearing Overheating & Mechanical Stress",
                    severity=severity,
                    confidence="High",
                    supporting_evidence={
                        "vibration_rms": vib,
                        "vib_threshold": thresh,
                        "temperature_c": temp,
                        "temp_threshold": self.cfg.pump_temp_warning
                    },
                    explanation=(
                        f"Elevated pump vibration ({vib:.2f} mm/s) accompanied by bearing housing temperature ({temp:.1f}°C). "
                        "Symptoms indicate bearing cage distress, lubrication degradation, or excessive axial thrust."
                    ),
                    missing_measurements=[
                        "Bearing vibration demodulation spectrum",
                        "Lube oil contamination / particle count"
                    ],
                    recommended_action="Check bearing oil level and condition; schedule bearing replacement before catastrophic failure."
                ))
            else:
                # High vibration with normal hydraulic output and normal temperature: Ambiguous
                highlighted[COMPONENT_PUMP_BEARINGS] = "WARNING"
                highlighted[COMPONENT_PUMP_IMPELLER] = "WARNING"
                highlighted[COMPONENT_PUMP_SHAFT] = "WARNING"
                diagnoses.append(ComponentDiagnosticEntry(
                    machine_type="pump",
                    suspected_component=COMPONENT_UNIDENTIFIABLE,
                    component_name="Pump Mechanical Structure",
                    fault_category="Elevated Pump Vibration — Component Ambiguous",
                    severity=severity,
                    confidence="Low",
                    supporting_evidence={"vibration_rms": vib, "vib_threshold": thresh, "pressure_bar": press, "flow_m3h": flow},
                    explanation=(
                        f"Pump vibration ({vib:.2f} mm/s) exceeds threshold ({thresh:.1f} mm/s), but hydraulic performance is stable. "
                        "Cannot isolate without FFT whether origin is impeller imbalance, shaft runout, or bearing looseness."
                    ),
                    missing_measurements=[
                        "Vibration velocity FFT spectrum",
                        "Blade pass frequency (BPF = Vanes × RPM / 60)",
                        "Shaft radial runout proximity probes"
                    ],
                    recommended_action="Conduct spectral FFT analysis to separate 1X rotational unbalance from blade pass frequency (BPF)."
                ))
        elif is_ptemp_crit or is_ptemp_warn:
            severity = "CRITICAL" if is_ptemp_crit else "WARNING"
            highlighted[COMPONENT_PUMP_BEARINGS] = severity
            highlighted[COMPONENT_PUMP_SEAL] = "WARNING"
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=COMPONENT_PUMP_BEARINGS,
                component_name="Pump Bearing / Seal Housing",
                fault_category="Pump Bearing / Seal Housing Overheating",
                severity=severity,
                confidence="Moderate",
                supporting_evidence={
                    "temperature_c": temp,
                    "temp_threshold": self.cfg.pump_temp_critical if is_ptemp_crit else self.cfg.pump_temp_warning
                },
                explanation=(
                    f"Pump housing temperature is {temp:.1f}°C, exceeding thermal limits. "
                    "Check for deficient bearing lubrication, churning, or seal face friction."
                ),
                missing_measurements=["Individual seal gland temperature probe", "Bearing oil temperature probe"],
                recommended_action="Inspect bearing oil reservoir level and seal quench fluid flow."
            ))
        else:
            diagnoses.append(ComponentDiagnosticEntry(
                machine_type="pump",
                suspected_component=None,
                component_name="Centrifugal Pump Subsystem",
                fault_category="Normal Operating Condition",
                severity="NORMAL",
                confidence="High",
                supporting_evidence={
                    "temperature_c": temp,
                    "vibration_rms": vib,
                    "pressure_bar": press,
                    "flow_m3h": flow,
                    "rpm": rpm
                },
                explanation="Pump operating steadily within nominal hydraulic, vibration, and thermal parameters.",
                missing_measurements=[],
                recommended_action="Continue routine monitoring."
            ))


# Global diagnostic engine instance
diagnostic_engine = DiagnosticEngine()
