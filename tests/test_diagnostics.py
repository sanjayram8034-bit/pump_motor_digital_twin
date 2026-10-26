"""
Comprehensive Verification Tests for Component-Level Diagnostics Engine and 3D Mappings.

Tests:
1. Normal Operating Condition (Both machines nominal -> all 3D components NORMAL)
2. Motor Overheating Scenarios (Stator windings vs Casing airflow blockage)
3. Elevated Motor Vibration Ambiguity (Single accelerometer -> ambiguous, not falsely assigned)
4. Corroborated Motor Bearing Issue (Vibration + Temp -> Drive-End bearing highlight)
5. Rotor / Shaft Issue (Vibration + Slip + Heavy Current -> Rotor / Shaft)
6. Pump Low-Flow / Suction Loss / Dry-Run (Pressure + Flow collapse -> Mechanical Seal + Impeller)
7. Pump Deadhead Condition (High Pressure + Zero Flow -> Volute Casing)
8. Pump Bearing Fault (Vibration + Temp -> Pump Bearings)
9. Missing, Stale, and Invalid Telemetry Handling
10. Separation of Machine ML Status vs Component Diagnosis
11. 3D Component Identifier Validity (All highlighted keys exist in VALID_3D_COMPONENTS)
12. Recovery Workflow (Fault cleared -> components reset to NORMAL without ghost alerts)
13. FastAPI Endpoint Integration (/twin returns valid diagnostics schema)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

# Ensure root package directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.twin.diagnostics import (
    DiagnosticEngine,
    DiagnosticThresholds,
    diagnostic_engine,
    VALID_3D_COMPONENTS,
    COMPONENT_MOTOR_HOUSING,
    COMPONENT_MOTOR_DE_BEARING,
    COMPONENT_MOTOR_NDE_BEARING,
    COMPONENT_MOTOR_SHAFT,
    COMPONENT_MOTOR_ROTOR,
    COMPONENT_MOTOR_STATOR,
    COMPONENT_PUMP_CASING,
    COMPONENT_PUMP_IMPELLER,
    COMPONENT_PUMP_SHAFT,
    COMPONENT_PUMP_BEARINGS,
    COMPONENT_PUMP_SEAL,
    COMPONENT_UNIDENTIFIABLE,
    COMPONENT_SENSORS
)
from backend.twin.state import (
    DigitalTwinState,
    TelemetryPayload,
    MotorTelemetry,
    PumpTelemetry
)
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def engine():
    return DiagnosticEngine()


def test_3d_component_catalog_completeness():
    """Verify that all required motor and pump 3D components exist in definition."""
    motor_comps = set(VALID_3D_COMPONENTS["motor"])
    expected_motor = {
        "motor_housing", "de_bearing", "nde_bearing",
        "motor_shaft", "motor_rotor", "stator_windings"
    }
    assert motor_comps == expected_motor

    pump_comps = set(VALID_3D_COMPONENTS["pump"])
    expected_pump = {
        "pump_casing", "pump_impeller", "pump_shaft",
        "pump_bearings", "mechanical_seal"
    }
    assert pump_comps == expected_pump


def test_normal_operating_conditions(engine):
    """Test nominal baseline telemetry produces NORMAL for all 3D components."""
    motor = MotorTelemetry(temperature=56.0, vibration=1.5, current=14.0, rpm=1485.0)
    pump = PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1485.0, pressure=4.8, flow_rate=88.0)

    result = engine.evaluate(motor, pump, last_updated=datetime.now(timezone.utc), data_source="live_sensor")

    assert "NOMINAL" in result.summary.upper()
    # Check that all motor components are NORMAL
    for comp in VALID_3D_COMPONENTS["motor"]:
        assert result.highlighted_components[comp] == "NORMAL"
    # Check that all pump components are NORMAL
    for comp in VALID_3D_COMPONENTS["pump"]:
        assert result.highlighted_components[comp] == "NORMAL"

    severities = [d.severity for d in result.diagnoses]
    assert all(s == "NORMAL" for s in severities)


def test_motor_overheating_stator_windings(engine):
    """High temperature + High current -> Stator windings thermal overload."""
    motor = MotorTelemetry(temperature=95.0, vibration=2.0, current=26.0, rpm=1450.0)
    pump = PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1450.0, pressure=4.8, flow_rate=88.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_MOTOR_STATOR] == "CRITICAL"
    stator_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_MOTOR_STATOR)
    assert stator_diag.severity == "CRITICAL"
    assert "Stator" in stator_diag.component_name
    assert stator_diag.confidence in ["High", "Moderate"]
    assert "current_a" in stator_diag.supporting_evidence
    assert len(stator_diag.missing_measurements) > 0


def test_motor_cooling_casing_issue(engine):
    """High temperature with normal current -> Frame cooling issue."""
    motor = MotorTelemetry(temperature=76.0, vibration=1.8, current=13.8, rpm=1485.0)
    pump = PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1485.0, pressure=4.8, flow_rate=88.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_MOTOR_HOUSING] == "WARNING"
    housing_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_MOTOR_HOUSING)
    assert housing_diag.severity == "WARNING"
    assert "Cooling" in housing_diag.fault_category or "Housing" in housing_diag.component_name


def test_motor_vibration_ambiguity_not_falsely_assigned(engine):
    """
    CRITICAL CONSTRAINT: Single abnormal vibration measurement must NOT automatically
    confirm a specific component if sensors cannot distinguish between DE, NDE, or shaft.
    """
    motor = MotorTelemetry(temperature=55.0, vibration=5.0, current=14.0, rpm=1485.0)
    pump = PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1485.0, pressure=4.8, flow_rate=88.0)

    result = engine.evaluate(motor, pump)
    ambiguous_diag = next(d for d in result.diagnoses if d.machine_type == "motor" and d.severity in ["WARNING", "CRITICAL"])
    assert ambiguous_diag.suspected_component == COMPONENT_UNIDENTIFIABLE
    assert ambiguous_diag.confidence in ["Low", "Uncertain"]
    assert "cannot distinguish" in ambiguous_diag.explanation.lower() or "ambiguous" in ambiguous_diag.fault_category.lower()


def test_motor_bearing_fault_with_corroborating_thermal_evidence(engine):
    """Elevated vibration + elevated temperature -> Drive-End bearing distress."""
    motor = MotorTelemetry(temperature=75.0, vibration=5.5, current=15.0, rpm=1480.0)
    pump = PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1480.0, pressure=4.8, flow_rate=88.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_MOTOR_DE_BEARING] in ["WARNING", "CRITICAL"]
    de_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_MOTOR_DE_BEARING)
    assert "Drive-End" in de_diag.component_name
    assert de_diag.severity in ["WARNING", "CRITICAL"]


def test_motor_rotor_and_shaft_fault_scenario(engine):
    """Vibration + High Slip (low RPM under load) + Heavy current -> Rotor bar / shaft unbalance."""
    motor = MotorTelemetry(temperature=76.0, vibration=6.8, current=26.5, rpm=1380.0)
    pump = PumpTelemetry(temperature=48.0, vibration=1.8, current=9.5, rpm=1380.0, pressure=4.2, flow_rate=82.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_MOTOR_ROTOR] in ["WARNING", "CRITICAL"]
    rotor_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_MOTOR_ROTOR)
    assert "Rotor" in rotor_diag.component_name


def test_pump_dry_run_and_mechanical_seal_threat(engine):
    """Collapsed pressure + collapsed flow while running -> Mechanical seal and impeller critical."""
    motor = MotorTelemetry(temperature=56.0, vibration=2.2, current=6.1, rpm=1493.0)
    pump = PumpTelemetry(temperature=92.0, vibration=9.2, current=2.5, rpm=1493.0, pressure=0.3, flow_rate=5.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_PUMP_SEAL] == "CRITICAL"
    assert result.highlighted_components[COMPONENT_PUMP_IMPELLER] == "CRITICAL"

    seal_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_PUMP_SEAL)
    assert seal_diag.severity == "CRITICAL"
    assert "Dry-Run" in seal_diag.fault_category or "Cavitation" in seal_diag.fault_category
    assert seal_diag.confidence == "High"
    assert "TRIP PUMP" in seal_diag.recommended_action


def test_pump_deadhead_closed_valve(engine):
    """High pressure + near-zero flow -> Pump casing overpressure."""
    motor = MotorTelemetry(temperature=58.0, vibration=2.0, current=11.0, rpm=1488.0)
    pump = PumpTelemetry(temperature=68.0, vibration=3.0, current=8.0, rpm=1488.0, pressure=9.5, flow_rate=4.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_PUMP_CASING] == "CRITICAL"
    casing_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_PUMP_CASING)
    assert "Deadhead" in casing_diag.fault_category


def test_pump_impeller_hydraulic_degradation(engine):
    """Low head/flow despite rated speed -> Impeller wear."""
    motor = MotorTelemetry(temperature=56.0, vibration=1.8, current=11.0, rpm=1485.0)
    pump = PumpTelemetry(temperature=47.0, vibration=2.2, current=10.0, rpm=1485.0, pressure=2.5, flow_rate=45.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_PUMP_IMPELLER] == "WARNING"
    impeller_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_PUMP_IMPELLER)
    assert "Impeller" in impeller_diag.component_name


def test_pump_bearing_overheat_and_vibration(engine):
    """High pump vibration + elevated pump temperature -> Pump bearings."""
    motor = MotorTelemetry(temperature=56.0, vibration=1.8, current=14.0, rpm=1485.0)
    pump = PumpTelemetry(temperature=78.0, vibration=6.5, current=11.8, rpm=1485.0, pressure=4.7, flow_rate=87.0)

    result = engine.evaluate(motor, pump)
    assert result.highlighted_components[COMPONENT_PUMP_BEARINGS] in ["WARNING", "CRITICAL"]
    pb_diag = next(d for d in result.diagnoses if d.suspected_component == COMPONENT_PUMP_BEARINGS)
    assert "Bearing" in pb_diag.component_name


def test_missing_telemetry_handling(engine):
    """When both sensor streams are None, returns appropriate warning."""
    result = engine.evaluate(None, None)
    assert len(result.diagnoses) > 0
    sensor_diag = result.diagnoses[0]
    assert sensor_diag.suspected_component == COMPONENT_SENSORS
    assert sensor_diag.severity == "WARNING"


def test_stale_sensor_telemetry_detection(engine):
    """Detects when live telemetry has not updated within stale threshold."""
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=15)
    motor = MotorTelemetry(temperature=56.0, vibration=1.5, current=14.0, rpm=1485.0)
    pump = PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1485.0, pressure=4.8, flow_rate=88.0)

    result = engine.evaluate(motor, pump, last_updated=stale_time, data_source="live_sensor")
    stale_diags = [d for d in result.diagnoses if d.fault_category == "Sensor Telemetry Stale"]
    assert len(stale_diags) == 1
    assert stale_diags[0].severity == "WARNING"


def test_machine_ml_status_vs_component_diagnosis_separation():
    """
    CRITICAL CONSTRAINT: Machine-level status must remain distinct from
    component-level diagnosis. A machine can be WARNING or CRITICAL without
    every component failing.
    """
    twin = DigitalTwinState()
    # Scenario: High pump vibration triggers pump warning
    payload = TelemetryPayload(
        motor=MotorTelemetry(temperature=56.0, vibration=1.5, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=78.0, vibration=6.2, current=11.8, rpm=1485.0, pressure=4.7, flow_rate=87.0)
    )
    snap = twin.update_telemetry(payload)

    # Machine-level status is WARNING
    assert snap.pump.status in ["WARNING", "CRITICAL"]
    # Motor machine status is RUNNING
    assert snap.motor.status == "RUNNING"

    # Component diagnosis specifies pump_bearings, while motor_housing and pump_casing remain NORMAL
    assert snap.diagnostics is not None
    assert snap.diagnostics.highlighted_components[COMPONENT_PUMP_BEARINGS] in ["WARNING", "CRITICAL"]
    assert snap.diagnostics.highlighted_components[COMPONENT_MOTOR_HOUSING] == "NORMAL"
    assert snap.diagnostics.highlighted_components[COMPONENT_PUMP_CASING] == "NORMAL"


def test_recovery_workflow_clears_component_highlights():
    """Test that after a fault clears, components recover to NORMAL state."""
    twin = DigitalTwinState()

    # Step 1: Inject fault
    fault_payload = TelemetryPayload(
        motor=MotorTelemetry(temperature=95.0, vibration=2.0, current=26.0, rpm=1450.0),
        pump=PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1450.0, pressure=4.8, flow_rate=88.0)
    )
    snap1 = twin.update_telemetry(fault_payload)
    assert snap1.diagnostics.highlighted_components[COMPONENT_MOTOR_STATOR] == "CRITICAL"

    # Step 2: Recover to normal
    recovery_payload = TelemetryPayload(
        motor=MotorTelemetry(temperature=56.0, vibration=1.5, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=46.0, vibration=1.2, current=11.5, rpm=1485.0, pressure=4.8, flow_rate=88.0)
    )
    snap2 = twin.update_telemetry(recovery_payload)
    assert snap2.diagnostics.highlighted_components[COMPONENT_MOTOR_STATOR] == "NORMAL"
    for comp, state in snap2.diagnostics.highlighted_components.items():
        assert state == "NORMAL", f"Component {comp} did not reset to NORMAL upon recovery"


def test_fastapi_twin_endpoint_returns_diagnostics(client):
    """Verify GET /twin and POST /telemetry return diagnostics in schema."""
    resp = client.get("/twin")
    assert resp.status_code == 200
    data = resp.json()
    assert "diagnostics" in data
    assert data["diagnostics"] is not None
    assert "highlighted_components" in data["diagnostics"]
    assert "diagnoses" in data["diagnostics"]
    assert "summary" in data["diagnostics"]

    # Post simulation and verify diagnostics are populated
    sim_resp = client.post("/telemetry/simulate?scenario=motor_overheat")
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    assert sim_data["diagnostics"]["highlighted_components"]["stator_windings"] == "CRITICAL"
