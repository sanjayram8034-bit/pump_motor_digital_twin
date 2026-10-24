"""
Comprehensive Automated Verification Suite for Pump & Motor Digital Twin ML.

Tests:
1. Synthetic Data Generators (reproducibility, schemas, physical bounds)
2. Motor & Pump Predictors (inference, validation, graceful fallback)
3. DigitalTwinState integration
4. FastAPI Endpoints (/telemetry, /twin, /)
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure root package directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.motor_model.data_generator import (
    generate_motor_synthetic_data,
    MOTOR_FEATURE_NAMES,
    MOTOR_CLASSES
)
from ml.pump_model.data_generator import (
    generate_pump_synthetic_data,
    PUMP_FEATURE_NAMES,
    PUMP_CLASSES
)
from ml.motor_model.predict import MotorPredictor, motor_predictor
from ml.pump_model.predict import PumpPredictor, pump_predictor
from backend.twin.state import DigitalTwinState, TelemetryPayload, MotorTelemetry, PumpTelemetry
from backend.main import app


def test_motor_synthetic_data_generator():
    """Verify reproducible generation, feature ordering, and class balance."""
    df1 = generate_motor_synthetic_data(n_samples_per_class=100, random_seed=42)
    df2 = generate_motor_synthetic_data(n_samples_per_class=100, random_seed=42)
    assert len(df1) == 300
    assert list(df1.columns) == MOTOR_FEATURE_NAMES + ["condition"]
    assert set(df1["condition"].unique()) == set(MOTOR_CLASSES)
    # Check reproducibility
    assert df1.equals(df2), "Synthetic generator must be strictly deterministic given random_seed"


def test_pump_synthetic_data_generator():
    """Verify pump synthetic data generation and schemas."""
    df = generate_pump_synthetic_data(n_samples_per_class=100, random_seed=42)
    assert len(df) == 300
    assert list(df.columns) == PUMP_FEATURE_NAMES + ["condition"]
    assert set(df["condition"].unique()) == set(PUMP_CLASSES)


def test_motor_predictor_validation_and_inference():
    """Test motor input validation and model predictions."""
    assert motor_predictor.model_available is True

    # 1. Input validation error checks
    with pytest.raises(ValueError):
        motor_predictor.predict({"temperature": 60.0})  # Missing vibration, current, rpm

    with pytest.raises(ValueError):
        motor_predictor.predict({"temperature": "invalid", "vibration": 1.0, "current": 10.0, "rpm": 1480.0})

    # 2. Normal sample prediction
    res_normal = motor_predictor.predict({
        "temperature": 55.0,
        "vibration": 1.5,
        "current": 13.5,
        "rpm": 1485.0
    })
    assert res_normal["condition"] == "Normal"
    assert res_normal["model_available"] is True
    assert 0.0 <= res_normal["confidence"] <= 1.0

    # 3. Critical sample prediction
    res_crit = motor_predictor.predict({
        "temperature": 105.0,
        "vibration": 10.5,
        "current": 38.0,
        "rpm": 1150.0
    })
    assert res_crit["condition"] == "Critical"
    assert res_crit["model_available"] is True


def test_pump_predictor_validation_and_inference():
    """Test pump input validation and model predictions."""
    assert pump_predictor.model_available is True

    # 1. Missing feature error
    with pytest.raises(ValueError):
        pump_predictor.predict({"temperature": 50.0, "pressure": 4.5})

    # 2. Normal sample prediction
    res_normal = pump_predictor.predict({
        "temperature": 48.0,
        "vibration": 1.4,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 85.0
    })
    assert res_normal["condition"] == "Normal"
    assert res_normal["model_available"] is True

    # 3. Critical dry-run cavitation prediction
    res_crit = pump_predictor.predict({
        "temperature": 95.0,
        "vibration": 9.5,
        "current": 3.5,
        "rpm": 1250.0,
        "pressure": 0.3,
        "flow_rate": 4.0
    })
    assert res_crit["condition"] == "Critical"
    assert res_crit["model_available"] is True


def test_graceful_fallback_when_model_missing():
    """Verify system falls back smoothly without crashing when model artifact is absent."""
    missing_motor = MotorPredictor(model_path=Path("non_existent_motor.joblib"))
    assert missing_motor.model_available is False
    res_m = missing_motor.predict({
        "temperature": 55.0,
        "vibration": 1.5,
        "current": 13.5,
        "rpm": 1485.0
    })
    assert res_m["model_available"] is False
    assert res_m["condition"] in MOTOR_CLASSES
    assert "Rule-based fallback" in res_m["details"]

    missing_pump = PumpPredictor(model_path=Path("non_existent_pump.joblib"))
    assert missing_pump.model_available is False
    res_p = missing_pump.predict({
        "temperature": 48.0,
        "vibration": 1.4,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 85.0
    })
    assert res_p["model_available"] is False
    assert res_p["condition"] in PUMP_CLASSES
    assert "Rule-based fallback" in res_p["details"]


def test_fastapi_endpoints_and_twin_response():
    """Test /twin and /telemetry endpoints and verify exposed ML fields."""
    client = TestClient(app)

    # 1. GET /twin initial
    r_twin = client.get("/twin")
    assert r_twin.status_code == 200
    data = r_twin.json()
    assert "motor" in data and "pump" in data
    assert data["motor"]["model_available"] is True
    assert data["pump"]["model_available"] is True

    # 2. POST /telemetry with normal telemetry
    payload = {
        "motor": {
            "temperature": 56.0,
            "vibration": 1.5,
            "current": 13.8,
            "rpm": 1485.0
        },
        "pump": {
            "temperature": 47.0,
            "vibration": 1.4,
            "current": 12.5,
            "rpm": 1485.0,
            "pressure": 4.5,
            "flow_rate": 82.0
        }
    }
    r_post = client.post("/telemetry", json=payload)
    assert r_post.status_code == 200
    res = r_post.json()

    # Check motor ML response
    assert res["motor"]["predicted_condition"] == "Normal"
    assert res["motor"]["model_available"] is True
    assert res["motor"]["prediction_confidence"] is not None
    assert res["motor"]["status"] == "RUNNING"

    # Check pump ML response
    assert res["pump"]["predicted_condition"] == "Normal"
    assert res["pump"]["model_available"] is True
    assert res["pump"]["prediction_confidence"] is not None
    assert res["pump"]["status"] == "RUNNING"
    assert res["system_status"] == "OPERATIONAL"

    # 3. GET / dashboard
    r_dash = client.get("/")
    assert r_dash.status_code == 200
    assert "Industrial Digital Twin" in r_dash.text
    assert "ML Condition" in r_dash.text


def test_simulation_baseline_reset():
    """Verify normal scenario resets both components to rated baseline."""
    from backend.twin.simulation import generate_coupled_telemetry, DEFAULT_SIM_CONFIG
    payload = generate_coupled_telemetry("normal")
    assert payload.motor.rpm == DEFAULT_SIM_CONFIG.rated_rpm
    assert payload.motor.current == DEFAULT_SIM_CONFIG.rated_motor_current
    assert payload.motor.temperature == DEFAULT_SIM_CONFIG.rated_motor_temp
    assert payload.motor.vibration == DEFAULT_SIM_CONFIG.baseline_motor_vib

    assert payload.pump.rpm == DEFAULT_SIM_CONFIG.rated_rpm
    assert payload.pump.flow_rate == DEFAULT_SIM_CONFIG.rated_flow_rate
    assert payload.pump.pressure == DEFAULT_SIM_CONFIG.rated_pressure
    assert payload.pump.current == DEFAULT_SIM_CONFIG.rated_pump_current


def test_simulation_idle_stopped_system():
    """Verify idle scenario correctly represents a stopped system."""
    from backend.twin.simulation import generate_coupled_telemetry
    payload = generate_coupled_telemetry("idle")
    assert payload.motor.rpm == 0.0
    assert payload.motor.current == 0.0
    assert payload.pump.rpm == 0.0
    assert payload.pump.current == 0.0
    assert payload.pump.pressure == 0.0
    assert payload.pump.flow_rate == 0.0


def test_simulation_motor_warning_and_derived_pump_coupling():
    """
    Verify motor stress scenario:
    - Direct shaft coupling: RPM_pump == RPM_motor
    - Pump Affinity laws: Q and P decrease proportionally to motor RPM drop
    - Transmitted vibration: pump vibration rises based on transmissibility ratio
    - Independent ML: motor reports Warning, pump reports Normal
    """
    from backend.twin.simulation import generate_coupled_telemetry, DEFAULT_SIM_CONFIG
    payload = generate_coupled_telemetry("motor_warning")

    # 1. Motor fault parameters
    assert payload.motor.temperature == 78.5
    assert payload.motor.vibration == 5.2
    assert payload.motor.current == 23.2
    assert payload.motor.rpm == 1445.0

    # 2. Direct mechanical coupling (RPM identity)
    assert payload.pump.rpm == payload.motor.rpm == 1445.0

    # 3. Pump Affinity Laws: Q ∝ N, P ∝ N^2
    assert payload.pump.flow_rate < DEFAULT_SIM_CONFIG.rated_flow_rate
    assert payload.pump.pressure < DEFAULT_SIM_CONFIG.rated_pressure
    expected_flow = round(DEFAULT_SIM_CONFIG.rated_flow_rate * (1445.0 / 1485.0), 1)
    assert abs(payload.pump.flow_rate - expected_flow) <= 0.1

    # 4. Transmitted vibration
    assert payload.pump.vibration > DEFAULT_SIM_CONFIG.baseline_pump_vib

    # 5. Independent ML predictions
    pred_m = motor_predictor.predict(payload.motor)
    pred_p = pump_predictor.predict(payload.pump)
    assert pred_m["condition"] == "Warning"
    assert pred_p["condition"] == "Normal"


def test_simulation_pump_dryrun_and_motor_current_reduction():
    """
    Verify pump dry-run scenario:
    - Pump pressure collapses (<= 0.5 Bar) and flow collapses (<= 10 m³/h)
    - Motor unloads: motor current drops from ~14.2A to no-load ~6.1A
    - Motor RPM rises towards synchronous speed (1493 RPM)
    - Directly coupled pump RPM == motor RPM
    - Motor temperature remains near normal baseline (~56.0°C), no artificial thermal spike
    - Transmitted cavitation shock increases motor vibration
    - Independent ML: Pump reports Critical, Motor reports Normal!
    """
    from backend.twin.simulation import generate_coupled_telemetry, DEFAULT_SIM_CONFIG
    payload = generate_coupled_telemetry("pump_critical")

    # 1. Pump failure
    assert payload.pump.pressure <= 0.5
    assert payload.pump.flow_rate <= 10.0
    assert payload.pump.temperature >= 90.0
    assert payload.pump.vibration >= 8.0

    # 2. Coupled motor electrical response
    assert payload.motor.current < 8.0  # Significant drop under fluid loss
    assert payload.motor.current < DEFAULT_SIM_CONFIG.rated_motor_current

    # 3. Coupled motor speed response (slip reduction)
    assert payload.motor.rpm > DEFAULT_SIM_CONFIG.rated_rpm
    assert payload.pump.rpm == payload.motor.rpm

    # 4. Motor thermal response (stays near normal baseline)
    assert abs(payload.motor.temperature - DEFAULT_SIM_CONFIG.rated_motor_temp) <= 1.0

    # 5. Transmitted cavitation shock
    assert payload.motor.vibration > DEFAULT_SIM_CONFIG.baseline_motor_vib

    # 6. ML Independent Predictions: Pump is Critical, Motor remains Normal!
    pred_p = pump_predictor.predict(payload.pump)
    pred_m = motor_predictor.predict(payload.motor)
    assert pred_p["condition"] == "Critical"
    assert pred_m["condition"] == "Normal"


def test_fastapi_simulation_routes():
    """Verify /telemetry/simulate POST and GET endpoints."""
    client = TestClient(app)

    # 1. POST /telemetry/simulate?scenario=pump_critical
    r_sim = client.post("/telemetry/simulate?scenario=pump_critical")
    assert r_sim.status_code == 200
    data = r_sim.json()
    assert data["system_status"] == "CRITICAL_ALERT"
    assert data["pump"]["predicted_condition"] == "Critical"
    assert data["motor"]["predicted_condition"] == "Normal"
    assert data["motor"]["telemetry"]["current"] < 8.0  # Unloaded motor current

    # 2. GET /telemetry/simulate?scenario=motor_warning
    r_get = client.get("/telemetry/simulate?scenario=motor_warning")
    assert r_get.status_code == 200
    payload = r_get.json()
    assert payload["motor"]["rpm"] == payload["pump"]["rpm"] == 1445.0

    # 3. POST /telemetry/simulate?scenario=normal resets
    r_reset = client.post("/telemetry/simulate?scenario=normal")
    assert r_reset.status_code == 200
    data_reset = r_reset.json()
    assert data_reset["motor"]["predicted_condition"] == "Normal"
    assert data_reset["pump"]["predicted_condition"] == "Normal"


def test_motor_temperature_guardrail_and_isolated_faults():
    """
    Verify Motor condition predictions for:
    - Normal temperature and normal sensor readings produce Normal.
    - Temperature-only input at 80°C produces at least Warning.
    - Temperature at or above 90°C produces Critical.
    - High vibration with otherwise normal readings is detected appropriately.
    - Severe combined fault remains Critical.
    """
    # 1. Normal operating conditions produce Normal
    norm_res = motor_predictor.predict({
        "temperature": 55.0,
        "vibration": 1.6,
        "current": 13.8,
        "rpm": 1485.0
    })
    assert norm_res["condition"] == "Normal"
    assert norm_res["model_available"] is True

    # 2. Temperature-only at 80°C (with nominal vibration, current, rpm) produces at least Warning
    temp80_res = motor_predictor.predict({
        "temperature": 80.0,
        "vibration": 1.6,
        "current": 13.8,
        "rpm": 1485.0
    })
    assert temp80_res["condition"] in ["Warning", "Critical"]
    assert temp80_res["condition"] != "Normal"
    assert temp80_res["guardrail_active"] is True
    assert temp80_res["prediction_source"] in ["temperature_guardrail", "ml_model_and_guardrail", "ml_model"]

    # 3. Temperature at or above 90°C produces Critical
    temp90_res = motor_predictor.predict({
        "temperature": 90.0,
        "vibration": 1.6,
        "current": 13.8,
        "rpm": 1485.0
    })
    assert temp90_res["condition"] == "Critical"
    assert temp90_res["guardrail_active"] is True

    temp95_res = motor_predictor.predict({
        "temperature": 95.0,
        "vibration": 1.6,
        "current": 13.8,
        "rpm": 1485.0
    })
    assert temp95_res["condition"] == "Critical"

    # 4. High vibration with otherwise normal readings (temp 55, current 13.8, rpm 1485)
    vib_res = motor_predictor.predict({
        "temperature": 55.0,
        "vibration": 5.2,
        "current": 13.8,
        "rpm": 1485.0
    })
    assert vib_res["condition"] in ["Warning", "Critical"]

    # 5. Severe combined fault remains Critical
    severe_res = motor_predictor.predict({
        "temperature": 105.0,
        "vibration": 10.5,
        "current": 38.0,
        "rpm": 1100.0
    })
    assert severe_res["condition"] == "Critical"
    assert severe_res["probabilities"]["Critical"] > 0.8


def test_pump_temperature_guardrail_and_isolated_faults():
    """
    Verify Pump condition predictions for:
    - Normal readings produce Normal.
    - Temperature-only input at 80°C produces at least Warning.
    - Temperature at or above 85°C produces Critical.
    - Low pressure or abnormal flow is tested independently.
    - Severe combined fault remains Critical.
    """
    # 1. Normal readings produce Normal
    norm_res = pump_predictor.predict({
        "temperature": 48.0,
        "vibration": 1.5,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 84.0
    })
    assert norm_res["condition"] == "Normal"
    assert norm_res["model_available"] is True

    # 2. Temperature-only at 80°C produces at least Warning
    temp80_res = pump_predictor.predict({
        "temperature": 80.0,
        "vibration": 1.5,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 84.0
    })
    assert temp80_res["condition"] in ["Warning", "Critical"]
    assert temp80_res["condition"] != "Normal"
    assert temp80_res["guardrail_active"] is True

    # 3. Temperature at or above 85°C produces Critical
    temp85_res = pump_predictor.predict({
        "temperature": 85.0,
        "vibration": 1.5,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 84.0
    })
    assert temp85_res["condition"] == "Critical"
    assert temp85_res["guardrail_active"] is True

    temp92_res = pump_predictor.predict({
        "temperature": 92.0,
        "vibration": 1.5,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 84.0
    })
    assert temp92_res["condition"] == "Critical"

    # 4. Low pressure tested independently (e.g. 2.0 Bar suction drop with normal temp/vib/current/rpm/flow)
    press_res = pump_predictor.predict({
        "temperature": 48.0,
        "vibration": 1.5,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 2.2,
        "flow_rate": 84.0
    })
    assert press_res["condition"] in ["Warning", "Critical"]

    # 5. Abnormal flow tested independently (e.g. 45 m³/h restriction with normal temp/vib/current/rpm/pressure)
    flow_res = pump_predictor.predict({
        "temperature": 48.0,
        "vibration": 1.5,
        "current": 13.0,
        "rpm": 1485.0,
        "pressure": 4.6,
        "flow_rate": 45.0
    })
    assert flow_res["condition"] in ["Warning", "Critical"]

    # 6. Severe combined fault remains Critical
    severe_res = pump_predictor.predict({
        "temperature": 95.0,
        "vibration": 10.0,
        "current": 3.5,
        "rpm": 1250.0,
        "pressure": 0.3,
        "flow_rate": 5.0
    })
    assert severe_res["condition"] == "Critical"
    assert severe_res["probabilities"]["Critical"] > 0.8


def test_integration_temperature_telemetry_and_twin_contract():
    """
    Verify End-to-End API Integration:
    - Valid POST /telemetry with 80°C updates state and refreshes predictions.
    - Temperature-only fault cannot be reported as Normal in effective prediction.
    - /twin returns expected schema and fields (including model_available, status, health).
    - Missing or malformed telemetry is handled safely.
    """
    client = TestClient(app)

    # 1. Ingest motor temperature at 80°C
    telemetry_80c = {
        "motor": {
            "temperature": 80.0,
            "vibration": 1.6,
            "current": 13.8,
            "rpm": 1485.0
        }
    }
    r_post = client.post("/telemetry", json=telemetry_80c)
    assert r_post.status_code == 200
    res = r_post.json()

    # Temperature-only at 80°C cannot be reported as Normal
    assert res["motor"]["predicted_condition"] != "Normal"
    assert res["motor"]["predicted_condition"] in ["Warning", "Critical"]
    assert res["motor"]["status"] in ["WARNING", "CRITICAL"]
    assert res["motor"]["health"] < 90.0
    assert res["motor"]["guardrail_active"] is True
    assert res["motor"]["prediction_source"] is not None

    # 2. Check /twin returns matching snapshot
    r_twin = client.get("/twin")
    assert r_twin.status_code == 200
    twin_data = r_twin.json()
    assert twin_data["motor"]["predicted_condition"] == res["motor"]["predicted_condition"]
    assert twin_data["motor"]["model_available"] is True
    assert twin_data["pump"]["model_available"] is True
    assert "system_status" in twin_data

    # 3. Critical temperature 92°C on pump
    pump_crit = {
        "pump": {
            "temperature": 92.0,
            "vibration": 1.5,
            "current": 13.0,
            "rpm": 1485.0,
            "pressure": 4.6,
            "flow_rate": 84.0
        }
    }
    r_crit = client.post("/telemetry", json=pump_crit)
    assert r_crit.status_code == 200
    res_crit = r_crit.json()
    assert res_crit["pump"]["predicted_condition"] == "Critical"
    assert res_crit["pump"]["status"] == "CRITICAL"
    assert res_crit["system_status"] == "CRITICAL_ALERT"

    # 4. Error handling: empty payload (neither motor nor pump)
    r_empty = client.post("/telemetry", json={})
    assert r_empty.status_code == 400

    # 5. Error handling: malformed feature values
    r_bad = client.post("/telemetry", json={
        "motor": {
            "temperature": "not_a_number",
            "vibration": 1.5,
            "current": 13.0,
            "rpm": 1485.0
        }
    })
    assert r_bad.status_code == 422


def test_idle_standby_and_critical_temperature_override():
    """
    Verify:
    1. At 32°C, 0 RPM/current, low vibration, 0 pressure/flow: returns healthy IDLE.
    2. At or above critical temperature (motor 90°C, pump 85°C), critical alert overrides idle logic.
    3. Both predictors and digital twin state reflect latest conditions consistently.
    """
    twin = DigitalTwinState()

    # 1. Healthy idle at 32°C (Wokwi standby baseline)
    payload_idle_32c = TelemetryPayload(
        motor=MotorTelemetry(temperature=32.0, vibration=0.08, current=0.0, rpm=0.0),
        pump=PumpTelemetry(temperature=32.0, vibration=0.08, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    )
    snap = twin.update_telemetry(payload_idle_32c)
    assert snap.motor.status == "IDLE"
    assert snap.motor.predicted_condition == "Normal"
    assert snap.motor.health == 100.0
    assert snap.motor.guardrail_active is False
    assert snap.pump.status == "IDLE"
    assert snap.pump.predicted_condition == "Normal"
    assert snap.pump.health == 100.0
    assert snap.pump.guardrail_active is False
    assert snap.system_status == "IDLE"

    # 2. Motor critical temperature (92°C >= 90°C) while idle
    payload_motor_crit_idle = TelemetryPayload(
        motor=MotorTelemetry(temperature=92.0, vibration=0.08, current=0.0, rpm=0.0),
        pump=PumpTelemetry(temperature=32.0, vibration=0.08, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    )
    snap_m_crit = twin.update_telemetry(payload_motor_crit_idle)
    assert snap_m_crit.motor.status == "CRITICAL"
    assert snap_m_crit.motor.predicted_condition == "Critical"
    assert snap_m_crit.motor.health == 35.0
    assert snap_m_crit.motor.guardrail_active is True
    assert "Critical" in snap_m_crit.motor.prediction_details
    assert snap_m_crit.pump.status == "IDLE"
    assert snap_m_crit.system_status == "CRITICAL_ALERT"

    # 3. Pump critical temperature (88°C >= 85°C) while idle
    payload_pump_crit_idle = TelemetryPayload(
        motor=MotorTelemetry(temperature=32.0, vibration=0.08, current=0.0, rpm=0.0),
        pump=PumpTelemetry(temperature=88.0, vibration=0.08, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    )
    snap_p_crit = twin.update_telemetry(payload_pump_crit_idle)
    assert snap_p_crit.pump.status == "CRITICAL"
    assert snap_p_crit.pump.predicted_condition == "Critical"
    assert snap_p_crit.pump.health == 30.0
    assert snap_p_crit.pump.guardrail_active is True
    assert snap_p_crit.motor.status == "IDLE"
    assert snap_p_crit.system_status == "CRITICAL_ALERT"


def test_temperature_and_fault_recovery_workflows():
    """
    Verify:
    1. Recovery from critical temperature back to healthy normal idle.
    2. Recovery from running warning (thermal or vibration) back to running normal.
    3. Status, health, guardrails, and system status correctly update on recovery.
    """
    twin = DigitalTwinState()

    # 1. Critical state in idle
    crit_payload = TelemetryPayload(
        motor=MotorTelemetry(temperature=95.0, vibration=0.05, current=0.0, rpm=0.0),
        pump=PumpTelemetry(temperature=90.0, vibration=0.05, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    )
    snap_crit = twin.update_telemetry(crit_payload)
    assert snap_crit.system_status == "CRITICAL_ALERT"
    assert snap_crit.motor.guardrail_active is True
    assert snap_crit.pump.guardrail_active is True

    # Recover to normal 32°C idle
    recov_idle = TelemetryPayload(
        motor=MotorTelemetry(temperature=32.0, vibration=0.05, current=0.0, rpm=0.0),
        pump=PumpTelemetry(temperature=32.0, vibration=0.05, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    )
    snap_recov = twin.update_telemetry(recov_idle)
    assert snap_recov.motor.status == "IDLE"
    assert snap_recov.motor.predicted_condition == "Normal"
    assert snap_recov.motor.health == 100.0
    assert snap_recov.motor.guardrail_active is False
    assert snap_recov.pump.status == "IDLE"
    assert snap_recov.pump.predicted_condition == "Normal"
    assert snap_recov.pump.health == 100.0
    assert snap_recov.pump.guardrail_active is False
    assert snap_recov.system_status == "IDLE"

    # 2. Running Warning condition (motor temperature 78.5°C)
    running_warn = TelemetryPayload(
        motor=MotorTelemetry(temperature=78.5, vibration=1.6, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=48.0, vibration=1.4, current=12.0, rpm=1485.0, pressure=4.5, flow_rate=82.0)
    )
    snap_warn = twin.update_telemetry(running_warn)
    assert snap_warn.motor.status == "WARNING"
    assert snap_warn.motor.predicted_condition == "Warning"
    assert snap_warn.motor.guardrail_active is True
    assert snap_warn.pump.status == "RUNNING"
    assert snap_warn.system_status == "WARNING"

    # Recover to nominal running baseline (motor temperature 56.0°C)
    running_normal = TelemetryPayload(
        motor=MotorTelemetry(temperature=56.0, vibration=1.6, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=48.0, vibration=1.4, current=12.0, rpm=1485.0, pressure=4.5, flow_rate=82.0)
    )
    snap_norm = twin.update_telemetry(running_normal)
    assert snap_norm.motor.status == "RUNNING"
    assert snap_norm.motor.predicted_condition == "Normal"
    assert snap_norm.motor.guardrail_active is False
    assert snap_norm.motor.health > 95.0
    assert snap_norm.pump.status == "RUNNING"
    assert snap_norm.system_status == "OPERATIONAL"


def test_system_status_aggregation_matrix():
    """Verify system_status correctly aggregates various combinations of motor and pump states."""
    twin = DigitalTwinState()

    # Initialized
    assert twin.system_status == "INITIALIZED"

    # Both idle -> IDLE
    snap1 = twin.update_telemetry(TelemetryPayload(
        motor=MotorTelemetry(temperature=28.0, vibration=0.01, current=0.0, rpm=0.0),
        pump=PumpTelemetry(temperature=28.0, vibration=0.01, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    ))
    assert snap1.system_status == "IDLE"

    # One running, one idle -> OPERATIONAL
    snap2 = twin.update_telemetry(TelemetryPayload(
        motor=MotorTelemetry(temperature=55.0, vibration=1.5, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=28.0, vibration=0.01, current=0.0, rpm=0.0, pressure=0.0, flow_rate=0.0)
    ))
    assert snap2.system_status == "OPERATIONAL"

    # One warning, one running -> WARNING
    snap3 = twin.update_telemetry(TelemetryPayload(
        motor=MotorTelemetry(temperature=78.0, vibration=1.5, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=48.0, vibration=1.4, current=12.0, rpm=1485.0, pressure=4.5, flow_rate=82.0)
    ))
    assert snap3.system_status == "WARNING"

    # One critical, one running -> CRITICAL_ALERT
    snap4 = twin.update_telemetry(TelemetryPayload(
        motor=MotorTelemetry(temperature=55.0, vibration=1.5, current=14.0, rpm=1485.0),
        pump=PumpTelemetry(temperature=92.0, vibration=1.4, current=12.0, rpm=1485.0, pressure=4.5, flow_rate=82.0)
    ))
    assert snap4.system_status == "CRITICAL_ALERT"


def test_invalid_telemetry_and_state_preservation():
    """Verify that invalid telemetry returns 422 and does NOT corrupt previous valid state."""
    client = TestClient(app)

    # 1. Establish valid baseline
    r_valid = client.post("/telemetry", json={
        "motor": {
            "temperature": 55.0,
            "vibration": 1.5,
            "current": 14.0,
            "rpm": 1485.0
        }
    })
    assert r_valid.status_code == 200
    valid_state = r_valid.json()
    assert valid_state["motor"]["telemetry"]["temperature"] == 55.0

    # 2. Attempt invalid physical temperature (> 300°C)
    r_invalid_temp = client.post("/telemetry", json={
        "motor": {
            "temperature": 550.0,
            "vibration": 1.5,
            "current": 14.0,
            "rpm": 1485.0
        }
    })
    assert r_invalid_temp.status_code == 422

    # 3. Verify previous state remains intact
    r_twin = client.get("/twin")
    assert r_twin.status_code == 200
    current_state = r_twin.json()
    assert current_state["motor"]["telemetry"]["temperature"] == 55.0


def test_offline_model_rule_fallback_behavior():
    """Verify offline rule fallback behaves correctly for critical temperature, idle, and warning."""
    missing_motor = MotorPredictor(model_path=Path("non_existent_motor.joblib"))
    assert missing_motor.model_available is False

    # 1. Critical temperature while idle in offline fallback mode
    res_crit_idle = missing_motor.predict({
        "temperature": 95.0,
        "vibration": 0.05,
        "current": 0.0,
        "rpm": 0.0
    })
    assert res_crit_idle["condition"] == "Critical"
    assert res_crit_idle["guardrail_active"] is True
    assert res_crit_idle["model_available"] is False

    # 2. Normal idle at 32°C in offline fallback mode
    res_normal_idle = missing_motor.predict({
        "temperature": 32.0,
        "vibration": 0.05,
        "current": 0.0,
        "rpm": 0.0
    })
    assert res_normal_idle["condition"] == "Normal"
    assert res_normal_idle["guardrail_active"] is False
    assert res_normal_idle["prediction_source"] == "standby"

    # 3. Warning temperature while running in offline fallback mode
    res_warn_run = missing_motor.predict({
        "temperature": 75.0,
        "vibration": 1.5,
        "current": 14.0,
        "rpm": 1485.0
    })
    assert res_warn_run["condition"] == "Warning"
    assert res_warn_run["guardrail_active"] is True


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__).resolve())])


