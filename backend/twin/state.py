"""
Digital Twin State Management for Industrial Pump and Motor System.

This module encapsulates:
- Pydantic schemas for Motor and Pump telemetry
- Digital Twin in-memory state tracking
- Integration of Motor and Pump ML condition classification models
- Graceful fallback when ML models are missing or unavailable
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from ml.motor_model.predict import MOTOR_TEMP_CRITICAL_THRESHOLD, MOTOR_TEMP_WARNING_THRESHOLD
from ml.pump_model.predict import PUMP_TEMP_CRITICAL_THRESHOLD, PUMP_TEMP_WARNING_THRESHOLD

from ml.motor_model.predict import motor_predictor
from ml.pump_model.predict import pump_predictor
from backend.twin.diagnostics import (
    diagnostic_engine,
    ComponentDiagnosticsResult
)


class MotorTelemetry(BaseModel):
    temperature: float = Field(..., description="Motor temperature in °C")
    vibration: float = Field(..., description="Motor vibration velocity in mm/s (RMS)")
    current: float = Field(..., description="Motor current in Amperes")
    rpm: float = Field(..., description="Motor speed in RPM")


class PumpTelemetry(BaseModel):
    temperature: float = Field(..., description="Pump bearing/housing temperature in °C")
    vibration: float = Field(..., description="Pump vibration velocity in mm/s")
    current: float = Field(..., description="Pump electrical current draw in Amperes")
    rpm: float = Field(..., description="Pump operating speed in RPM")
    pressure: float = Field(..., description="Discharge pressure in Bar")
    flow_rate: float = Field(..., description="Discharge flow rate in m³/h")


class TelemetryPayload(BaseModel):
    motor: Optional[MotorTelemetry] = None
    pump: Optional[PumpTelemetry] = None


class MotorState(BaseModel):
    telemetry: Optional[MotorTelemetry] = None
    health: float = Field(default=100.0, description="Health percentage (0-100%)")
    status: str = Field(default="IDLE", description="Operational status: IDLE, RUNNING, WARNING, CRITICAL")
    predicted_condition: Optional[str] = Field(default=None, description="Effective Condition: Normal, Warning, Critical")
    raw_ml_condition: Optional[str] = Field(default=None, description="Raw ML model prediction prior to guardrail evaluation")
    prediction_source: Optional[str] = Field(default=None, description="Source of effective prediction: ml_model, temperature_guardrail, ml_model_and_guardrail, rule_fallback")
    guardrail_active: bool = Field(default=False, description="Whether a safety guardrail was triggered")
    model_available: bool = Field(default=False, description="Whether ML inference model is loaded and available")
    prediction_confidence: Optional[float] = Field(default=None, description="Confidence score of predicted condition (0.0-1.0)")
    prediction_details: Optional[str] = Field(default=None, description="Diagnostic details or fallback status")
    last_updated: Optional[datetime] = None


class PumpState(BaseModel):
    telemetry: Optional[PumpTelemetry] = None
    health: float = Field(default=100.0, description="Health percentage (0-100%)")
    status: str = Field(default="IDLE", description="Operational status: IDLE, RUNNING, WARNING, CRITICAL")
    predicted_condition: Optional[str] = Field(default=None, description="Effective Condition: Normal, Warning, Critical")
    raw_ml_condition: Optional[str] = Field(default=None, description="Raw ML model prediction prior to guardrail evaluation")
    prediction_source: Optional[str] = Field(default=None, description="Source of effective prediction: ml_model, temperature_guardrail, ml_model_and_guardrail, rule_fallback")
    guardrail_active: bool = Field(default=False, description="Whether a safety guardrail was triggered")
    model_available: bool = Field(default=False, description="Whether ML inference model is loaded and available")
    prediction_confidence: Optional[float] = Field(default=None, description="Confidence score of predicted condition (0.0-1.0)")
    prediction_details: Optional[str] = Field(default=None, description="Diagnostic details or fallback status")
    last_updated: Optional[datetime] = None


class DigitalTwinSnapshot(BaseModel):
    system_status: str
    last_updated: Optional[datetime]
    data_source: str = Field(default="initialized", description="Source of telemetry: 'live_sensor', 'simulated', or 'initialized'")
    motor: MotorState
    pump: PumpState
    diagnostics: Optional[ComponentDiagnosticsResult] = Field(
        default=None,
        description="Explainable component-level fault diagnostics and 3D highlight mappings"
    )


def _evaluate_motor_telemetry(telemetry: MotorTelemetry) -> tuple[float, str, str, bool, Optional[float], str, Optional[str], Optional[str], bool]:
    """
    Evaluates motor condition using the trained scikit-learn model with safety guardrails and graceful fallback.
    Returns: (health, status, condition, model_available, confidence, details, raw_ml_cond, source, gr_active)
    """
    pred = motor_predictor.predict(telemetry)
    condition = pred["condition"]
    model_available = pred["model_available"]
    confidence = pred.get("confidence")
    details = pred.get("details", "")
    raw_ml_cond = pred.get("raw_ml_condition")
    source = pred.get("prediction_source", "ml_model")
    gr_active = pred.get("guardrail_active", False)

    if condition == "Critical":
        status = "CRITICAL"
        health = 35.0
    elif condition == "Warning":
        status = "WARNING"
        health = 72.0
    else:
        if telemetry.rpm < 50.0:
            status = "IDLE"
            health = 100.0
        else:
            status = "RUNNING"
            health = 98.5

    return health, status, condition, model_available, confidence, details, raw_ml_cond, source, gr_active


def _evaluate_pump_telemetry(telemetry: PumpTelemetry) -> tuple[float, str, str, bool, Optional[float], str, Optional[str], Optional[str], bool]:
    """
    Evaluates pump condition using the trained scikit-learn model with safety guardrails and graceful fallback.
    Returns: (health, status, condition, model_available, confidence, details, raw_ml_cond, source, gr_active)
    """
    pred = pump_predictor.predict(telemetry)
    condition = pred["condition"]
    model_available = pred["model_available"]
    confidence = pred.get("confidence")
    details = pred.get("details", "")
    raw_ml_cond = pred.get("raw_ml_condition")
    source = pred.get("prediction_source", "ml_model")
    gr_active = pred.get("guardrail_active", False)

    if condition == "Critical":
        status = "CRITICAL"
        health = 30.0
    elif condition == "Warning":
        status = "WARNING"
        health = 70.0
    else:
        if telemetry.rpm < 50.0:
            status = "IDLE"
            health = 100.0
        else:
            status = "RUNNING"
            health = 99.0

    return health, status, condition, model_available, confidence, details, raw_ml_cond, source, gr_active


class DigitalTwinState:
    """
    Authoritative in-memory state container for the pump and motor digital twin.
    Integrates ML inference models for predictive maintenance condition monitoring.
    """
    def __init__(self):
        self.motor = MotorState(model_available=motor_predictor.model_available)
        self.pump = PumpState(model_available=pump_predictor.model_available)
        self.system_status: str = "INITIALIZED"
        self.data_source: str = "initialized"
        self.last_updated: Optional[datetime] = None
        self.diagnostics: Optional[ComponentDiagnosticsResult] = diagnostic_engine.evaluate(
            motor_telem=None,
            pump_telem=None,
            last_updated=None,
            data_source=self.data_source
        )

    def update_telemetry(self, payload: TelemetryPayload, source: str = "live_sensor") -> "DigitalTwinSnapshot":
        now = datetime.now(timezone.utc)
        self.data_source = source

        if payload.motor is not None:
            health, status, condition, model_avail, conf, details, raw_cond, src, gr_act = _evaluate_motor_telemetry(payload.motor)
            self.motor.telemetry = payload.motor
            self.motor.health = health
            self.motor.status = status
            self.motor.predicted_condition = condition
            self.motor.raw_ml_condition = raw_cond
            self.motor.prediction_source = src
            self.motor.guardrail_active = gr_act
            self.motor.model_available = model_avail
            self.motor.prediction_confidence = conf
            self.motor.prediction_details = details
            self.motor.last_updated = now

        if payload.pump is not None:
            health, status, condition, model_avail, conf, details, raw_cond, src, gr_act = _evaluate_pump_telemetry(payload.pump)
            self.pump.telemetry = payload.pump
            self.pump.health = health
            self.pump.status = status
            self.pump.predicted_condition = condition
            self.pump.raw_ml_condition = raw_cond
            self.pump.prediction_source = src
            self.pump.guardrail_active = gr_act
            self.pump.model_available = model_avail
            self.pump.prediction_confidence = conf
            self.pump.prediction_details = details
            self.pump.last_updated = now

        # Aggregate overall system state
        if self.motor.status == "CRITICAL" or self.pump.status == "CRITICAL":
            self.system_status = "CRITICAL_ALERT"
        elif self.motor.status == "WARNING" or self.pump.status == "WARNING":
            self.system_status = "WARNING"
        elif self.motor.status == "RUNNING" or self.pump.status == "RUNNING":
            self.system_status = "OPERATIONAL"
        else:
            self.system_status = "IDLE"

        self.last_updated = now

        # Evaluate component-level diagnostics with explainable rules
        self.diagnostics = diagnostic_engine.evaluate(
            motor_telem=self.motor.telemetry,
            pump_telem=self.pump.telemetry,
            last_updated=self.last_updated,
            data_source=self.data_source
        )

        return self.get_snapshot()

    def get_snapshot(self) -> DigitalTwinSnapshot:
        # Keep model availability status up-to-date in snapshot
        if not self.motor.telemetry:
            self.motor.model_available = motor_predictor.model_available
        if not self.pump.telemetry:
            self.pump.model_available = pump_predictor.model_available

        if self.diagnostics is None:
            self.diagnostics = diagnostic_engine.evaluate(
                motor_telem=self.motor.telemetry,
                pump_telem=self.pump.telemetry,
                last_updated=self.last_updated,
                data_source=self.data_source
            )

        return DigitalTwinSnapshot(
            system_status=self.system_status,
            last_updated=self.last_updated,
            data_source=self.data_source,
            motor=self.motor,
            pump=self.pump,
            diagnostics=self.diagnostics
        )


# Global Digital Twin instance
digital_twin = DigitalTwinState()
