"""
Inference Module for Motor Condition Prediction.

Provides:
- Strict input validation against feature schema
- Model loading with lazy-init or manual reload
- Graceful fallback when model artifact is missing or corrupted
- Consistent feature ordering for inference
"""

import math
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import pandas as pd

MOTOR_FEATURE_NAMES = ["temperature", "vibration", "current", "rpm"]
MOTOR_CLASSES = ["Normal", "Warning", "Critical"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "motor_model.joblib"

# ==============================================================================
# PROVISIONAL DEMO THRESHOLDS
# DISCLAIMER: These thresholds are provisional demo values for algorithm
# validation in the digital twin simulation environment. They are NOT certified
# OEM, IEEE, NEMA, or ISO safety limits and must not be cited as real-world limits.
# ==============================================================================
MOTOR_TEMP_WARNING_THRESHOLD: float = 72.0   # °C -> at least Warning
MOTOR_TEMP_CRITICAL_THRESHOLD: float = 90.0  # °C -> Critical

CONDITION_SEVERITY: Dict[str, int] = {
    "Normal": 0,
    "Warning": 1,
    "Critical": 2
}


def _fallback_motor_rule(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Fallback rule-based assessment used when ML model artifact is unavailable.
    """
    temp = features.get("temperature", 0.0)
    vib = features.get("vibration", 0.0)
    curr = features.get("current", 0.0)
    rpm = features.get("rpm", 0.0)

    # Critical conditions override idle/standby logic
    is_temp_crit = temp >= MOTOR_TEMP_CRITICAL_THRESHOLD
    is_vib_crit = vib >= 7.1
    is_curr_crit = curr >= 28.0
    is_speed_crit = (rpm < 1300.0 and rpm >= 50.0)

    if is_temp_crit:
        return {
            "condition": "Critical",
            "confidence": 1.0,
            "probabilities": {"Normal": 0.02, "Warning": 0.08, "Critical": 0.90},
            "model_available": False,
            "details": f"Rule fallback: Critical temperature threshold exceeded ({temp:.1f}°C >= {MOTOR_TEMP_CRITICAL_THRESHOLD:.1f}°C).",
            "raw_ml_condition": None,
            "raw_ml_confidence": None,
            "prediction_source": "temperature_guardrail",
            "guardrail_active": True
        }

    if is_vib_crit or is_curr_crit or is_speed_crit:
        reason = "vibration (>= 7.1 mm/s)" if is_vib_crit else ("current (>= 28.0 A)" if is_curr_crit else "stall speed (< 1300 RPM)")
        return {
            "condition": "Critical",
            "confidence": 1.0,
            "probabilities": {"Normal": 0.02, "Warning": 0.08, "Critical": 0.90},
            "model_available": False,
            "details": f"Rule fallback: Critical fault detected ({reason}).",
            "raw_ml_condition": None,
            "raw_ml_confidence": None,
            "prediction_source": "rule_fallback",
            "guardrail_active": True
        }

    # Standby / Idle check (no critical fault active)
    if rpm < 50.0:
        return {
            "condition": "Normal",
            "confidence": 1.0,
            "probabilities": {"Normal": 1.0, "Warning": 0.0, "Critical": 0.0},
            "model_available": False,
            "details": "Motor is in standby/idle (speed < 50 RPM); rule fallback used (ML model offline).",
            "raw_ml_condition": None,
            "raw_ml_confidence": None,
            "prediction_source": "standby",
            "guardrail_active": False
        }

    # Running warning checks
    is_temp_warn = temp >= MOTOR_TEMP_WARNING_THRESHOLD
    is_vib_warn = vib >= 3.0
    is_curr_warn = curr >= 18.0

    if is_temp_warn:
        return {
            "condition": "Warning",
            "confidence": 0.85,
            "probabilities": {"Normal": 0.10, "Warning": 0.85, "Critical": 0.05},
            "model_available": False,
            "details": f"Rule fallback: Warning temperature threshold exceeded ({temp:.1f}°C >= {MOTOR_TEMP_WARNING_THRESHOLD:.1f}°C).",
            "raw_ml_condition": None,
            "raw_ml_confidence": None,
            "prediction_source": "temperature_guardrail",
            "guardrail_active": True
        }

    if is_vib_warn or is_curr_warn:
        reason = "vibration (>= 3.0 mm/s)" if is_vib_warn else "current (>= 18.0 A)"
        return {
            "condition": "Warning",
            "confidence": 0.80,
            "probabilities": {"Normal": 0.15, "Warning": 0.80, "Critical": 0.05},
            "model_available": False,
            "details": f"Rule fallback: Warning fault detected ({reason}).",
            "raw_ml_condition": None,
            "raw_ml_confidence": None,
            "prediction_source": "rule_fallback",
            "guardrail_active": True
        }

    return {
        "condition": "Normal",
        "confidence": 0.95,
        "probabilities": {"Normal": 0.95, "Warning": 0.04, "Critical": 0.01},
        "model_available": False,
        "details": "Rule-based fallback: operating within normal baseline (ML model offline).",
        "raw_ml_condition": None,
        "raw_ml_confidence": None,
        "prediction_source": "rule_fallback",
        "guardrail_active": False
    }


class MotorPredictor:
    """
    Inference predictor for the Motor Digital Twin.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.pipeline = None
        self.metadata = None
        self.model_available: bool = False
        self.load_model()

    def load_model(self) -> bool:
        """
        Attempts to load the serialized model artifact.
        Returns True if successfully loaded, False otherwise.
        """
        if not self.model_path.exists():
            self.model_available = False
            self.pipeline = None
            return False

        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                self.pipeline = bundle["pipeline"]
                self.metadata = bundle
            else:
                self.pipeline = bundle
                self.metadata = {}
            self.model_available = True
            return True
        except Exception as e:
            print(f"[MotorPredictor] Warning: Failed to load model from {self.model_path}: {e}")
            self.model_available = False
            self.pipeline = None
            return False

    def validate_input(self, data: Any) -> Dict[str, float]:
        """
        Validates telemetry input and returns an ordered dictionary of floats.
        Raises ValueError if fields are missing or invalid.
        """
        # Convert Pydantic model to dict if provided
        if hasattr(data, "model_dump"):
            raw = data.model_dump()
        elif hasattr(data, "dict"):
            raw = data.dict()
        elif isinstance(data, dict):
            raw = data
        else:
            raise ValueError(f"Expected dict or Pydantic object, got {type(data)}")

        cleaned = {}
        for feature in MOTOR_FEATURE_NAMES:
            if feature not in raw or raw[feature] is None:
                raise ValueError(f"Missing required motor feature: '{feature}'")
            try:
                val = float(raw[feature])
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feature}' must be a valid float, got: {raw[feature]}")

            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature '{feature}' must be finite, got: {val}")

            cleaned[feature] = val

        # Physical sanity checks
        if cleaned["rpm"] < 0:
            raise ValueError(f"Invalid motor RPM (cannot be negative): {cleaned['rpm']}")
        if cleaned["temperature"] < -50.0 or cleaned["temperature"] > 300.0:
            raise ValueError(f"Motor temperature out of physical sensor bounds: {cleaned['temperature']}")

        return cleaned

    def predict(self, telemetry: Any) -> Dict[str, Any]:
        """
        Predict motor condition (Normal, Warning, Critical) from telemetry.
        Falls back gracefully if the ML model is not available.
        """
        validated_features = self.validate_input(telemetry)

        # Check if model is available; if not, try reloading once
        if not self.model_available:
            self.load_model()

        if not self.model_available or self.pipeline is None:
            return _fallback_motor_rule(validated_features)

        temp = validated_features["temperature"]
        vib = validated_features["vibration"]
        curr = validated_features["current"]
        rpm = validated_features["rpm"]

        # 1. Critical safety guardrail evaluation (overrides standby/idle logic)
        is_temp_crit = temp >= MOTOR_TEMP_CRITICAL_THRESHOLD
        is_vib_crit = vib >= 7.1
        is_curr_crit = curr >= 28.0
        is_speed_crit = (rpm < 1300.0 and rpm >= 50.0)

        if is_temp_crit:
            return {
                "condition": "Critical",
                "confidence": 1.0,
                "probabilities": {"Normal": 0.0, "Warning": 0.0, "Critical": 1.0},
                "model_available": True,
                "details": (
                    f"Safety guardrail override (Critical): temperature {temp:.1f}°C "
                    f">= demo threshold ({MOTOR_TEMP_CRITICAL_THRESHOLD:.1f}°C)."
                ),
                "raw_ml_condition": "Critical",
                "raw_ml_confidence": 1.0,
                "prediction_source": "temperature_guardrail",
                "guardrail_active": True
            }

        if is_vib_crit or is_curr_crit or is_speed_crit:
            reason = "vibration >= 7.1 mm/s" if is_vib_crit else ("current >= 28.0 A" if is_curr_crit else "stall speed < 1300 RPM")
            source = "vibration_guardrail" if is_vib_crit else ("current_guardrail" if is_curr_crit else "speed_guardrail")
            return {
                "condition": "Critical",
                "confidence": 1.0,
                "probabilities": {"Normal": 0.0, "Warning": 0.0, "Critical": 1.0},
                "model_available": True,
                "details": f"Safety guardrail override (Critical): {reason}.",
                "raw_ml_condition": "Critical",
                "raw_ml_confidence": 1.0,
                "prediction_source": source,
                "guardrail_active": True
            }

        # 2. Standby / Idle check (when no critical fault is active)
        if rpm < 50.0:
            return {
                "condition": "Normal",
                "confidence": 1.0,
                "probabilities": {"Normal": 1.0, "Warning": 0.0, "Critical": 0.0},
                "model_available": True,
                "details": "Motor is in standby/idle (speed < 50 RPM).",
                "raw_ml_condition": "Normal",
                "raw_ml_confidence": 1.0,
                "prediction_source": "standby",
                "guardrail_active": False
            }

        # 3. Running Operation: Scikit-learn ML Model Inference
        try:
            input_df = pd.DataFrame([validated_features], columns=MOTOR_FEATURE_NAMES)
            prediction = self.pipeline.predict(input_df)[0]

            probabilities = None
            confidence = 1.0
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba(input_df)[0]
                classes = list(self.pipeline.classes_)
                probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
                if prediction in probabilities:
                    confidence = probabilities[prediction]

            ml_cond = str(prediction)
            ml_conf = round(float(confidence), 4)

            # Post-inference warning guardrails
            guardrail_condition = "Normal"
            guardrail_triggered = False
            gr_source = "ml_model"
            gr_reason = ""

            if temp >= MOTOR_TEMP_WARNING_THRESHOLD:
                guardrail_condition = "Warning"
                guardrail_triggered = True
                gr_source = "temperature_guardrail"
                gr_reason = f"temperature {temp:.1f}°C >= warning threshold ({MOTOR_TEMP_WARNING_THRESHOLD:.1f}°C)"
            elif vib >= 3.0:
                guardrail_condition = "Warning"
                guardrail_triggered = True
                gr_source = "vibration_guardrail"
                gr_reason = f"vibration {vib:.2f} mm/s >= ISO warning threshold (3.0 mm/s)"
            elif curr >= 18.0:
                guardrail_condition = "Warning"
                guardrail_triggered = True
                gr_source = "current_guardrail"
                gr_reason = f"current {curr:.1f} A >= electrical warning threshold (18.0 A)"

            ml_sev = CONDITION_SEVERITY.get(ml_cond, 0)
            gr_sev = CONDITION_SEVERITY.get(guardrail_condition, 0)

            # Final condition is the more severe of ML prediction and guardrail
            if gr_sev > ml_sev:
                effective_condition = guardrail_condition
                prediction_source = gr_source
                details = f"Safety guardrail override ({guardrail_condition}): {gr_reason}; raw ML model predicted {ml_cond}."
            elif gr_sev == ml_sev and guardrail_triggered:
                effective_condition = ml_cond
                prediction_source = "ml_model_and_guardrail"
                details = f"ML model and safety guardrail concordant ({effective_condition}): {gr_reason}."
            else:
                effective_condition = ml_cond
                prediction_source = "ml_model"
                details = f"ML inference executed via scikit-learn model ({effective_condition})."

            return {
                "condition": effective_condition,
                "confidence": ml_conf,
                "probabilities": probabilities,
                "model_available": True,
                "details": details,
                "raw_ml_condition": ml_cond,
                "raw_ml_confidence": ml_conf,
                "prediction_source": prediction_source,
                "guardrail_active": guardrail_triggered
            }
        except Exception as e:
            fallback = _fallback_motor_rule(validated_features)
            fallback["details"] = f"Inference exception ({e}); fell back to rule."
            return fallback


# Default shared instance
motor_predictor = MotorPredictor()


if __name__ == "__main__":
    predictor = MotorPredictor()
    test_sample = {"temperature": 62.0, "vibration": 1.5, "current": 13.5, "rpm": 1480.0}
    print("Testing Motor Predictor:")
    res = predictor.predict(test_sample)
    print(res)
