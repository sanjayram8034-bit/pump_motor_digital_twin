"""
FastAPI Backend for Industrial Pump and Motor Digital Twin.

Endpoints:
- POST /telemetry : Receive motor and/or pump telemetry, updates twin state.
- GET  /twin      : Retrieve complete current state of the twin.
- GET  /          : Serves the industrial web dashboard.
"""

import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure root package directory is in sys.path for direct script execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.twin.state import (
    digital_twin,
    TelemetryPayload,
    DigitalTwinSnapshot
)
from backend.twin.simulation import generate_coupled_telemetry

app = FastAPI(
    title="Pump & Motor Digital Twin API",
    description="Edge & Cloud Digital Twin service for Industrial Predictive Maintenance",
    version="1.0.0"
)

# Enable CORS for local testing, frontend browser dashboards, and remote gateways
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_FILE = BASE_DIR / "frontend" / "dashboard.html"


@app.get("/", tags=["Dashboard"])
def serve_dashboard():
    """Serves the industrial digital twin web dashboard."""
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    return {"message": "Digital Twin API Running. Dashboard file not found at " + str(FRONTEND_FILE)}


@app.post("/telemetry", response_model=DigitalTwinSnapshot, tags=["Digital Twin"])
def post_telemetry(payload: TelemetryPayload):
    """
    Ingest live telemetry from Motor and/or Pump sensors (e.g. ESP32, PLC, MQTT gateway).
    Updates in-memory digital twin state and evaluates baseline health/status.
    """
    if payload.motor is None and payload.pump is None:
        raise HTTPException(status_code=400, detail="Payload must contain at least motor or pump telemetry.")
    
    try:
        updated_state = digital_twin.update_telemetry(payload, source="live_sensor")
        return updated_state
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/telemetry/simulate", response_model=DigitalTwinSnapshot, tags=["Digital Twin Simulator"])
def simulate_coupled_telemetry(scenario: str = "normal"):
    """
    Generates and ingests dynamically coupled pump-motor telemetry for the requested scenario:
    - 'normal': Resets both motor and pump to nominal rated baseline.
    - 'motor_warning': Motor thermal/vibration fault with derived pump Affinity-law derating.
    - 'pump_critical': Pump suction loss / dry-run with reduced motor electrical current.
    - 'idle': Stopped system (0 RPM, 0 current, 0 flow).
    """
    payload = generate_coupled_telemetry(scenario)
    return digital_twin.update_telemetry(payload, source="simulated")


@app.get("/telemetry/simulate", response_model=TelemetryPayload, tags=["Digital Twin Simulator"])
def inspect_simulated_payload(scenario: str = "normal"):
    """Returns the simulated coupled TelemetryPayload without mutating twin state."""
    return generate_coupled_telemetry(scenario)


@app.get("/twin", response_model=DigitalTwinSnapshot, tags=["Digital Twin"])
def get_twin_state():
    """
    Returns the complete, authoritative digital twin state:
    - Current motor telemetry, health, and status
    - Current pump telemetry, health, and status
    - Overall system status and last update timestamp
    """
    return digital_twin.get_snapshot()


if __name__ == "__main__":
    # Single clear command support: python backend/main.py
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
