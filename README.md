# Industrial Pump & Motor Digital Twin (Predictive Maintenance)

A modular Digital Twin platform for real-time condition monitoring and predictive maintenance of industrial electric induction motors and centrifugal pump systems. Powered by FastAPI, Scikit-Learn ML classification pipelines, and a deterministic dynamically coupled physics simulation model.

---

## 1. Architecture & Data Flow

```text
                                  +---------------------------------------+
                                  |         IoT / Edge Telemetry          |
                                  |  - ESP32 Firmware (Wokwi / PlatformIO)|
                                  |  - DS18B20 Temp, MPU6050 IMU, ADCs    |
                                  +-------------------+-------------------+
                                                      |
                                             HTTP POST /telemetry
                                                      |
                                                      v
+----------------------------------------------------------------------------------------------------+
|                                      FastAPI Backend (:8000)                                       |
|                                                                                                    |
|  [main.py]                                                                                         |
|    ├── POST /telemetry            -> Ingest live sensor telemetry (ESP32 / Gateway)                |
|    ├── POST /telemetry/simulate   -> Generate & ingest coupled simulation scenarios                |
|    ├── GET  /twin                 -> Authoritative digital twin state snapshot                     |
|    └── GET  /                     -> Serves Industrial SCADA Web Dashboard                         |
|                                                                                                    |
|  [twin/state.py] (Authoritative In-Memory Digital Twin State)                                       |
|    ├── State management & live/simulated telemetry tracking                                       |
|    ├── Aggregates System Status: IDLE | OPERATIONAL | WARNING | CRITICAL_ALERT                    |
|    └── Evaluates Component Health (0-100%) and Status (IDLE, RUNNING, WARNING, CRITICAL)           |
+-------------------+------------------------------------------------------------+-------------------+
                    |                                                            |
                    v                                                            v
+---------------------------------------+                    +---------------------------------------+
|          Motor ML Subsystem           |                    |          Pump ML Subsystem            |
|  [ml/motor_model/predict.py]          |                    |  [ml/pump_model/predict.py]           |
|                                       |                    |                                       |
|  - StandardScaler + RandomForest      |                    |  - StandardScaler + RandomForest      |
|  - Inputs: [temp, vib, curr, rpm]     |                    |  - Inputs: [temp, vib, curr, rpm,     |
|  - Standby detection (RPM < 50)       |                    |            pressure, flow_rate]       |
|  - Safety Guardrail Overrides:        |                    |  - Standby detection (RPM < 50)       |
|    * Critical Temp (>= 90.0°C)        |                    |  - Safety Guardrail Overrides:        |
|    * Critical Vib  (>= 7.1 mm/s)      |                    |    * Critical Temp (>= 85.0°C)        |
|    * Warning Temp  (>= 72.0°C)        |                    |    * Critical Vib  (>= 7.1 mm/s)      |
|  - Rule fallback if model offline     |                    |    * Suction/Dry Run (P<=0.8, Q<15)   |
|                                       |                    |    * Warning Temp  (>= 65.0°C)        |
+-------------------+-------------------+                    +-------------------+-------------------+
                    |                                                            |
                    +-----------------------------+------------------------------+
                                                  |
                                            Live State JSON
                                                  |
                                                  v
                               +---------------------------------------+
                               |     Industrial SCADA Web Console      |
                               |  - Vanilla HTML5 / CSS3 / ES6 JS      |
                               |  - 1.5s live polling from GET /twin   |
                               |  - Live Hardware vs Simulation badge  |
                               |  - Real-time diagnostic alert reasons |
                               +---------------------------------------+
```

---

## 2. API Endpoints & Telemetry Contract

### Telemetry Schemas

#### Motor Telemetry (`MotorTelemetry`)
| Field | Type | Unit | Description | Valid Physical Range |
|---|---|---|---|---|
| `temperature` | `float` | °C | Motor stator / casing temperature | -50.0 to 300.0 |
| `vibration` | `float` | mm/s | Mechanical vibration velocity (RMS) | >= 0.0 |
| `current` | `float` | A | Electrical current draw | >= 0.0 |
| `rpm` | `float` | RPM | Shaft rotation speed | >= 0.0 |

#### Pump Telemetry (`PumpTelemetry`)
| Field | Type | Unit | Description | Valid Physical Range |
|---|---|---|---|---|
| `temperature` | `float` | °C | Pump bearing housing temperature | -50.0 to 300.0 |
| `vibration` | `float` | mm/s | Pump vibration velocity (RMS) | >= 0.0 |
| `current` | `float` | A | Equivalent electrical load current | >= 0.0 |
| `rpm` | `float` | RPM | Pump operating speed | >= 0.0 |
| `pressure` | `float` | Bar | Discharge differential pressure | -1.0 to 100.0 |
| `flow_rate` | `float` | m³/h | Volumetric discharge flow rate | 0.0 to 1000.0 |

### REST Endpoints

1. **`POST /telemetry`**
   - **Description**: Ingests live telemetry from sensors (ESP32, PLC, MQTT gateway).
   - **Body**: `{"motor": MotorTelemetry, "pump": PumpTelemetry}` (at least one component required).
   - **Response**: `DigitalTwinSnapshot` with updated health, condition, and diagnostic details.
   - **Error Handling**: Missing components return `400 Bad Request`; invalid types or out-of-physical-bounds values return `422 Unprocessable Entity` without corrupting previous valid state.

2. **`POST /telemetry/simulate?scenario={normal|motor_warning|pump_critical|idle}`**
   - **Description**: Ingests a deterministic coupled physics simulation scenario into the twin and sets `data_source: "simulated"`.
   - **Response**: Updated `DigitalTwinSnapshot`.

3. **`GET /telemetry/simulate?scenario={scenario}`**
   - **Description**: Inspects the simulated coupled telemetry payload without mutating twin state.

4. **`GET /twin`**
   - **Description**: Returns the authoritative in-memory state snapshot of the digital twin, including subsystem statuses, health scores, model availability, data source (`live_sensor`, `simulated`, or `initialized`), and detailed diagnostic reasons.

5. **`GET /`**
   - **Description**: Serves the single-page industrial SCADA web console.

---

## 3. Setup & Execution Commands

### Prerequisites
- Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.14)
- Pip

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Run Automated Verification Test Suite
Run the 19 automated tests covering generators, ML models, coupling physics, idle recovery, and API endpoints:
```bash
python -m pytest tests/test_twin_ml.py -v
```

### 3. Launch the Digital Twin Backend & Dashboard
Start the server:
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
*Or bind to localhost only:*
```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Access the Dashboard & Documentation
- **Industrial SCADA Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 4. Connecting the ESP32 (Wokwi Simulation)

The embedded firmware is located in `d:\EEp\pump_motor_wokwi_connected`.

### Firmware Overview
- **Microcontroller**: ESP32 DevKit C v4
- **Sensors**:
  - DS18B20 digital temperature sensor (GPIO 4 via OneWire)
  - MPU6050 6-axis IMU for vibration magnitude (I²C SDA=21, SCL=22)
  - Potentiometer 1 for simulated motor current (ADC GPIO 34)
  - Potentiometer 2 for simulated flow rate (ADC GPIO 35)
  - SSD1306 128×64 OLED display for edge readings
- **Transmission**: POSTs JSON telemetry to `http://host.wokwi.internal:8000/telemetry` every 1500 ms.

### Steps to Run Wokwi with the Local Backend:
1. Ensure the backend is running on port 8000:
   ```bash
   python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Start the **Wokwi IoT Gateway** on your machine. The gateway bridges Wokwi virtual WiFi (`Wokwi-GUEST`) to your host machine so `host.wokwi.internal:8000` routes directly to your FastAPI server.
3. Open `pump_motor_wokwi_connected` in Wokwi (either in VS Code with the Wokwi extension or on wokwi.com).
4. Start the simulation. Observe:
   - ESP32 connects to WiFi and begins streaming telemetry.
   - The web dashboard at `http://localhost:8000` switches its Telemetry Source badge to `LIVE SENSOR (ESP32/IoT)`.
   - Telemetry gauges and health metrics update dynamically.
   - Adjusting temperature or potentiometers in Wokwi triggers real-time condition changes on the dashboard.

---

## 5. Machine Learning Models & Provisional Guardrails

### Model Architecture
- **Motor Model**: `StandardScaler` + `RandomForestClassifier` (100 estimators, balanced class weights).
  - Feature columns: `["temperature", "vibration", "current", "rpm"]`
- **Pump Model**: `StandardScaler` + `RandomForestClassifier` (100 estimators, balanced class weights).
  - Feature columns: `["temperature", "vibration", "current", "rpm", "pressure", "flow_rate"]`

### Retraining (Optional)
The pre-trained model artifacts are serialized in `ml/motor_model/motor_model.joblib` and `ml/pump_model/pump_model.joblib`. To retrain from synthetic generators:
```bash
python ml/motor_model/train.py
python ml/pump_model/train.py
```

### Provisional Demo Thresholds
> **[DISCLAIMER]**
> All thresholds below are provisional demo parameters calibrated for software demonstration and fault injection testing in this MVP. They are **NOT certified OEM, IEEE, NEMA, Hydraulic Institute, or ISO safety limits** and must not be used for life-critical or industrial protection systems without empirical calibration on physical hardware.

| Parameter | Subsystem | Warning Threshold | Critical Threshold | Rationale in Demo |
|---|---|---|---|---|
| **Temperature** | Motor | `>= 72.0 °C` | `>= 90.0 °C` | Stator winding thermal insulation margin |
| **Temperature** | Pump | `>= 65.0 °C` | `>= 85.0 °C` | Bearing housing and mechanical seal limits |
| **Vibration** | Both | `>= 3.0 mm/s` | `>= 7.1 mm/s` | ISO 10816 Zone C (restricted) & Zone D (damage danger) |
| **Current** | Both | `>= 18.0 A` | `>= 28.0 A` | Electrical overload and near locked-rotor protection |
| **Discharge Pressure** | Pump (Running) | `<= 2.5` or `>= 6.5 Bar` | `<= 0.8` or `>= 8.5 Bar` | Suction loss, cavitation, or discharge deadheading |
| **Flow Rate** | Pump (Running) | `<= 50.0 m³/h` | `< 15.0 m³/h` | Delivery collapse and vapor-lock cavitation |
| **Shaft Speed** | Both | N/A | `< 50 RPM = Standby/Idle` | Distinguishes stopped/idle from operational states |

### Idle Standby & Critical Overrides
- **Normal Standby**: When shaft speed is `< 50 RPM` and parameters are nominal (e.g. ambient 32°C, 0 A, 0 Bar, 0 m³/h), both components report `IDLE` status, `Normal` condition, 100% health, and `guardrail_active: false`.
- **Critical Overrides**: If temperature exceeds the critical threshold (Motor >= 90°C, Pump >= 85°C) or severe vibration/current is detected while idle, critical safety guardrails override the idle state to report `CRITICAL` status and `CRITICAL_ALERT` system state.
- **Recovery**: When telemetry returns to normal ambient idle (e.g. 32°C), both components return to healthy `IDLE` state.

---

## 6. Verification Status: Software Verified vs. Hardware Required

### What Has Been Verified in This MVP
1. **End-to-End API Pipeline**: Ingestion via `POST /telemetry`, state retrieval via `GET /twin`, and scenario simulation via `POST /telemetry/simulate`.
2. **Idle Standby & Recovery**: Verified at 32°C with 0 RPM/current/flow/pressure that both motor and pump remain healthy IDLE (health 100%, guardrails inactive).
3. **Critical Alert Overrides**: Verified that temperatures >= 90°C (motor) and >= 85°C (pump) override idle and trigger critical alerts.
4. **Recovery Workflows**: Verified recovery from critical temperature to normal idle, and recovery from running warning to running normal.
5. **Coupled Physics Simulation**: Verified speed identity ($RPM_{pump} = RPM_{motor}$), Affinity Laws ($Q \propto N, P \propto N^2$), motor current unloading under pump fluid starvation (~6.1 A), and structural vibration transmission.
6. **Error Resilience & State Preservation**: Verified that invalid payloads (missing components, malformed types, or out-of-physical-bounds data) return 400/422 without corrupting existing digital twin state.
7. **Offline Fallback**: Verified that missing or corrupted `.joblib` model artifacts trigger deterministic rule-based fallbacks with full guardrail protection.
8. **Dashboard SCADA Console**: Verified live polling, live vs. simulated source indicator, health bars, and diagnostic alert reasons.

### What Requires Physical Hardware & Real-World Data
1. **Physical Sensor Calibration**: Conversion of raw MPU6050 IMU accelerometer readings to calibrated industrial vibration velocity RMS (mm/s) according to ISO 10816 standards requires physical shaker-table calibration.
2. **Real-World Fault Data**: Current models are trained on physics-grounded synthetic datasets. Production deployment requires training or fine-tuning on real motor stator current signature analysis (MCSA) and vibration spectrum data from physical test rigs.
3. **Actual Thermal Time-Constants**: Physical motor and pump cooldown curves depend on physical ambient convective and conductive heat dissipation coefficients.
