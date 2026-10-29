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
Run the 35 automated verification tests covering synthetic data generators, ML models, physical coupling, idle recovery, API endpoints, explainable component diagnostics, sensor staleness, and 3D component mappings:
```bash
python -m pytest tests/ -v
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
- **Industrial SCADA Dashboard with 3D Visual Twin**: [http://localhost:8000](http://localhost:8000)
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
   - The web dashboard at `http://localhost:8000` switches its Telemetry Source badge to `LIVE_SENSOR`.
   - Telemetry gauges, 3D highlights, and component diagnostics update dynamically.
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
> All thresholds below are provisional demo parameters calibrated for software demonstration and fault injection testing in this digital twin. They are **NOT certified OEM, IEEE, NEMA, Hydraulic Institute, or ISO safety limits** and must not be used for life-critical or industrial protection systems without empirical calibration on physical hardware.

| Parameter | Subsystem | Warning Threshold | Critical Threshold | Rationale in Demo |
|---|---|---|---|---|
| **Temperature** | Motor | `>= 72.0 °C` | `>= 90.0 °C` | Stator winding thermal insulation margin |
| **Temperature** | Pump | `>= 65.0 °C` | `>= 85.0 °C` | Bearing housing and mechanical seal limits |
| **Vibration** | Both | `>= 3.0 mm/s` | `>= 7.1 mm/s` | ISO 10816 Zone C (restricted) & Zone D (damage danger) |
| **Current** | Both | `>= 18.0 A` | `>= 28.0 A` | Electrical overload and near locked-rotor protection |
| **Discharge Pressure** | Pump (Running) | `<= 2.5` or `>= 6.5 Bar` | `<= 0.8` or `>= 8.5 Bar` | Suction loss, cavitation, or discharge deadheading |
| **Flow Rate** | Pump (Running) | `<= 50.0 m³/h` | `< 15.0 m³/h` | Delivery collapse and vapor-lock cavitation |
| **Shaft Speed** | Both | N/A | `< 50 RPM = Standby/Idle` | Distinguishes stopped/idle from operational states |

---

## 6. Interactive 3D Visual Twin

The web dashboard incorporates a Three.js WebGL 3D physical model of the coupled motor-pump skid directly into the industrial SCADA interface.

### 3D Model Components
#### Induction Motor (`motor`)
1. **`motor_housing`**: Heavy ribbed cast-iron stator frame with 14 longitudinal cooling fins, junction terminal box, end bell covers, and mounting feet.
2. **`de_bearing`**: Drive-End heavy-duty roller bearing assembly supporting the shaft near the flexible coupling.
3. **`nde_bearing`**: Non-Drive-End roller bearing positioned at the rear bell near the cooling fan.
4. **`motor_shaft`**: Central steel drive shaft transmitting torque through the flexible coupling.
5. **`motor_rotor`**: Induction squirrel-cage rotor core with conductive bars and laminated steel rings.
6. **`stator_windings`**: Concentric copper electromagnetic stator winding coils and slot laminations.

#### Centrifugal Pump (`pump`)
1. **`pump_casing`**: Centrifugal spiral volute casing with axial suction nozzle (inlet) and vertical discharge nozzle (outlet).
2. **`pump_impeller`**: 6-vane backward-curved bronze centrifugal impeller mounted inside the volute.
3. **`pump_shaft`**: Stainless steel pump drive shaft coupled to the motor shaft.
4. **`pump_bearings`**: Heavy-duty bearing pedestal frame with oil sight glass and twin rolling bearing assemblies.
5. **`mechanical_seal`**: Silicon-carbide mechanical seal cartridge preventing fluid leakage along the rotating shaft.

### Key Visual Features
- **Realistic Industrial Exterior**: High-contrast SCADA dark theme with metallic shaders and true-to-life component proportions.
- **Cutaway / Transparent Inspection Mode**: Toggling `Cutaway: ON` transitions exterior housings (`motor_housing` and `pump_casing`) to a semi-transparent holographic material (opacity 0.22), fully exposing internal windings, rotor core, bearings, mechanical seal, and impeller.
- **Dynamic Mechanical Rotation**: When running (RPM > 50), the motor shaft, rotor, pump shaft, and impeller rotate continuously at a speed proportional to telemetry RPM.
- **Real-Time Highlight Shader**: Components under Warning pulse with an amber emissive glow (`#F59E0B`), and components under Critical alert pulse with radiant red (`#EF4444`). Selected components glow electric cyan (`#06B6D4`).
- **Interactive Raycasting**: Hovering over any part displays a tooltip with component identification and live status; clicking a part selects it, focuses the camera, and opens the Deep-Dive Component Inspector.
- **Camera Presets & Orbit Controls**: Quick toggle buttons for `Coupled Skid`, `Motor View`, and `Pump View`, plus reset view and auto-rotate controls.
- **Graceful 2D Fallback**: If WebGL or Three.js CDN is blocked, the dashboard automatically activates an interactive SVG/Canvas cutaway schematic with full clickability and state synchronization.

---

## 7. Explainable Component Diagnostics Engine

The diagnostic engine (`backend/twin/diagnostics.py`) sits alongside the machine-level ML models. It consumes real-time telemetry and twin state to isolate suspected components using evidence-based physical correlation rules.

### Machine-Level Status vs. Component-Level Diagnosis
- **Machine ML Status**: Evaluates subsystem health (e.g. Pump is `WARNING` or Motor is `RUNNING`).
- **Component Diagnosis**: Physical fault isolation. A machine in `CRITICAL` status does not mean all its components are broken; the diagnostic engine pinpoints specific root causes (e.g. `mechanical_seal` running dry or `stator_windings` overheating) while keeping other parts marked `NORMAL`.
- **Ambiguity Preservation**: When available sensor evidence is insufficient to distinguish between candidate components (e.g. high casing vibration without localized temperature or spectral data), the engine reports `suspected_component: "unidentifiable"` with `Low` confidence and notes the missing measurements rather than guessing.

### Diagnostic Output Schema
Each diagnosis provides:
- `machine_type`: `"motor"`, `"pump"`, or `"system"`
- `suspected_component`: 3D component key (e.g. `"de_bearing"`, `"stator_windings"`, or `"unidentifiable"`)
- `component_name`: Human-readable label
- `fault_category`: Engineering fault definition
- `severity`: `"NORMAL"`, `"WARNING"`, or `"CRITICAL"`
- `confidence`: Qualitative rating (`"High"`, `"Moderate"`, `"Low"`, or `"Uncertain"`)
- `supporting_evidence`: Dictionary of observed sensor values vs. thresholds
- `explanation`: Engineering rationale explaining why the readings indicate this fault
- `missing_measurements`: Physical sensor limitations preventing absolute confirmation
- `recommended_action`: Prescribed maintenance inspection action

---

## 8. Demonstration Scenarios Guide

Trigger these scenarios via the dashboard buttons or HTTP `POST /telemetry/simulate?scenario={name}`:

| Scenario Key | Primary Injected Condition | Machine Status | Suspected Component | 3D Highlights | Supporting Evidence |
|---|---|---|---|---|---|
| `normal` | Rated BEP operation (1485 RPM, 4.8 Bar, 88.5 m³/h) | Motor: RUNNING<br>Pump: RUNNING | None (Nominal) | All NORMAL | All parameters within rated thresholds |
| `motor_overheat` | High stator temperature (94.5°C), high current (24.8A) | Motor: CRITICAL | `stator_windings` | Stator: CRITICAL<br>Housing: WARNING | Temp 94.5°C >= 90°C, Current 24.8A >= 22A |
| `motor_bearing` | Elevated vibration (5.8 mm/s) + mild temp (75.0°C) | Motor: WARNING | `de_bearing` | DE Bearing: WARNING<br>Shaft: WARNING | Vib 5.8 mm/s >= 3.0 mm/s, Temp 75°C >= 72°C |
| `motor_rotor_fault` | High vibration (6.5 mm/s), high slip (1385 RPM), 25.5A | Motor: CRITICAL | `motor_rotor` | Rotor: CRITICAL<br>Shaft: WARNING | RPM 1385 < 1440, Current 25.5A, Vib 6.5 mm/s |
| `motor_vib_ambiguous` | Vibration 4.8 mm/s, nominal current & temp | Motor: WARNING | `unidentifiable` | DE/NDE/Shaft: WARNING | Vib 4.8 mm/s, but no localized spectral/temp data |
| `pump_critical` | Suction loss: 0.3 Bar, 5 m³/h, 92°C, 9.5 mm/s | Pump: CRITICAL | `mechanical_seal` | Seal: CRITICAL<br>Impeller: CRITICAL | Pressure 0.3 Bar <= 0.8 Bar, Flow 5 < 15 m³/h |
| `pump_deadhead` | Discharge valve closed: 9.4 Bar, 3.8 m³/h | Pump: CRITICAL | `pump_casing` | Casing: CRITICAL<br>Impeller: WARNING | Pressure 9.4 Bar >= 8.5 Bar, Flow 3.8 < 15 m³/h |
| `pump_impeller` | Low pressure (2.4 Bar), low flow (42 m³/h) at 1485 RPM | Pump: WARNING | `pump_impeller` | Impeller: WARNING | P 2.4 < 3.2 Bar, Q 42 < 60 m³/h at rated RPM |
| `pump_bearing` | Pump vibration (6.2 mm/s), pump temp (78.5°C) | Pump: CRITICAL | `pump_bearings` | Bearings: CRITICAL<br>Shaft: WARNING | Vib 6.2 mm/s >= 3.0 mm/s, Temp 78.5°C >= 65°C |
| `idle` | Stopped system: 0 RPM, 0 A, 0 Bar, 0 m³/h, 26°C | Motor: IDLE<br>Pump: IDLE | None (Standby) | All NORMAL | RPM 0 < 50, all idle readings nominal |

---

## 9. Real, Simulated, and Estimated Telemetry Breakdown

To maintain scientific integrity during project review, the data provenance is clearly distinguished:

| Parameter | Wokwi ESP32 Hardware Stream | Backend Coupled Simulator | Digital Twin Estimation |
|---|---|---|---|
| **Motor Temperature** | **Real Sensor**: DS18B20 1-Wire probe | Deterministic thermal model | Direct sensor reading (°C) |
| **Vibration** | **Real Sensor**: MPU6050 Accelerometer magnitude ($\|a\| - g$) | Transmissibility coupling ($RMS_{base} + r \cdot \Delta RMS$) | RMS mechanical velocity (mm/s) |
| **Motor Current** | **Simulated Input**: ADC Potentiometer (GPIO 34) | Induction load/no-load model | Equivalent current draw (A) |
| **Discharge Flow Rate** | **Simulated Input**: ADC Potentiometer (GPIO 35) | Affinity Law derivation ($Q \propto N$) | Volumetric flow ($m^3/h$) |
| **Shaft Speed (RPM)** | **Estimated/Derived**: Correlated with Current & Flow | Speed identity ($RPM_p = RPM_m$) | Operating speed (RPM) |
| **Discharge Pressure** | **Estimated/Derived**: Function of Flow ADC | Affinity Law derivation ($P \propto N^2$) | Discharge head (Bar) |
| **Component Fault Isolation** | N/A (Edge device sends raw telemetry) | N/A | **Evidence-based rule correlation** |

---

## 10. Current Limitations & Industrial Validation Roadmap

### Current Limitations
1. **Single-Point Casing Sensing**: Physical hardware uses one accelerometer and one temperature sensor. In real industrial machines, individual triaxial accelerometers are mounted on DE and NDE bearing housings.
2. **Lack of Spectral Analysis**: The firmware computes scalar vibration magnitude. Distinguishing shaft unbalance (1X), misalignment (2X), and bearing race defects (BPFO, BPFI) requires continuous Fast Fourier Transform (FFT) frequency spectrum analysis.
3. **Synthetic Training Baseline**: The machine-level ML models are trained on synthetic data generated from physical bounds. While mathematically sound, they have not been trained on physical run-to-failure degradation curves.
4. **Provisional Demo Thresholds**: Thresholds are calibrated for simulation responsiveness rather than certified OEM standards.

### Industrial Validation Roadmap
1. **High-Frequency Accelerometer Acquisition**: Deploy 10 kHz IEPE industrial accelerometers with edge FFT demodulation (PeakVue / envelope analysis).
2. **Motor Current Signature Analysis (MCSA)**: Sample 3-phase current waveforms at 5 kHz to detect rotor bar break sidebands ($f_b = (1 \pm 2s)f_L$).
3. **Oil Tribology & Ultrasonic Sensors**: Incorporate acoustic emission sensors for micro-crack detection and particulate optical sensors for bearing lubrication degradation.
4. **Physical Benchmark Validation**: Train models against benchmark empirical datasets (e.g. Case Western Reserve University (CWRU) Bearing Data Center or NASA PCoE Milling/Pump Datasets).
