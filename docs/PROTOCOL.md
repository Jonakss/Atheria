# ATHERIA WebSocket Protocol

This document describes the WebSocket protocol used for communication between the Atheria backend (Python) and frontend (React).

## Connection

- **URL**: `ws://<host>:<port>/ws` (Default: `ws://localhost:8000/ws`)
- **Handshake**: Standard WebSocket handshake.

## Message Structure

All messages (both Client -> Server and Server -> Client) generally follow this JSON structure:

```json
{
  "type": "message_type",
  "payload": { ... }
}
```

### Legacy/Frontend Command Format (Client -> Server)

The frontend often sends commands in this format, which the `WebSocketService` translates:

```json
{
  "scope": "inference",
  "command": "play",
  "args": {
    "param1": "value1"
  }
}
```

This is internally mapped to `type: "inference.play"`, `payload: args`.

## Server -> Client Messages

### 1. Initial State (`initial_state`)
Sent immediately upon connection.

```json
{
  "type": "initial_state",
  "payload": {
    "experiments": [ ... ],
    "training_status": "idle" | "running",
    "inference_status": "paused" | "running",
    "compile_status": { "is_compiled": boolean, ... },
    "active_experiment": "ExperimentName" | null
  }
}
```

### 2. Simulation Frame (`simulation_frame`)
Contains visualization data for a single simulation step. Can be sent as JSON or Binary (MsgPack/CBOR).

```json
{
  "type": "simulation_frame",
  "payload": {
    "step": 123,
    "timestamp": 1678900000,
    "simulation_info": {
        "step": 123,
        "is_paused": boolean,
        "fps": 30.0,
        "live_feed_enabled": boolean
    },
    "map_data": [ [ ... ] ], // 2D array (density, phase, etc.)
    "hist_data": { ... },
    "poincare_coords": [ [x, y], ... ],
    "phase_attractor": { ... },
    "flow_data": { "dx": ..., "dy": ..., "magnitude": ... },
    "complex_3d_data": { "real": ..., "imag": ... },
    "phase_hsv_data": { "hue": ..., "saturation": ..., "value": ... }
  }
}
```

### 3. Status Updates
- `inference_status_update`: `{ "status": "running" | "paused" | "idle", "simulation_info": ... }`
- `training_status_update`: `{ "status": "running" | "idle", ... }`
- `analysis_status_update`: `{ "status": "running" | "completed", "type": "universe_atlas" | ... }`

### 4. Notifications (`notification`)
System notifications to be displayed to the user.

```json
{
  "type": "notification",
  "payload": {
    "status": "info" | "success" | "warning" | "error",
    "message": "Human readable message"
  }
}
```

## Client -> Server Commands

Commands are grouped by **Scope**. See `docs/API_COMMANDS.md` for a complete list of available commands.

### Common Commands

#### Scope: `inference`
- `play`: Start simulation.
- `pause`: Pause simulation.
- `load_experiment`: `{ "experiment_name": "Name" }`
- `set_viz`: `{ "viz_type": "density" | "phase" | "flow" | ... }`
- `inject_energy`: `{ "energy_type": "random_noise", "amount": 1.0 }`

#### Scope: `simulation`
- `set_fps`: `{ "fps": 60 }`
- `set_live_feed`: `{ "enabled": true }`

## Binary Protocol (Optimization)

For high-performance data transfer (e.g., large `map_data`), the server may send binary frames.
1. **Metadata Header**: JSON message with `type: "..._binary"`, `format`: "msgpack", `size`: N.
2. **Binary Payload**: Raw bytes following the header.

The frontend detects binary frames and decodes them using the specified format.
