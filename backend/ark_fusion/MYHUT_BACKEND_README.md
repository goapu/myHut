# myHut Backend Integration

This backend is ready to sit inside the myHut repository as:

```text
backend/ark_fusion/
```

It contains two server styles:

1. `upload_server.py` — matches the iPhone Swift `ARDepthRecorder` streaming API on port `5001`.
2. `service/api.py` — zip-dataset reconstruction API on port `8000`.

## iPhone streaming server

Start the upload server from `backend/ark_fusion`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python upload_server.py --host 0.0.0.0 --port 5001 --workspace .
```

The iPhone app should point to your Mac local IP and port `5001`.

The server receives:

```text
POST /upload/rgb
POST /upload/depth
POST /upload/confidence
POST /upload/pose
POST /upload/intrinsics
POST /upload/timestamp
POST /upload/frame_metadata
POST /upload/metadata
POST /command
```

Incoming sessions are stored in:

```text
incoming/<session_name>/
```

When the app sends `action=save`, the session is moved to:

```text
captures/<session_name>/
```

When the app sends `action=discard`, the incoming session is deleted.

## Run reconstruction manually

```bash
python arkit_reconstruct.py \
  --dataset ./captures/room_full_YYYY-MM-DD_HH-mm-ss \
  --output ./outputs/room_full_YYYY-MM-DD_HH-mm-ss \
  --mode room_full \
  --use-confidence
```

Valid modes:

```text
object
measure_room
room_full
```

## Run reconstruction through the upload server

```bash
curl -X POST http://localhost:5001/reconstruct/SESSION_NAME \
  -H 'Content-Type: application/json' \
  -d '{"mode":"room_full","use_confidence":true}'
```

Check outputs:

```bash
curl http://localhost:5001/outputs/SESSION_NAME
```

## Private information

Do not upload these to GitHub:

```text
LocalConfig.swift
incoming/
captures/
outputs/
*.bin
*.ply
.env
```

Upload only `LocalConfig.example.swift` in the iOS app.
