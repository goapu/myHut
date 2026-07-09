# myHut

**myHut** is an iPhone ARKit RGB-D scanning app with a Python/Open3D backend for object reconstruction, room measurement, and full-room 3D reconstruction.

The project combines a SwiftUI + ARKit iOS app with an `ark_fusion` backend that receives RGB-D frames from the phone, saves capture sessions, and reconstructs 3D geometry using depth, camera pose, intrinsics, confidence maps, and metadata.

---

## Features

* iPhone ARKit RGB-D capture
* Live RGB, depth, pose, intrinsics, confidence, and metadata streaming
* Object scanning mode
* Room size measurement mode
* Full room reconstruction mode
* Python backend for receiving and saving capture sessions
* Open3D-based reconstruction pipeline
* Optional confidence-map filtering
* Mode-specific reconstruction profiles
* Safe local configuration using `LocalConfig.example.swift`

---

## Project Structure

```text
myHut/
├── README.md
├── .gitignore
│
├── iOS/
│   └── myHut-iOS/
│       ├── RGBrecoder.xcodeproj
│       └── RGBrecoder/
│           ├── ContentView.swift
│           ├── ARDepthRecorder.swift
│           ├── RGBrecoderApp.swift
│           ├── Info.plist
│           ├── LocalConfig.example.swift
│           └── Assets.xcassets/
│
└── backend/
    └── ark_fusion/
        ├── upload_server.py
        ├── arkit_reconstruct.py
        ├── requirements.txt
        ├── pyproject.toml
        ├── Dockerfile
        ├── docker-compose.yml
        ├── README.md
        ├── MYHUT_BACKEND_README.md
        │
        ├── arkit_sensor_fusion/
        ├── service/
        ├── evaluation/
        ├── profiles/
        └── tests/
```

---

## Capture Modes

myHut supports three capture workflows:

| Mode             | Purpose                                              | Backend Mode   |
| ---------------- | ---------------------------------------------------- | -------------- |
| Object Scan      | Capture one focused object for 3D reconstruction     | `object`       |
| Room Measurement | Capture walls, floor, and ceiling for dimensions     | `measure_room` |
| Full Room Scan   | Capture the full room including furniture and layout | `room_full`    |

---

## iOS App Setup

Open the Xcode project:

```bash
open iOS/myHut-iOS/RGBrecoder.xcodeproj
```

The app requires a local backend server IP address.

Copy the example config:

```bash
cp iOS/myHut-iOS/RGBrecoder/LocalConfig.example.swift \
   iOS/myHut-iOS/RGBrecoder/LocalConfig.swift
```

Edit `LocalConfig.swift`:

```swift
enum LocalConfig {
    static let serverIP = "YOUR_MAC_LOCAL_IP"
    static let serverPort = "5001"
}
```

Replace `YOUR_MAC_LOCAL_IP` with your Mac or PC local network IP address.

On macOS, you can find your Wi-Fi IP with:

```bash
ipconfig getifaddr en0
```

Example:

```swift
enum LocalConfig {
    static let serverIP = "192.168.178.25"
    static let serverPort = "5001"
}
```

`LocalConfig.swift` is ignored by Git and should not be uploaded.

---

## Backend Setup

Go to the backend folder:

```bash
cd backend/ark_fusion
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the upload server:

```bash
python upload_server.py --host 0.0.0.0 --port 5001 --workspace .
```

The backend will listen for data from the iPhone app at:

```text
http://YOUR_MAC_LOCAL_IP:5001
```

---

## Backend Upload API

The iOS app streams frame data to these endpoints:

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

Temporary captures are saved to:

```text
backend/ark_fusion/incoming/
```

When the app sends a save command, the session is moved to:

```text
backend/ark_fusion/captures/
```

When the app sends a discard command, the temporary capture is deleted.

---

## Reconstructing a Capture

After saving a capture from the iPhone app, run reconstruction from the backend folder.

For object reconstruction:

```bash
python arkit_reconstruct.py \
  --dataset ./captures/object_YYYY-MM-DD_HH-mm-ss \
  --mode object \
  --use-confidence
```

For room measurement:

```bash
python arkit_reconstruct.py \
  --dataset ./captures/measure_room_YYYY-MM-DD_HH-mm-ss \
  --mode measure_room \
  --use-confidence
```

For full room reconstruction:

```bash
python arkit_reconstruct.py \
  --dataset ./captures/room_full_YYYY-MM-DD_HH-mm-ss \
  --mode room_full \
  --use-confidence
```

---

## Expected Capture Dataset Layout

A saved capture session should look like this:

```text
captures/
└── room_full_YYYY-MM-DD_HH-mm-ss/
    ├── metadata/
    ├── rgb/
    ├── depth/
    ├── confidence/
    ├── pose/
    ├── intrinsics/
    ├── timestamp/
    └── frame_metadata/
```

Each frame may include:

* RGB image
* depth map
* confidence map
* ARKit camera pose
* camera intrinsics
* timestamp
* frame metadata

---

## Git Safety

The repository intentionally ignores local/private/generated files such as:

```text
LocalConfig.swift
incoming/
captures/
outputs/
results/
rgbd_capture/
arkit_output/
*.bin
*.ply
.venv/
__pycache__/
```

Do not upload:

* your real local IP config
* captured RGB-D scan sessions
* generated meshes
* virtual environments
* large reconstruction outputs

Only upload the safe template:

```text
LocalConfig.example.swift
```

---

## Development Workflow

After making changes:

```bash
git status
git add .
git commit -m "Describe your change"
git push
```

Example:

```bash
git add .
git commit -m "Improve myHut capture UI"
git push
```

---

## Notes

* The iPhone and backend computer must be on the same local network.
* The backend server must be running before starting capture.
* iOS may ask for local network permission.
* ARKit scene depth requires a compatible iPhone or iPad with LiDAR support.
* Better scans come from slow movement, good lighting, and complete coverage of the object or room.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
