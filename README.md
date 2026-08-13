# myHut

**myHut** is an iPhone ARKit RGB-D scanning app with a Python/Open3D backend for object reconstruction, room measurement, and full-room 3D reconstruction.

The project combines a SwiftUI + ARKit iOS app with an `ark_fusion` backend. The iPhone captures RGB images, depth maps, camera poses, intrinsics, confidence maps, timestamps, and metadata, then streams them to a local backend server for saving and reconstruction.


---

## Overview

myHut is designed for three capture workflows:

| Capture Mode         | Purpose                                                | Backend Mode   |
| -------------------- | ------------------------------------------------------ | -------------- |
| **Object Scan**      | Capture one focused object for 3D reconstruction       | `object`       |
| **Room Measurement** | Capture walls, floor, and ceiling for room dimensions  | `measure_room` |
| **Full Room Scan**   | Capture a complete room including furniture and layout | `room_full`    |

The app streams RGB-D data over your local network. The backend receives the data, stores capture sessions, and runs the reconstruction pipeline.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/goapu/myHut.git
cd myHut
```

Start the backend:

```bash
cd backend/ark_fusion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python upload_server.py --host 0.0.0.0 --port 5001 --workspace .
```

Open the iOS app:

```bash
open ../../iOS/myHut-iOS/RGBrecoder.xcodeproj
```

Before running the app, create your local config file:

```bash
cp ../../iOS/myHut-iOS/RGBrecoder/LocalConfig.example.swift \
   ../../iOS/myHut-iOS/RGBrecoder/LocalConfig.swift
```

Edit `LocalConfig.swift` and set your backend computer’s local IP address.

---

## Features

* SwiftUI + ARKit iPhone capture app
* Live RGB-D streaming to a local backend
* RGB image capture
* Float32 depth-map capture
* ARKit confidence-map capture when available
* Camera pose and intrinsics export
* Frame timestamps and metadata export
* Object reconstruction workflow
* Room measurement workflow
* Full-room reconstruction workflow
* Python/Open3D reconstruction backend
* Local configuration template for safe GitHub usage
* Git ignore rules for private config, scan data, and generated outputs

---

## Project Structure

```text
myHut/
├── README.md
├── LICENSE
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

## Requirements

### iOS

* Xcode
* iPhone or iPad with ARKit scene depth support
* LiDAR-capable device recommended
* iOS local network permission enabled

### Backend

* Python 3.10+
* macOS, Linux, or Windows
* Open3D-compatible environment
* iPhone and backend computer on the same local network

---

## iOS App Setup

Open the Xcode project:

```bash
open iOS/myHut-iOS/RGBrecoder.xcodeproj
```

The app needs your backend computer’s local IP address.

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

On macOS, you can find your Wi-Fi IP address with:

```bash
ipconfig getifaddr en0
```

Example:

```swift
enum LocalConfig {
    static let serverIP = "192.168.1.100"
    static let serverPort = "5001"
}
```

> `LocalConfig.swift` is ignored by Git and should not be uploaded to GitHub.

---

## Backend Setup

Go to the backend folder:

```bash
cd backend/ark_fusion
```

Create and activate a virtual environment:

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

The backend will listen at:

```text
http://YOUR_MAC_LOCAL_IP:5001
```

For example:

```text
http://192.168.1.100:5001
```

---

## Upload API

The iOS app streams capture data to these endpoints:

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

The backend first stores active capture sessions in:

```text
backend/ark_fusion/incoming/
```

When the app sends a save command, the session is moved to:

```text
backend/ark_fusion/captures/
```

When the app sends a discard command, the temporary capture is deleted.

---

## Saved Capture Layout

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

## Reconstructing a Capture

Run reconstruction from the backend folder:

```bash
cd backend/ark_fusion
source .venv/bin/activate
```

### Object Reconstruction

```bash
python arkit_reconstruct.py \
  --dataset ./captures/object_YYYY-MM-DD_HH-mm-ss \
  --mode object \
  --use-confidence
```

### Room Measurement

```bash
python arkit_reconstruct.py \
  --dataset ./captures/measure_room_YYYY-MM-DD_HH-mm-ss \
  --mode measure_room \
  --use-confidence
```

### Full Room Reconstruction

```bash
python arkit_reconstruct.py \
  --dataset ./captures/room_full_YYYY-MM-DD_HH-mm-ss \
  --mode room_full \
  --use-confidence
```

---

## Capture Mode Details

### Object Scan

Use this mode for a single object, furniture item, or small scene component.

Best practices:

* Keep the object centered.
* Move slowly around the object.
* Avoid cluttered backgrounds.
* Avoid reflective, transparent, or very dark materials.
* Capture the object from multiple angles.

Backend mode:

```text
object
```

### Room Measurement

Use this mode to estimate room dimensions from walls, floor, ceiling, and corners.

Best practices:

* Capture all visible walls.
* Capture floor-wall and ceiling-wall edges.
* Move slowly along the room perimeter.
* Avoid relying on furniture surfaces for measurement.
* Try to include opposite walls when possible.

Backend mode:

```text
measure_room
```

### Full Room Scan

Use this mode for a complete room reconstruction, including furniture and layout.

Best practices:

* Walk slowly through the room.
* Capture furniture, walls, floor, and corners.
* Revisit key areas from multiple viewpoints.
* Avoid very fast rotations.
* Keep lighting stable.

Backend mode:

```text
room_full
```

---

## GitHub Safety

The repository should ignore local, private, and generated files:

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

* your real `LocalConfig.swift`
* your local IP address configuration
* captured RGB-D sessions
* generated meshes
* reconstruction outputs
* Python virtual environments
* cache files

Only upload the safe template:

```text
LocalConfig.example.swift
```

If a private or generated file was accidentally added, remove it from Git tracking while keeping it locally:

```bash
git rm --cached path/to/file
git add .gitignore
git commit -m "Remove private or generated file from Git"
git push
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

Examples:

```bash
git commit -m "Improve myHut capture UI"
```

```bash
git commit -m "Add ark_fusion backend"
```

```bash
git commit -m "Update reconstruction pipeline"
```

---

## Scanning Tips

For better results:

* Move slowly while scanning.
* Keep the object or room surfaces in view.
* Avoid shiny, transparent, or very dark surfaces.
* Use good lighting.
* For object scans, keep the object centered and reduce background clutter.
* For room measurement, capture walls, floor, ceiling edges, and corners.
* For full-room scans, walk slowly and cover all major surfaces.

---

## Troubleshooting

### The iPhone cannot connect to the backend

Check that:

* the iPhone and backend computer are on the same Wi-Fi network
* `LocalConfig.swift` contains the correct local IP address
* the backend is running on port `5001`
* your firewall allows incoming connections
* iOS local network permission is enabled

You can test the backend from another device on the network by visiting:

```text
http://YOUR_MAC_LOCAL_IP:5001
```

### GitHub shows private config or capture data

Remove private/generated files from Git tracking:

```bash
git rm --cached path/to/file
git add .gitignore
git commit -m "Remove private or generated files"
git push
```

### Backend dependencies fail to install

Make sure your virtual environment is active:

```bash
source .venv/bin/activate
```

Then upgrade pip:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### No capture appears in `captures/`

Check that:

* the backend server is running
* the app is sending data to the correct IP address
* the capture was saved, not discarded
* the backend received the `/command` save request
* `incoming/` contains the active session before saving

### Reconstruction fails or produces an empty mesh

Check that:

* the capture folder contains RGB, depth, pose, and intrinsics files
* the selected backend mode matches the capture mode
* the depth data is valid
* the iPhone moved slowly during capture
* the scan contains enough overlapping viewpoints

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
