# 🚨 Intrusion Detection System

A real-time, multi-camera intrusion detection system built with Flask and YOLOv8. This system monitors user-defined zones in live RTSP or USB camera feeds and captures evidence of unauthorized entry, exit, or motion. All events are logged with timestamps, camera ID, and screenshots, and can be viewed, filtered, or exported from the built-in web dashboard.

---

## 🔍 Features

- 🎯 Real-time object detection using YOLOv8
- 📷 Supports multiple cameras (USB or RTSP)
- 🗺️ Customizable polygonal zone detection (user-defined quadrilateral)
- 📸 Screenshot capture on intrusion
- 🗃️ SQLite-based logging using SQLAlchemy
- 🕹️ Web dashboard built with Flask & Jinja2
- 🧾 Intrusion log filtering by:
  - Event type (entry, exit, motion)
  - Date range
- 📤 Export logs to CSV
- ✅ Lightweight and runs on most edge devices

---

## 🧰 Prerequisites

- Python 3.10+
- OpenCV-compatible camera (USB or RTSP IP Camera)
- Git
- pip / conda

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ayush021-Dev/intrusion_detection_system.git
cd intrusion_detection_system
```

### 2. Create & Activate Virtual Environment
Using conda (recommended):
```bash
conda create -n intrusion_detection python=3.10
conda activate intrusion_detection
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎛️ Configuration

### Camera Configuration (`cameras.json`)
Located in the project root:

```json
[
  {
    "name": "Office Entry",
    "camera_index": "rtsp://user:pass@192.168.1.10:554/stream1",
    "is_active": true,
    "zone_points": [[100, 100], [400, 100], [400, 300], [100, 300]]
  },
  {
    "name": "Lab USB Cam",
    "camera_index": 0,
    "is_active": true,
    "zone_points": [[50, 50], [300, 50], [300, 200], [50, 200]]
  }
]
```

- `camera_index` can be an integer (USB) or string (RTSP URL).
- `zone_points` defines a quadrilateral area in the feed.
- Modify this file to add/remove cameras.

---

## 🧪 Usage

### 1. Run the Flask App
```bash
python app.py
```

### 2. Open Dashboard
Visit: [http://localhost:5000](http://localhost:5000)

### 3. Dashboard Features

- 📹 **Live Camera Feeds** with zone overlays
- 🧭 **Custom Zone Drawing** (click 4 points to define)
- 🧾 **Logs Table** with:
  - Filters (event type, date)
  - Export to `.csv`
- 🖼️ **Screenshot Previews** of intrusions
- 🔄 Automatic syncing with `cameras.json`

---

## 🗄️ Log Schema (SQLite)

Each intrusion log entry includes:

- `id` – Unique log ID
- `timestamp` – Date & time of event
- `camera_id` – Name or index from `cameras.json`
- `event_type` – `"entry"`, `"exit"`, or `"motion"`
- `screenshot_path` – Local path to image

---

## 📤 Export Logs

- Navigate to the **Logs** section in the dashboard
- Filter logs by date or type
- Click **Export to CSV** to download filtered results

---

## ✅ Future Improvements

- Email/Telegram alert integration
- Face recognition tagging (optional)
- Admin login for dashboard access
- Real-time zone editing on live feeds

---

