# Face Detection Attendance System

A Python-based attendance system that uses facial recognition to automatically mark attendance when a registered person is detected on camera.

## Features

- Real-time face detection and recognition
- Automatic attendance marking with timestamps
- Face registration for new users
- Attendance records stored in CSV format
- View attendance records by date

## Prerequisites

- Python 3.8+
- A working webcam

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: On Windows, you may need to install `dlib` first:
```bash
pip install cmake
pip install dlib
```

## Usage

### Quick Start

Run the main menu:
```bash
python main.py
```

### Register a New Face

```bash
python register_face.py
```

Or use option 2 in the main menu.

### Start Attendance System

```bash
python attendance_system.py
```

Or use option 1 in the main menu.

### Keyboard Controls (during attendance)

- `q` - Quit the system
- `r` - Reload known faces and reset today's attendance

## Project Structure

```
├── main.py                 # Main entry point with menu
├── attendance_system.py    # Core attendance logic
├── register_face.py        # Face registration module
├── requirements.txt        # Python dependencies
├── known_faces/            # Directory for registered face images (auto-created)
└── attendance.csv          # Attendance records (auto-created)
```

## How It Works

1. Register faces by running `register_face.py` and providing a name
2. Start the attendance system with `main.py` or `attendance_system.py`
3. The system will detect faces and automatically mark attendance when a known face is recognized
4. Each person is marked only once per day
5. Attendance records are saved to `attendance.csv`

## Notes

- Face images are stored in the `known_faces/` directory
- Attendance data is stored in `attendance.csv`
- The system processes every 3rd frame for better performance
- Only one face detection per person per day is recorded
