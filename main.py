import sys
import argparse
from attendance_system import AttendanceSystem
from register_face import register_face


def main():
    parser = argparse.ArgumentParser(description="Face Detection Attendance System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    start_parser = subparsers.add_parser("start", help="Start attendance system")
    start_parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    register_parser = subparsers.add_parser("register", help="Register a new face")
    register_parser.add_argument("--name", type=str, required=True, help="Person's name")
    register_parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    view_parser = subparsers.add_parser("view", help="View attendance records")
    view_parser.add_argument("--date", type=str, default=None, help="Filter by date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.command == "start":
        system = AttendanceSystem()
        system.run(camera_index=args.camera)

    elif args.command == "register":
        register_face(args.name, camera_index=args.camera)

    elif args.command == "view":
        system = AttendanceSystem()
        system.view_attendance(args.date)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
