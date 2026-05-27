import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import os


class AttendanceSystem:
    def __init__(self, known_faces_dir="known_faces", attendance_file="attendance.csv", model_file="face_model.yml"):
        self.known_faces_dir = Path(known_faces_dir)
        self.attendance_file = attendance_file
        self.model_file = model_file
        self.images_dir = self.known_faces_dir / "images"
        self.labels_file = self.known_faces_dir / "labels.txt"
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.label_names = {}
        self.known_faces_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
            self.recognizer.read(self.model_file)
            with open(self.labels_file, "r") as f:
                for line in f:
                    idx, name = line.strip().split(",", 1)
                    self.label_names[int(idx)] = name
            print(f"Loaded model with {len(self.label_names)} known faces")
        else:
            print("No trained model found. Register faces first.")

    def _train_model(self):
        images_path = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        if not images_path:
            print("No training images found")
            return False

        faces = []
        labels = []
        label_map = {}
        label_counter = 0

        for img_path in images_path:
            name = img_path.stem.rsplit("_", 1)[0] if "_" in img_path.stem and img_path.stem.rsplit("_", 1)[1].isdigit() else img_path.stem
            if name not in label_map:
                label_map[name] = label_counter
                label_counter += 1

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_map[name])

        if not faces:
            print("No valid faces found for training")
            return False

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(self.model_file)

        self.label_names = {v: k for k, v in label_map.items()}
        with open(self.labels_file, "w") as f:
            for name, idx in label_map.items():
                f.write(f"{idx},{name}\n")

        print(f"Model trained with {len(label_map)} people")
        return True

    def _mark_attendance(self, name):
        today = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
            if ((df["Name"] == name) & (df["Date"] == today)).any():
                return
            new_row = pd.DataFrame(
                [{"Name": name, "Date": today, "Time In": datetime.now().strftime("%H:%M:%S")}]
            )
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame(
                [{"Name": name, "Date": today, "Time In": datetime.now().strftime("%H:%M:%S")}]
            )
        df.to_csv(self.attendance_file, index=False)
        print(f"Attendance marked: {name} at {datetime.now().strftime('%H:%M:%S')}")

    def run(self, camera_index=0):
        if not os.path.exists(self.model_file):
            print("No trained model found. Please register faces first.")
            return

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        marked_today = set()
        confidence_threshold = 80

        print("Starting attendance system. Press 'q' to quit, 't' to retrain, 'r' to reload model")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                label, confidence = self.recognizer.predict(roi_gray)

                if confidence < confidence_threshold and label in self.label_names:
                    name = self.label_names[label]
                    color = (0, 255, 0)
                    conf_text = f"{name} ({int(confidence)}%)"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    conf_text = "Unknown"

                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    display_frame, conf_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

                if name != "Unknown" and name not in marked_today:
                    self._mark_attendance(name)
                    marked_today.add(name)

            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(
                display_frame, f"Time: {current_time}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                display_frame, f"Marked Today: {len(marked_today)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("Face Detection Attendance System", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                self._train_model()
                marked_today.clear()
            elif key == ord("r"):
                self._load_model()
                marked_today.clear()

        cap.release()
        cv2.destroyAllWindows()
        print("Attendance system stopped")

    def view_attendance(self, date=None):
        if not os.path.exists(self.attendance_file):
            print("No attendance records found")
            return
        df = pd.read_csv(self.attendance_file)
        if date:
            df = df[df["Date"] == date]
        if df.empty:
            print(f"No records found for {date}" if date else "No records found")
        else:
            print(f"\nAttendance Records{' for ' + date if date else ''}:")
            print(df.to_string(index=False))
        return df


if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()
