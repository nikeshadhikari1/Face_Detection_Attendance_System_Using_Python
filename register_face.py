import cv2
from pathlib import Path
import os


def register_face(name, known_faces_dir="known_faces", camera_index=0):
    output_dir = Path(known_faces_dir) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_count = 0
    target_faces = 30

    print(f"\nRegistering face for: {name}")
    print(f"Please stay still. Capturing {target_faces} samples...")
    print("Press 'q' to quit early")

    while face_count < target_faces:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            resized = cv2.resize(roi_gray, (200, 200))
            filepath = output_dir / f"{name}_{face_count}.jpg"
            cv2.imwrite(str(filepath), resized)
            face_count += 1

        cv2.putText(
            display, f"Collected: {face_count}/{target_faces}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        if len(faces) == 0:
            cv2.putText(
                display, "No face detected - position your face", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        cv2.imshow("Register Face - Stay Still", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if face_count >= 10:
        print(f"Successfully captured {face_count} face samples")
        print("Training model...")
        from attendance_system import AttendanceSystem
        system = AttendanceSystem()
        system._train_model()
        return True
    else:
        print(f"Only captured {face_count} samples. Need at least 10. Try again.")
        return False


if __name__ == "__main__":
    name = input("Enter person's name: ").strip()
    if name:
        register_face(name)
    else:
        print("Name cannot be empty")
