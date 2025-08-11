import cv2
import dlib
import face_recognition
import numpy as np
import os
import time

# Load face detector & emotion model
face_detector = dlib.get_frontal_face_detector()
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")  # Smile detector

# Load known faces for recognition
known_faces = []
known_names = []

# Load known images from "faces" folder
faces_path = "faces"  # Create a folder named 'faces' in the same directory
if not os.path.exists(faces_path):
    os.makedirs(faces_path)

for filename in os.listdir(faces_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = face_recognition.load_image_file(os.path.join(faces_path, filename))
        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_faces.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])  # Save filename as name

# Initialize webcam
cap = cv2.VideoCapture(0)

# Face tracking
face_ids = {}
next_face_id = 1

# Previous frame time for FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib
    faces = face_detector(gray)

    # Initialize counters
    left_count = 0
    center_count = 0
    right_count = 0
    total_smiling = 0  # Count how many faces are smiling

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Frame center for position tracking
    frame_center = frame.shape[1] // 2

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Assign face ID (Multi-face tracking)
        face_detected = False
        for fid, (fx, fy, fw, fh) in face_ids.items():
            if abs(fx - x) < 50 and abs(fy - y) < 50:
                face_ids[fid] = (x, y, w, h)
                face_detected = True
                break

        if not face_detected:
            face_ids[next_face_id] = (x, y, w, h)
            next_face_id += 1

        face_id = next((fid for fid, (fx, fy, fw, fh) in face_ids.items() if abs(fx - x) < 50 and abs(fy - y) < 50), None)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Face Position Detection (Left, Center, Right)
        face_center = x + w // 2
        if face_center < frame_center - 100:
            position = "Left"
            left_count += 1
        elif face_center > frame_center + 100:
            position = "Right"
            right_count += 1
        else:
            position = "Center"
            center_count += 1

        cv2.putText(frame, f"Position: {position}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Face Recognition
        face_enc = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])
        if face_enc:
            matches = face_recognition.compare_faces(known_faces, face_enc[0])
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
            cv2.putText(frame, name, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Emotion Detection (More Accurate Smile Detection)
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=30, minSize=(30, 30))

        if len(smiles) > 0:
            total_smiling += 1  # Count smiling faces
            cv2.putText(frame, "Smiling 😊", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display total face count
    cv2.putText(frame, f"Total Faces: {len(faces)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display face position counts
    cv2.putText(frame, f"Left: {left_count} | Center: {center_count} | Right: {right_count}", 
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display total smiling faces
    cv2.putText(frame, f"Smiling: {total_smiling}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the output
    cv2.imshow("Advanced Face Detection", frame)

    # Press 'S' to save the detected faces
    if cv2.waitKey(1) & 0xFF == ord('s'):
        for fid, (x, y, w, h) in face_ids.items():
            face_img = frame[y:y+h, x:x+w]
            face_filename = f"saved_faces/face_{fid}.jpg"
            if not os.path.exists("saved_faces"):
                os.makedirs("saved_faces")
            cv2.imwrite(face_filename, face_img)
            print(f"Saved face: {face_filename}")

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
