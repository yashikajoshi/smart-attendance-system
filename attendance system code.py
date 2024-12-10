import dlib
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known faces and their encodings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    names = ["me", "anuj", "shubham", "kajal", "harshada", "yashika", "rohit", "pankaj"]
    
    for name in names:
        img = cv2.imread(f"faces/{name}.jpg")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_locations = detector(img_rgb)
        
        for face in face_locations:
            shape = predictor(img_rgb, face)
            encoding = np.array(face_recognition.face_encodings(img_rgb, [face])[0])
            known_face_encodings.append(encoding)
            known_face_names.append(name.capitalize())
    
    return known_face_encodings, known_face_names

# Main function for attendance
def main():
    video_capture = cv2.VideoCapture(0)
    known_face_encodings, known_face_names = load_known_faces()
    students = known_face_names.copy()
    
    now = datetime.now()
    current_date = now.strftime("%y-%m-%d")
    with open(f"{current_date}.csv", "w+", newline="") as f:
        lnwriter = csv.writer(f)

        while True:
            _, frame = video_capture.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector = dlib.get_frontal_face_detector()
            face_locations = detector(rgb_frame)
            face_encodings = [face_recognition.face_encodings(rgb_frame, [face])[0] for face in face_locations]

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                best _match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name in students:
                        students.remove(name)
                        current_time = now.strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time])
                        cv2.putText(frame, f"{name} present", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
