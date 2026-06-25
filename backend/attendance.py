from insightface.app import FaceAnalysis
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import os
import csv
from datetime import datetime

# ---------------------------
# ArcFace
# ---------------------------
face_app = FaceAnalysis(name="buffalo_l")

face_app.prepare(
    ctx_id=-1,
    det_size=(640,640)
)

# ---------------------------
# Load Database
# ---------------------------
database_path = "database"

known_names = []
known_embeddings = []

os.makedirs(
    database_path,
    exist_ok=True
)

for file in os.listdir(database_path):

    if file.endswith(".npy"):

        embedding = np.load(
            os.path.join(
                database_path,
                file
            )
        )

        embedding = (
            embedding /
            np.linalg.norm(
                embedding
            )
        )

        known_names.append(
            os.path.splitext(file)[0]
        )

        known_embeddings.append(
            embedding
        )

print(
    "\nLoaded Users:"
)

print(
    known_names
)

if len(known_names) == 0:

    print(
        "No registered users found."
    )

    exit()

# ---------------------------
# Attendance CSV
# ---------------------------
csv_file = "attendance.csv"

if not os.path.exists(
    csv_file
):

    with open(
        csv_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Name",
                "Date",
                "Time",
                "Status"
            ]
        )

# ---------------------------
# Mark Attendance
# ---------------------------
def mark_attendance(name):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    exists = False

    with open(
        csv_file,
        "r"
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if (
                len(row) > 1
                and
                row[0] == name
                and
                row[1] == today
            ):
                exists = True
                break

    if exists:
        return

    now = datetime.now()

    with open(
        csv_file,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                name,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                "Present"
            ]
        )

    print(
        f"{name} marked present"
    )

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

# ---------------------------
# Main Loop
# ---------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = face_app.get(frame)

    for face in faces:

        embedding = face.embedding

        embedding = (
            embedding /
            np.linalg.norm(
                embedding
            )
        )

        best_similarity = 0
        best_name = "Unknown"

        for i, stored_embedding in enumerate(
            known_embeddings
        ):

            similarity = (
                1 -
                cosine(
                    embedding,
                    stored_embedding
                )
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_name = known_names[i]

        THRESHOLD = 0.75

        if best_similarity >= THRESHOLD:

            name = best_name

            color = (0,255,0)

            mark_attendance(
                name
            )

        else:

            name = "Unknown"

            color = (0,0,255)

        confidence = (
            best_similarity * 100
        )

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} | {confidence:.1f}%",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()