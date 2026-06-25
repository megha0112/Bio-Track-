from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os

# -----------------------------
# FIXED DATABASE PATH (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "..", "database")
DATABASE = os.path.normpath(DATABASE)

os.makedirs(DATABASE, exist_ok=True)

# -----------------------------
# INPUT
# -----------------------------
name = input("Enter employee name: ").strip()

# -----------------------------
# MODEL LOAD
# -----------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))

cap = cv2.VideoCapture(0)

embeddings = []

print("\nCapture 5 good samples")
print("Press S = Capture")
print("Press Q = Finish\n")

# -----------------------------
# CAMERA LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    faces = face_app.get(frame)

    if len(faces) > 0:

        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Score: {face.det_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Samples: {len(embeddings)}/5",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Register Employee", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # CAPTURE SAMPLE
    # -----------------------------
    if key == ord('s'):

        if len(faces) == 0:
            print("No face detected")
            continue

        face = faces[0]

        if face.det_score < 0.50:
            print(f"Low quality detection ({face.det_score:.2f})")
            continue

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)

        print(f"Sample {len(embeddings)} captured")

    # -----------------------------
    # EXIT
    # -----------------------------
    elif key == ord('q'):

        if len(embeddings) < 3:
            print("Capture at least 3 samples")
            continue
        else:
            break

    if len(embeddings) >= 5:
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# SAVE TEMPLATE
# -----------------------------
embeddings = np.array(embeddings)

template = np.mean(embeddings, axis=0)
template = template / np.linalg.norm(template)

np.save(os.path.join(DATABASE, f"{name}.npy"), template)

print(f"{name} registered successfully ✅")