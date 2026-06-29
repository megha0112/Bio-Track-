import base64
import csv
from datetime import datetime
import os
from flask import Flask, jsonify, request
from flask_cors import CORS  # Added CORS
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# ---------------------------
# Path & Environment Setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If running on Railway, prioritize the volume mount path for data persistence
if os.getenv("RAILWAY_ENVIRONMENT") or os.path.exists("/app/database"):
    DATABASE = "/app/database"
else:
    DATABASE = os.path.normpath(os.path.join(BASE_DIR, "database"))

CSV_FILE = os.path.join(DATABASE, "attendance.csv")
os.makedirs(DATABASE, exist_ok=True)

# Initialize Flask (Removed frontend template_folder since Vercel handles frontend)
app = Flask(__name__)

# Enable CORS - Replace with your actual live Vercel URL after deploying
CORS(app, origins=[
    "http://localhost:3000", # For local frontend testing if needed
    "https://your-vercel-app.vercel.app" # Your production Vercel frontend
])

# Initialize Face Analysis Engine
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1, det_size=(320, 320))

REGISTRATION_SESSIONS = {}

# Ensure CSV template structure exists in the persistent directory
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status"])

# ---------------------------
# Global Database Tracker
# ---------------------------
known_names = []
known_embeddings = []

def reload_database():
    global known_names, known_embeddings
    known_names = []
    known_embeddings = []
    for file in sorted(os.listdir(DATABASE)):
        if file.endswith(".npy"):
            path = os.path.join(DATABASE, file)
            embedding = np.load(path)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm
            known_names.append(os.path.splitext(file)[0])
            known_embeddings.append(embedding)
    print(f"[DB] Loaded {len(known_names)} users: {known_names}")

reload_database()

# ---------------------------
# API Routes
# ---------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "Bio-Track API"})

@app.route("/api/users", methods=["GET"])
def get_users():
    return jsonify({"users": known_names})

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    records = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    records.append({
                        "name": row[0],
                        "date": row[1],
                        "time": row[2],
                        "status": row[3]
                    })
    return jsonify({"attendance": records[::-1]})

@app.route("/api/debug", methods=["GET"])
def debug():
    return jsonify({
        "loaded_names": known_names,
        "embedding_norms": [float(np.linalg.norm(e)) for e in known_embeddings],
        "count": len(known_names),
        "database_path": DATABASE
    })

# ---------------------------
# Upload — Face Detection
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.json.get("image", "")
    if not data:
        return jsonify({"status": "error", "message": "No image payload"}), 400

    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Could not decode image"}), 400

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    faces = face_app.get(frame)
    if len(faces) == 0:
        return jsonify({"status": "unknown", "message": "No face visible", "results": []})

    face_results = []
    marked_count = 0
    today = datetime.now().strftime("%Y-%m-%d")

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(640, x2), min(480, y2)
        bbox_coordinates = [x1, y1, x2 - x1, y2 - y1]

        if face.det_score < 0.60:
            continue

        embedding = face.embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            continue
        embedding = embedding / norm

        if not known_names:
            face_results.append({
                "name": "Unknown", "score": 0.0,
                "status": "unknown", "bbox": bbox_coordinates
            })
            continue

        best_similarity = -1
        best_name = "Unknown"

        for i, stored_embedding in enumerate(known_embeddings):
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = known_names[i]

        THRESHOLD = 0.75

        if best_similarity >= THRESHOLD:
            already_marked = False
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0] == best_name and row[1] == today:
                            already_marked = True
                            break

            if not already_marked:
                now = datetime.now()
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([best_name, today, now.strftime("%H:%M:%S"), "Present"])
                status = "success"
                marked_count += 1
            else:
                status = "already_marked"

            face_results.append({
                "name": best_name,
                "score": float(best_similarity),
                "status": status,
                "bbox": bbox_coordinates
            })
        else:
            face_results.append({
                "name": "Unknown",
                "score": float(best_similarity),
                "status": "unknown",
                "bbox": bbox_coordinates
            })

    return jsonify({
        "status": "processed",
        "message": f"Tracked {len(face_results)} faces. Marked {marked_count} new.",
        "results": face_results
    })

# ---------------------------
# Register New Users
# ---------------------------
@app.route("/api/register", methods=["POST"])
def register_user():
    data = request.json
    name = data.get("name", "").strip()
    img_b64 = data.get("image", "")

    if not name or not img_b64:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400

    img_data = base64.b64decode(img_b64.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Could not decode image"}), 400

    frame = cv2.flip(frame, 1)
    faces = face_app.get(frame)
    if len(faces) == 0:
        return jsonify({"status": "error", "message": "No face detected", "bbox": None})

    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    bbox_coordinates = [x1, y1, x2 - x1, y2 - y1]

    if face.det_score < 0.50:
        return jsonify({
            "status": "error",
            "message": f"Low quality detection ({face.det_score:.2f})",
            "bbox": bbox_coordinates
        })

    embedding = face.embedding / np.linalg.norm(face.embedding)

    if name not in REGISTRATION_SESSIONS:
        REGISTRATION_SESSIONS[name] = []

    REGISTRATION_SESSIONS[name].append(embedding)
    current_count = len(REGISTRATION_SESSIONS[name])

    if current_count >= 5:
        stacked_embeddings = np.array(REGISTRATION_SESSIONS[name])
        mean_template = np.mean(stacked_embeddings, axis=0)
        mean_template = mean_template / np.linalg.norm(mean_template)
        
        # Save straight to persistent storage
        np.save(os.path.join(DATABASE, f"{name}.npy"), mean_template)
        reload_database()
        del REGISTRATION_SESSIONS[name]

        return jsonify({
            "status": "completed",
            "message": f"Successfully registered {name} with 5 angle profiles!",
            "count": current_count,
            "bbox": bbox_coordinates
        })

    return jsonify({
        "status": "captured",
        "message": f"Sample {current_count}/5 captured successfully.",
        "count": current_count,
        "bbox": bbox_coordinates
    })

if __name__ == "__main__":
    # Bind to 0.0.0.0 so Railway can expose it externally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
