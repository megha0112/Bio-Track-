import numpy as np
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis

# Smaller model for lower memory usage
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1, det_size=(320, 320))


def get_embedding(frame):
    faces = face_app.get(frame)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda x: x.det_score)

    if face.det_score < 0.60:
        return None

    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def compare_face(embedding, known_embeddings, known_names):
    best_name = "Unknown"
    best_score = 0.0

    for i, db_emb in enumerate(known_embeddings):
        score = 1 - cosine(embedding, db_emb)

        if score > best_score:
            best_score = score
            best_name = known_names[i]

    return best_name, best_score