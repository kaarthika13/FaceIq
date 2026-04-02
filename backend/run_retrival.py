import cv2
import torch
import numpy as np

from detector import FaceDetector
from embedding import FaceEmbeddingExtractor


# -------------------------
# LOAD DATABASE
# -------------------------
DB_PATH = "weights/lfw_embeddings.pth"


def load_database():
    data = torch.load(DB_PATH)
    embeddings = data["embeddings"].numpy()
    paths = data["paths"]

    print(f"✅ Loaded {len(paths)} embeddings")
    return embeddings, paths


# -------------------------
# FIND SIMILAR
# -------------------------
def find_top_k(query_emb, db_embs, db_paths, k=5):

    # cosine similarity
    sims = np.dot(db_embs, query_emb)

    # sort descending
    top_k_idx = np.argsort(sims)[::-1][:k]

    results = []
    for i in top_k_idx:
        results.append((db_paths[i], sims[i]))

    return results


# -------------------------
# MAIN
# -------------------------
def main():

    detector = FaceDetector(device="cpu")
    extractor = FaceEmbeddingExtractor(device="cpu")

    db_embs, db_paths = load_database()

    # 🔹 INPUT IMAGE
    img = cv2.imread("../test.jpeg")

    if img is None:
        print("❌ Image not found")
        return

    # 🔹 DETECT FACE
    face, bbox, conf = detector.get_largest_face(img)

    if face is None:
        print("❌ No face detected")
        return

    print("✅ Face detected")

    # 🔹 EMBEDDING
    query_emb = extractor.extract(face)

    print("🔥 Finding matches...")

    results = find_top_k(query_emb, db_embs, db_paths, k=5)

    # -------------------------
    # SHOW RESULTS
    # -------------------------
    for i, (path, score) in enumerate(results):
        print(f"{i+1}. Score: {score:.3f} → {path}")

        match_img = cv2.imread(path)

        if match_img is not None:
            cv2.imshow(f"Match {i+1}", match_img)

    cv2.imshow("Query", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()