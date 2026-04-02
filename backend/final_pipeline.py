
import cv2
import torch

from detector import FaceDetector
from attributes import AttributeExtractor
from embedding import FaceEmbeddingExtractor
from gradcam import generate_attribute_gradcam
import numpy as np


# -------------------------
# LOAD DATABASE
# -------------------------
DB_PATH = "weights/lfw_embeddings.pth"


def load_database():
    data = torch.load(DB_PATH)
    return data["embeddings"].numpy(), data["paths"]


def find_top_k(query_emb, db_embs, db_paths, k=5):
    sims = np.dot(db_embs, query_emb)
    idx = np.argsort(sims)[::-1][:k]
    return [(db_paths[i], sims[i]) for i in idx]


# -------------------------
# MAIN PIPELINE
# -------------------------
def run_pipeline(image):

    detector = FaceDetector(device="cpu")
    attr_extractor = AttributeExtractor(device="cpu")
    embed_extractor = FaceEmbeddingExtractor(device="cpu")

    db_embs, db_paths = load_database()

    # 🔹 Detect face
    face, bbox, conf = detector.get_largest_face(image)

    if face is None:
        print("❌ No face detected")
        return

    print("✅ Face detected")

    # 🔹 Level 1 + 2 attributes
    attributes = attr_extractor.extract(face)
    print("🧠 Attributes:", attributes)

    # 🔹 Level 3 embedding
    emb = embed_extractor.extract(face)
    print("📌 Embedding shape:", emb.shape)

    # 🔹 Retrieval
    results = find_top_k(emb, db_embs, db_paths)

    print("\n🔥 Top 5 Similar Faces:")
    for i, (path, score) in enumerate(results):
        print(f"{i+1}. {score:.3f} → {path}")

        img = cv2.imread(path)
        if img is not None:
            cv2.imshow(f"Match {i+1}", img)

    # 🔹 Grad-CAM
    heatmap, conf = generate_attribute_gradcam(
        face,
        attr_extractor.model,
        attr_extractor.transform,
        torch.device("cpu"),
        target_attribute_idx=13  # Smiling
    )

    print(f"🔥 GradCAM Confidence: {conf:.3f}")

    # 🔹 Show outputs
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Face", face)
    cv2.imshow("GradCAM", heatmap)

    cv2.waitKey(0)


# -------------------------
# INPUT OPTIONS
# -------------------------
def run_image():
    img = cv2.imread("../test.jpeg")
    if img is None:
        print("❌ Image not found")
        return
    run_pipeline(img)


def run_webcam():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Webcam error")
        return

    run_pipeline(frame)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    print("Choose mode:")
    print("1 → Image")
    print("2 → Webcam")

    choice = input("Enter choice: ")

    if choice == "1":
        run_image()
    else:
        run_webcam()

