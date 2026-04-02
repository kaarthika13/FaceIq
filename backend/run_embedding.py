
"""
run_embeddings.py
-----------------
Generate embedding for a single image
"""

import cv2
from detector import FaceDetector
from backend.extract_embedding import FaceEmbeddingExtractor


def main():

    detector = FaceDetector(device="cpu")
    extractor = FaceEmbeddingExtractor(device="cpu")

    print("🔥 Using backend:", extractor.backend_name)

    # 🔹 Load image
    img = cv2.imread("../test.jpeg")

    if img is None:
        print("❌ Image not found")
        return

    # 🔹 Detect face
    face, _, _ = detector.get_largest_face(img)

    if face is None:
        print("❌ Face not detected")
        return

    # 🔹 Extract embedding
    embedding = extractor.extract(face)

    if embedding is None:
        print("❌ Embedding failed")
        return

    print("✅ Embedding generated!")
    print("📏 Shape:", embedding.shape)   # should be (512,)
    print("🔢 First 10 values:", embedding[:10])


if __name__ == "__main__":
    main()

