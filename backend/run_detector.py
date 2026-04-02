
"""
run_attributes.py
-----------------
Full pipeline:
RCNN → Face → DeepFace → Attributes
"""

import cv2
from detector import FaceDetector
from attributes import AttributeExtractor


def main():

    # 🔹 Initialize modules
    detector = FaceDetector(device="cpu")   # or "cuda"
    extractor = AttributeExtractor()

    # 🔹 Load image
    img_path = "../test.jpeg"   # change path if needed
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found")
        return

    # 🔹 Detect face using RCNN
    face, bbox, conf = detector.get_largest_face(img)

    if face is None:
        print("❌ No face detected")
        return

    print("✅ Face detected with confidence:", conf)

    # 🔹 Analyze face using DeepFace
    attributes = extractor.extract(face)

    print("✅ Attributes:", attributes)

    # 🔹 Show results
    output = detector.draw_bbox(img, bbox, conf)

    cv2.imshow("Detected", output)
    cv2.imshow("Face", face)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

