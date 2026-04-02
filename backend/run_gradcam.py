
import cv2
import torch
from detector import FaceDetector
from attributes import AttributeExtractor
from gradcam import generate_attribute_gradcam


def main():

    # -------------------------
    # INIT
    # -------------------------
    detector = FaceDetector(device="cpu")
    extractor = AttributeExtractor(device="cpu")

    # -------------------------
    # LOAD IMAGE
    # -------------------------
    img = cv2.imread("../test.jpeg")

    if img is None:
        print("❌ Image not found")
        return

    # -------------------------
    # FACE DETECTION
    # -------------------------
    face, _, _ = detector.get_largest_face(img)

    if face is None:
        print("❌ No face detected")
        return

    print("✅ Face detected")

    # -------------------------
    # GRAD-CAM
    # -------------------------
    heatmap, confidence = generate_attribute_gradcam(
        face_bgr=face,
        model=extractor.model,
        transform=extractor.transform,
        device=torch.device("cpu"),
        target_attribute_idx=13   # Smiling
    )

    print(f"🔥 Attribute Confidence: {confidence:.3f}")

    # -------------------------
    # SHOW OUTPUT
    # -------------------------
    cv2.imshow("Original Face", face)
    cv2.imshow("Grad-CAM Heatmap", heatmap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

