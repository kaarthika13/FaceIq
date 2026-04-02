import cv2
from detector import FaceDetector
from attributes import AttributeExtractor

# Load image
img = cv2.imread("../test.jpeg")

if img is None:
    print("❌ Image not found")
    exit()

# Step 1: Detect face
detector = FaceDetector()
face, bbox, conf = detector.get_largest_face(img)

if face is None:
    print("❌ No face detected")
    exit()

# Step 2: Extract attributes
extractor = AttributeExtractor()
attributes = extractor.extract(face)

print("✅ Attributes:", attributes)

# Show result
output = detector.draw_bbox(img, bbox, conf)

cv2.imshow("Detected", output)
cv2.imshow("Face", face)
cv2.waitKey(0)