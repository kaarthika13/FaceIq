"""
Face Detector (R-CNN + Haar Hybrid)
-----------------------------------
Uses:
1. Faster R-CNN → detect person
2. Haar Cascade → refine face inside person
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class FaceDetector:

    STANDARD_SIZE = (224, 224)

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # 🔹 Faster R-CNN
        try:
            self.rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        except:
            self.rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        self.rcnn.to(self.device)
        self.rcnn.eval()

        # 🔹 Transform
        self.transform = transforms.ToTensor()

        # 🔹 Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)

    # --------------------------------------------------
    # STEP 1: RCNN → Person Detection
    # --------------------------------------------------
    def _rcnn_person_boxes(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.rcnn(tensor)[0]

        boxes = []

        for box, label, score in zip(
            preds["boxes"], preds["labels"], preds["scores"]
        ):
            if label.item() == 1 and score.item() > 0.7:  # person class
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                boxes.append((x1, y1, x2, y2, score.item()))

        return boxes

    # --------------------------------------------------
    # STEP 2: Haar → Face inside person
    # --------------------------------------------------
    def _haar_inside_person(self, person_crop):
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return None

        # pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        return (x, y, x + w, y + h)

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def get_largest_face(self, image: np.ndarray):

        if image is None:
            print("❌ Image is None")
            return None, None, 0.0

        h, w = image.shape[:2]

        # 🔥 STEP 1: RCNN
        person_boxes = self._rcnn_person_boxes(image)

        for box in person_boxes:
            x1, y1, x2, y2, conf = box

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            person_crop = image[y1:y2, x1:x2]

            # 🔥 STEP 2: Haar refinement
            face_box = self._haar_inside_person(person_crop)

            if face_box is not None:
                fx1, fy1, fx2, fy2 = face_box

                face_crop = person_crop[fy1:fy2, fx1:fx2]

                # convert back to original image coords
                return face_crop, (x1+fx1, y1+fy1, x1+fx2, y1+fy2), conf

        # 🔁 Fallback: Haar on full image
        face_box = self._haar_inside_person(image)

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            face_crop = image[y1:y2, x1:x2]
            return face_crop, (x1, y1, x2, y2), 0.9

        return None, None, 0.0

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------
    def align_and_resize(self, face_crop):
        return cv2.resize(face_crop, self.STANDARD_SIZE)

    def draw_bbox(self, image, bbox, confidence):
        x1, y1, x2, y2 = bbox
        img = image.copy()

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 111, 0), 2)
        cv2.putText(
            img,
            f"Face {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 111, 0),
            2,
        )
        return img


# --------------------------------------------------
# TEST CODE
# --------------------------------------------------
if __name__ == "__main__":

    detector = FaceDetector(device="cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread("../test.jpeg")

    if img is None:
        print("❌ Image not found")
        exit()

    face, bbox, conf = detector.get_largest_face(img)

    if face is not None:
        print("✅ Face detected!", conf)

        output = detector.draw_bbox(img, bbox, conf)

        cv2.imshow("Detected", output)
        cv2.imshow("Face", face)
        cv2.waitKey(0)

    else:
        print("❌ No face detected")