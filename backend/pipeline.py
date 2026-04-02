
"""
pipeline.py
-----------
Main pipeline connecting all modules
"""

import cv2
import torch
import numpy as np

from detector import FaceDetector
from attributes import AttributeExtractor
from embedding import FaceEmbeddingExtractor
from gradcam import generate_attribute_gradcam


class PipelineResult:
    def __init__(self):
        self.success = False
        self.error = None

        self.annotated_image_bgr = None
        self.detected_face_bgr = None
        self.detection_confidence = 0

        self.age = None
        self.age_group = None
        self.gender = None
        self.gender_confidence = None
        self.emotion = None
        self.emotion_confidence = None
        self.emotion_all_probs = None
        self.emotion_emoji = "😐"

        self.attributes = {}
        self.prominent_attributes = {}

        self.embedding = None
        self.embedding_backend = "resnet50"

        self.retrieved_faces = []

        self.emotion_gradcam_bgr = None
        self.attribute_gradcam_bgr = None


class FacePipeline:

    def __init__(self, device="cpu"):
        self.device = device

        self.detector = FaceDetector(device=device)
        self.attr = AttributeExtractor(device=device)
        self.embed = FaceEmbeddingExtractor(device=device)

        # Load database
        import os

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lfw_path = os.path.join(BASE_DIR, "weights", "lfw_embeddings.pth")

        print("LFW PATH:", lfw_path)
        print("EXISTS:", os.path.exists(lfw_path))

        db = torch.load(lfw_path, map_location=self.device)
        self.db_embeddings = db["embeddings"].numpy()
        self.db_paths = db["paths"]

    def process(self, image_bgr, top_k=5,
                enable_attribute_filter=True,
                generate_gradcam=True):

        result = PipelineResult()

        try:
            # -------------------------
            # 1️⃣ FACE DETECTION
            # -------------------------
            face, bbox, conf = self.detector.get_largest_face(image_bgr)

            if face is None:
                result.error = "No face detected"
                return result

            result.success = True
            result.detected_face_bgr = face
            result.detection_confidence = conf

            result.annotated_image_bgr = self.detector.draw_bbox(image_bgr, bbox, conf)

            # -------------------------
            # 2️⃣ ATTRIBUTES
            # -------------------------
            attrs = self.attr.extract(face)

            result.age = attrs["age"]
            result.gender = attrs["gender"]
            result.emotion = attrs["emotion"]

            # simple placeholders
            

            result.attributes = attrs.get("attributes", {})

            result.prominent_attributes = {
                k: v for k, v in result.attributes.items() if v > 0.6
            }

            # -------------------------
            # 3️⃣ EMBEDDING
            # -------------------------
            emb = self.embed.extract(face)
            result.embedding = emb
            result.embedding_backend = "resnet50"

            # -------------------------
            # 4️⃣ RETRIEVAL
            # -------------------------
            sims = np.dot(self.db_embeddings, emb)
            idx = np.argsort(sims)[::-1][:top_k]

            for i, ind in enumerate(idx):
                result.retrieved_faces.append({
                    "rank": i + 1,
                    "similarity_pct": float(sims[ind] * 100),
                    "thumbnail": self.db_paths[ind],
                    "gender": "-",
                    "age": "-",
                    "emotion": "-"
                })

            # -------------------------
            # 5️⃣ GRADCAM
            # -------------------------
            if generate_gradcam:
                heatmap, _ = generate_attribute_gradcam(
                    face,
                    self.attr.model,
                    self.attr.transform,
                    torch.device(self.device),
                    target_attribute_idx=13
                )
                result.attribute_gradcam_bgr = heatmap

        except Exception as e:
            result.error = str(e)

        return result