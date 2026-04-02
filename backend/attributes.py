
"""
attributes.py
-------------
Final Hybrid Attribute Extractor
✔ DeepFace → age + emotion
✔ CelebA → gender + ALL attributes
✔ Consistent + heatmap-ready
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from deepface import DeepFace
import cv2
from PIL import Image


# -----------------------------
# CelebA Attribute Names
# -----------------------------
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes",
    "Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair",
    "Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin",
    "Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones",
    "Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
    "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline",
    "Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair",
    "Wearing_Earrings","Wearing_Hat","Wearing_Lipstick",
    "Wearing_Necklace","Wearing_Necktie","Young"
]


# -----------------------------
# Model
# -----------------------------
class CelebAAttributeModel(nn.Module):
    def __init__(self, num_attributes=40):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_attributes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# -----------------------------
# Extractor
# -----------------------------
import os

class AttributeExtractor:

    def __init__(self, model_path=None, device="cpu"):
        self.device = torch.device(device)

        # ✅ FIX: absolute path
        if model_path is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, "weights", "celeba_model.pth")

        print("MODEL PATH:", model_path)
        print("EXISTS:", os.path.exists(model_path))

        self.model = CelebAAttributeModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.attribute_names = CELEBA_ATTRIBUTES

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract(self, face_img):

        if face_img is None:
            return None

        # -------------------------
        # 1️⃣ DeepFace
        # -------------------------
        face_resized = cv2.resize(face_img, (224, 224))

        df_result = DeepFace.analyze(
            face_resized,
            actions=['age', 'emotion'],
            enforce_detection=False,
            detector_backend="retinaface",
            align=True
        )

        if isinstance(df_result, list):
            df_result = df_result[0]

        age = int(df_result["age"])
        emotion = df_result["dominant_emotion"]

        # -------------------------
        # 2️⃣ CelebA
        # -------------------------
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # 🔥 FIXED: correct index
        male_index = self.attribute_names.index("Male")
        male_prob = probs[male_index]

        if male_prob > 0.6:
            gender = "Male"
        elif male_prob < 0.4:
            gender = "Female"
        else:
            gender = "Uncertain"

        # 🔥 ALL attributes (important)
        attr_dict = {
            name: float(prob)
            for name, prob in zip(self.attribute_names, probs)
        }

        # 🔥 prominent (for UI only)
        prominent_attrs = {
            k: v for k, v in attr_dict.items() if v > 0.6
        }

        return {
            "age": age,
            "gender": gender,
            "emotion": emotion,
            "attributes": attr_dict,              # 🔥 full set
            "prominent_attributes": prominent_attrs
        }

