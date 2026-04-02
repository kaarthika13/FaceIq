import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2


class ResNetEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(feature_dim, 512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class FaceEmbeddingExtractor:

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        self.model = ResNetEmbeddingModel().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def extract(self, face_img):
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)

        return emb.squeeze().cpu().numpy()

    def cosine_similarity(self, e1, e2):
        return float(np.dot(e1, e2))