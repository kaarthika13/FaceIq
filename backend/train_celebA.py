"""
train_celeba.py
---------------
Train CelebA attribute model using CSV and save .pth
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_DIR = os.path.join(BASE_DIR, "datasets", "CelebA", "img_align_celeba", "img_align_celeba")
ATTR_FILE = os.path.join(BASE_DIR, "datasets", "CelebA", "list_attr_celeba.csv")
SAVE_PATH = os.path.join(BASE_DIR, "backend", "weights", "celeba_model.pth")

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

# -------------------------------------------------------
# DATASET
# -------------------------------------------------------

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load CSV
        self.df = pd.read_csv(attr_file)

        # Extract columns
        self.image_names = self.df.iloc[:, 0].values
        self.attributes = self.df.iloc[:, 1:].values

        # Convert -1 → 0
        self.attributes = (self.attributes == 1).astype(int)
        print("CSV sample:", self.image_names[:5])
        print("IMG_DIR:", self.img_dir)
        print("Folder sample:", os.listdir(self.img_dir)[:5])
        # 🔥 FILTER VALID IMAGES
        self.data = []
        missing = 0

        print("🔍 Checking dataset...")

        for img, attr in zip(self.image_names, self.attributes):
            img = str(img).strip()  # ✅ FIX (remove spaces)

            img_path = os.path.join(self.img_dir, img)

            if os.path.exists(img_path):
                self.data.append((img, attr))
            else:
                missing += 1

        print(f"✅ Valid images: {len(self.data)}")
        print(f"❌ Missing images skipped: {missing}")

        if len(self.data) == 0:
            raise ValueError("❌ No valid images found. Check dataset paths and filenames!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, attrs = self.data[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(attrs, dtype=torch.float32)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

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

# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(IMG_DIR, ATTR_FILE, transform)

    # 🔥 Optional: quick test (uncomment if needed)
    # dataset = torch.utils.data.Subset(dataset, range(20000))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = CelebAAttributeModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("🔥 Training started...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)

    print(f"✅ Model saved at: {SAVE_PATH}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    train()