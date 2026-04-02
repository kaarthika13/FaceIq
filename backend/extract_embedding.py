"""
extract_embeddings.py
---------------------
Generate embeddings for LFW dataset and save
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -------------------------
# PATHS (CORRECT ✅)
# -------------------------
LFW_DIR = "../datasets/LFW/lfw-deepfunneled/lfw-deepfunneled"
SAVE_PATH = "weights/lfw_embeddings.pth"

IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# DATASET
# -------------------------
class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for subdir, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".jpg"):
                    self.samples.append(os.path.join(subdir, f))

        print(f"✅ Found images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path


# -------------------------
# MODEL (FEATURE EXTRACTOR)
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC
model = model.to(DEVICE)
model.eval()


# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# -------------------------
# MAIN
# -------------------------
def main():

    dataset = LFWDataset(LFW_DIR, transform)

    if len(dataset) == 0:
        print("❌ Dataset empty → check path")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_embeddings = []
    all_paths = []

    print("🔥 Extracting embeddings...")

    with torch.no_grad():
        for images, paths in loader:

            images = images.to(DEVICE)

            features = model(images)
            features = torch.flatten(features, 1)

            # 🔥 normalize
            features = F.normalize(features, p=2, dim=1)

            all_embeddings.append(features.cpu())
            all_paths.extend(paths)

    all_embeddings = torch.cat(all_embeddings)

    print("✅ Shape:", all_embeddings.shape)

    # 🔥 create folder
    os.makedirs("weights", exist_ok=True)

    torch.save({
        "embeddings": all_embeddings,
        "paths": all_paths
    }, SAVE_PATH)

    print("💾 Saved at:", SAVE_PATH)


if __name__ == "__main__":
    main()

