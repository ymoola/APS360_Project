import torch
from torchvision import transforms
from PIL import Image
from model.vit_feature_extractor import ViTFeatureExtractor
from model.ann_classifier import ClassifierVIT
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Calculus', 'Caries', 'Gingivitis', 'hypodontia', 'Ulcer', 'Tooth Discoloration']

# Go one level up to get to repo root (APS360_Project)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(root_dir, "model_VIT_bs32_lr0.0001_epoch24")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model_path: str = model_path) -> str:
    # Load models
    vit = ViTFeatureExtractor().to(device)
    classifier = ClassifierVIT().to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()
    vit.eval()

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = vit(input_tensor)
        outputs = classifier(features)
        predicted_idx = outputs.argmax(dim=1).item()

    return class_names[predicted_idx]
