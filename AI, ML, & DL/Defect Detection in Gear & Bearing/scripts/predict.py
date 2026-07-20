"""
predict.py
==========
CLI inference tool for the Gear & Bearing defect detection model.
Accepts a spectrogram image path and outputs the predicted defect class.

Usage:
    python predict.py <path_to_spectrogram_image>
    python predict.py ../data/real_world_samples/ball_fault_noisy.png
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os


def predict_defect(image_path):
    """
    Run inference on a single spectrogram image using the domain-adapted model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class names — MUST match ImageFolder alphabetical order from training
    class_names = [
        'ball_fault', 'broken_tooth', 'healthy_gear',
        'inner_race_fault', 'normal_bearing', 'outer_race_fault'
    ]

    # Load model architecture
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Load domain-adapted weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, '..', 'models',
                                'gear_bearing_resnet50_domain_adapted.pth')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(script_dir, '..', 
                                    'gear_bearing_resnet50_domain_adapted.pth')

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Image transformation (matches training preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Run inference
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    predicted_class = class_names[preds[0]]
    conf_pct = confidence[0].item() * 100

    print(f"Predicted Defect : {predicted_class}")
    print(f"Confidence       : {conf_pct:.1f}%")
    print(f"All Probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls:20s}: {probabilities[0][i].item()*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_spectrogram_image>")
        print("Example: python predict.py ../data/real_world_samples/ball_fault_noisy.png")
    else:
        predict_defect(sys.argv[1])
